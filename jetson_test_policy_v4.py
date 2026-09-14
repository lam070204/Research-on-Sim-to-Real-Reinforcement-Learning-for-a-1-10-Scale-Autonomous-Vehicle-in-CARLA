#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jetson Orin dry-run test:
REAL camera -> frozen VAE -> obs100 -> PPO deterministic -> print/log action.
NO motor/servo transmission.

serial-csv mode expects one line:
    speed_mps,yaw_rate_rad_s,ax_mps2\n
V4 domain randomization is NOT applied on REAL.
"""
from __future__ import print_function

import argparse
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from encoder_runtime_rgb_v3 import EncodeRGBV3
from observation_builder_rgb_v3 import ObservationBuilderRGBV3
from networks.on_policy.ppo.ppo_agent_v3 import PPOAgent

W, H = 160, 80


def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--model-name', default='automav3_rgb_v4_vision_dr')
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--camera', default='0')
    p.add_argument('--gstreamer', default=None)
    p.add_argument('--camera-width', type=int, default=640)
    p.add_argument('--camera-height', type=int, default=480)
    p.add_argument('--camera-fps', type=float, default=30.0)
    p.add_argument('--camera-timeout', type=float, default=0.5)
    p.add_argument('--sensor-mode', choices=['zero','constant','serial-csv'], default='zero')
    p.add_argument('--speed', type=float, default=0.0)
    p.add_argument('--yaw-rate', type=float, default=0.0)
    p.add_argument('--ax', type=float, default=0.0)
    p.add_argument('--serial-port', default='/dev/ttyUSB0')
    p.add_argument('--serial-baud', type=int, default=115200)
    p.add_argument('--sensor-timeout', type=float, default=0.20)
    p.add_argument('--control-hz', type=float, default=50.0)
    p.add_argument('--duration', type=float, default=60.0)
    p.add_argument('--print-every', type=int, default=25)
    p.add_argument('--speed-cap', type=float, default=1.0)
    p.add_argument('--steer-cap', type=float, default=1.0)
    p.add_argument('--display', action='store_true')
    p.add_argument('--log', default=None)
    return p.parse_args()


def inspect_checkpoint(path):
    ckpt = torch.load(str(path), map_location='cpu')
    if not isinstance(ckpt, dict):
        raise RuntimeError('Checkpoint không phải dict.')
    if ckpt.get('version') != 'PPO_V3_GD14':
        raise RuntimeError('Sai checkpoint version: {}'.format(ckpt.get('version')))
    if int(ckpt.get('obs_dim', -1)) != 100:
        raise RuntimeError('obs_dim != 100')
    if int(ckpt.get('action_dim', -1)) != 2:
        raise RuntimeError('action_dim != 2')
    if 'policy_state_dict' not in ckpt:
        raise RuntimeError('Checkpoint thiếu policy_state_dict')
    return ckpt


def crop_resize_rgb(bgr):
    h, w = bgr.shape[:2]
    aspect = float(w) / float(h)
    if aspect > 2.0:
        nw = int(round(2.0 * h)); x0 = (w - nw)//2
        bgr = bgr[:, x0:x0+nw]
    elif aspect < 2.0:
        nh = int(round(w / 2.0)); y0 = (h - nh)//2
        bgr = bgr[y0:y0+nh, :]
    bgr = cv2.resize(bgr, (W,H), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


class CameraLatest(object):
    def __init__(self, a):
        self.a=a; self.lock=threading.Lock(); self.stop=threading.Event()
        self.frame=None; self.seq=0; self.ts=0.0; self.err=None; self.cap=None
    def start(self):
        if self.a.gstreamer:
            self.cap=cv2.VideoCapture(self.a.gstreamer, cv2.CAP_GSTREAMER)
        else:
            src=self.a.camera
            try: src=int(src)
            except ValueError: pass
            self.cap=cv2.VideoCapture(src)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,self.a.camera_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,self.a.camera_height)
            self.cap.set(cv2.CAP_PROP_FPS,self.a.camera_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        if not self.cap.isOpened(): raise RuntimeError('Không mở được camera')
        threading.Thread(target=self._run,daemon=True).start()
    def _run(self):
        try:
            while not self.stop.is_set():
                ok,bgr=self.cap.read()
                if not ok or bgr is None:
                    time.sleep(0.005); continue
                rgb=crop_resize_rgb(bgr); now=time.monotonic()
                with self.lock:
                    self.frame=rgb; self.seq+=1; self.ts=now
        except Exception as e: self.err=e
    def get(self):
        if self.err: raise RuntimeError('Camera thread: {}'.format(self.err))
        with self.lock:
            if self.frame is None: return None,0,0.0
            return self.frame.copy(),self.seq,self.ts
    def close(self):
        self.stop.set()
        if self.cap is not None: self.cap.release()


class ConstantSensor(object):
    def __init__(self,s,y,a): self.s=float(s); self.y=float(y); self.a=float(a)
    def start(self): pass
    def get(self):
        return dict(speed_mps=self.s,yaw_rate_rad_s=self.y,longitudinal_accel_mps2=self.a,seq=0,age_s=0.0)
    def close(self): pass


class SerialCsvSensor(object):
    def __init__(self,port,baud):
        import serial
        self.ser=serial.Serial(port,baudrate=baud,timeout=0.05)
        self.lock=threading.Lock(); self.stop=threading.Event(); self.state=None; self.err=None; self.seq=0; self.ts=0.0
    def start(self): threading.Thread(target=self._run,daemon=True).start()
    def _run(self):
        try:
            while not self.stop.is_set():
                raw=self.ser.readline()
                if not raw: continue
                try:
                    p=raw.decode('ascii').strip().split(',')
                    if len(p)!=3: continue
                    s,y,a=map(float,p)
                    if not np.isfinite([s,y,a]).all(): continue
                    now=time.monotonic()
                    with self.lock:
                        self.seq+=1; self.ts=now
                        self.state=dict(speed_mps=max(0.0,s),yaw_rate_rad_s=y,longitudinal_accel_mps2=a,seq=self.seq)
                except Exception: continue
        except Exception as e: self.err=e
    def get(self):
        if self.err: raise RuntimeError('Serial thread: {}'.format(self.err))
        with self.lock:
            if self.state is None: return None
            d=dict(self.state); d['age_s']=time.monotonic()-self.ts; return d
    def close(self):
        self.stop.set()
        try: self.ser.close()
        except Exception: pass


def sync_cuda(device):
    if device=='cuda' and torch.cuda.is_available(): torch.cuda.synchronize()


def main():
    a=args_parse()
    ckpt_path=Path(a.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file(): raise FileNotFoundError(str(ckpt_path))
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA không khả dụng trên PyTorch hiện tại')
    if not (0.0 < a.speed_cap <= 1.0): raise ValueError('speed-cap phải (0,1]')
    if not (0.0 < a.steer_cap <= 1.0): raise ValueError('steer-cap phải (0,1]')

    meta=inspect_checkpoint(ckpt_path)
    print('='*90)
    print('JETSON ORIN PPO V4 DRY-RUN')
    print('checkpoint:',ckpt_path)
    print('device    :',a.device)
    print('control Hz:',a.control_hz)
    print('sensor    :',a.sensor_mode)
    print('MOTOR TX  : DISABLED')
    print('REAL DR   : OFF')
    print('='*90)

    encoder=EncodeRGBV3(device=a.device)
    builder=ObservationBuilderRGBV3(encoder=encoder)
    agent=PPOAgent(town=a.model_name,action_std_init=float(meta.get('action_std',0.05)),device=a.device)
    agent.load(checkpoint_path=str(ckpt_path))

    cam=CameraLatest(a); cam.start()
    if a.sensor_mode=='zero': sensor=ConstantSensor(0,0,0)
    elif a.sensor_mode=='constant': sensor=ConstantSensor(a.speed,a.yaw_rate,a.ax)
    else: sensor=SerialCsvSensor(a.serial_port,a.serial_baud)
    sensor.start()

    stamp=time.strftime('%Y%m%d_%H%M%S')
    log_path=Path(a.log) if a.log else Path('logs')/('jetson_policy_test_'+stamp+'.csv')
    log_path.parent.mkdir(parents=True,exist_ok=True)
    f=log_path.open('w',newline=''); wr=csv.writer(f)
    wr.writerow(['tick','time_s','avg_hz','cam_seq','cam_age_ms','sensor_seq','sensor_age_ms','speed','yaw','ax','prev_steer','prev_speed','raw_steer','raw_speed','logical_steer','logical_speed','infer_ms'])

    # Wait first data.
    deadline=time.monotonic()+5.0
    frame=None; st=None
    while time.monotonic()<deadline:
        frame,_,_=cam.get(); st=sensor.get()
        if frame is not None and st is not None: break
        time.sleep(0.01)
    if frame is None: raise RuntimeError('Không có camera frame')
    if st is None: raise RuntimeError('Không có sensor state')

    prev_steer=0.0; prev_speed=0.0
    # CUDA warmup.
    for _ in range(5):
        obs=builder.build(frame,st['speed_mps'],st['yaw_rate_rad_s'],st['longitudinal_accel_mps2'],prev_steer,prev_speed)
        _=agent.get_action(obs,train=False)
    sync_cuda(a.device)

    period=1.0/a.control_hz; start=time.monotonic(); next_t=start; tick=0; ema=None
    print('WARMUP PASS | log={}'.format(log_path))
    try:
        while True:
            if a.duration>0 and time.monotonic()-start>=a.duration: break
            now=time.monotonic()
            if now<next_t: time.sleep(next_t-now)
            t0=time.monotonic(); next_t+=period

            frame,cam_seq,cam_ts=cam.get()
            if frame is None: raise RuntimeError('Camera frame None')
            cam_age=t0-cam_ts
            if cam_age>a.camera_timeout: raise RuntimeError('Camera stale {:.3f}s'.format(cam_age))

            st=sensor.get()
            if st is None: raise RuntimeError('Sensor state None')
            s_age=float(st.get('age_s',0.0))
            if a.sensor_mode=='serial-csv' and s_age>a.sensor_timeout:
                raise RuntimeError('STM32 sensor stale {:.3f}s'.format(s_age))

            old_ps,old_pv=prev_steer,prev_speed
            sync_cuda(a.device); q0=time.perf_counter()
            obs=builder.build(frame,float(st['speed_mps']),float(st['yaw_rate_rad_s']),float(st['longitudinal_accel_mps2']),old_ps,old_pv)
            action=agent.get_action(obs,train=False)
            sync_cuda(a.device); infer_ms=(time.perf_counter()-q0)*1000.0
            ema=infer_ms if ema is None else 0.95*ema+0.05*infer_ms

            raw_steer=float(action[0]); raw_speed=float(action[1])
            logical_steer=float(np.clip(raw_steer,-a.steer_cap,a.steer_cap))
            logical_speed=float(np.clip(raw_speed,0.0,a.speed_cap))
            # In real drive these two must become the logical commands actually sent to STM32.
            prev_steer,prev_speed=logical_steer,logical_speed

            tick+=1; elapsed=time.monotonic()-start; hz=tick/max(elapsed,1e-9)
            wr.writerow([tick,'{:.6f}'.format(elapsed),'{:.3f}'.format(hz),cam_seq,'{:.3f}'.format(cam_age*1000),st.get('seq',0),'{:.3f}'.format(s_age*1000),'{:.6f}'.format(st['speed_mps']),'{:.6f}'.format(st['yaw_rate_rad_s']),'{:.6f}'.format(st['longitudinal_accel_mps2']),'{:.6f}'.format(old_ps),'{:.6f}'.format(old_pv),'{:.6f}'.format(raw_steer),'{:.6f}'.format(raw_speed),'{:.6f}'.format(logical_steer),'{:.6f}'.format(logical_speed),'{:.3f}'.format(infer_ms)])
            if tick%50==0: f.flush()

            if a.print_every>0 and tick%a.print_every==0:
                print('[JETSON] tick={:06d} avgHz={:5.1f} camAge={:5.1f}ms speed={:.3f} yaw={:+.4f} ax={:+.4f} PPO=({:+.3f},{:.3f}) logical=({:+.3f},{:.3f}) infer={:.2f}ms ema={:.2f}ms'.format(tick,hz,cam_age*1000,float(st['speed_mps']),float(st['yaw_rate_rad_s']),float(st['longitudinal_accel_mps2']),raw_steer,raw_speed,logical_steer,logical_speed,infer_ms,ema))

            if a.display:
                bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                bgr=cv2.resize(bgr,(640,320),interpolation=cv2.INTER_NEAREST)
                cv2.putText(bgr,'steer={:+.3f} speed={:.3f}'.format(logical_steer,logical_speed),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
                cv2.imshow('PPO REAL 160x80',bgr)
                if (cv2.waitKey(1)&0xff) in (27,ord('q')): break
    except KeyboardInterrupt:
        print('\nStopped by user')
    finally:
        f.flush(); f.close(); cam.close(); sensor.close(); cv2.destroyAllWindows()

    print('='*90)
    print('DRY-RUN COMPLETE | no motor command sent')
    print('log:',log_path)
    print('='*90)

if __name__=='__main__': main()
