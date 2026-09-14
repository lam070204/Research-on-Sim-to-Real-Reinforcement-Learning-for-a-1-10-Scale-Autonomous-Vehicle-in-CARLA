#!/usr/bin/env python3
from __future__ import print_function
import argparse, sys
from pathlib import Path
import torch, cv2, numpy as np

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--camera',default='0'); a=p.parse_args()
    print('='*80); print('JETSON RUNTIME CHECK')
    print('Python:',sys.version.replace('\n',' ')); print('Torch:',torch.__version__); print('CUDA:',torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
    print('OpenCV:',cv2.__version__); print('NumPy:',np.__version__)
    cp=Path(a.checkpoint).expanduser().resolve(); ck=torch.load(str(cp),map_location='cpu')
    print('checkpoint:',cp); print('version:',ck.get('version'),'obs:',ck.get('obs_dim'),'action:',ck.get('action_dim'))
    if ck.get('version')!='PPO_V3_GD14' or int(ck.get('obs_dim',-1))!=100 or int(ck.get('action_dim',-1))!=2: raise RuntimeError('Checkpoint contract FAIL')
    src=a.camera
    try: src=int(src)
    except ValueError: pass
    cap=cv2.VideoCapture(src); ok,frame=cap.read(); cap.release()
    if not ok or frame is None: raise RuntimeError('Camera FAIL')
    print('Camera PASS shape={}'.format(frame.shape)); print('RESULT: JETSON RUNTIME CHECK PASS'); print('='*80)
if __name__=='__main__': main()
