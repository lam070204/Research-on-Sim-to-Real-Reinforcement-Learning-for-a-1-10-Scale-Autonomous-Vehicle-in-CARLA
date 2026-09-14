# PPO V5 SINGLE-MAP STABLE035 — mapvuong

Mục tiêu của bản này là tách hẳn bài toán "ổn định xe thật" khỏi multi-map và Full-DR mạnh.

## Cấu hình chính

- Map duy nhất: `/Game/mapvuong/mapvuong`
- PPO vẫn 2 action: `[steer, speed]`
- Observation vẫn 100: `latent95 + speed + yaw + ax + prevSteer + prevSpeed`
- HARD MAX speed: `0.35 m/s`
- Reward tốc độ theo đúng tỷ lệ repo gốc:
  - MIN = `0.35 * 15/25 = 0.210 m/s`
  - TARGET = `0.35 * 22/25 = 0.308 m/s`
  - MAX = `0.350 m/s`
- Reward chính: `speed_factor * center_factor * heading_factor`
- Recovery injection: OFF
- Spawn perturbation: OFF
- checkpoint mỗi 25,000 transition
- 2 CARLA worker cùng một map để train nhanh, không có negative transfer giữa map.

## Chống đánh lái đảo mạnh

Bản này không thêm một reward steering phức tạp.

Thay vào đó, command steering được giới hạn ở command-space trước Dynamics DR:

- steer deadband = `0.015`
- steer slew/rate limit = `3.0 /s`
- ở 50 Hz: tối đa khoảng `0.06` steer mỗi tick

Speed command cũng được slew ở `0.50 m/s²`.

Điểm quan trọng: lọc command nằm TRƯỚC dynamics randomization, nên `prev_steer` / `prev_speed` trong obs100 vẫn là command logic mà xe thật cần nhận. Khi deploy Jetson phải dùng cùng transform.

## Speed action không bị hard-clip sai distribution

Actor V5 vẫn output speed chuẩn hóa `[0,1]`.

Environment đổi tuyến tính:

`speed_cmd_mps = action[1] * 0.35`

Không phải `clip(action[1], 0, 0.35)`.

Jetson runtime cũng phải đổi giống vậy.

## Controlled Domain Randomization

Không dùng 75–100% Full-DR ngay từ đầu.

Schedule mặc định 1M step:

1. `0–100k`: CLEAN, 0% DR.
2. `100k–300k`: VISION_MILD, 25% episode randomized.
3. `300k–600k`: VISION_OR_DYNAMICS, 35%.
4. `600k–900k`: CONTROLLED_ONE_FAMILY, 45%.
5. `900k+`: NOMINAL_POLISH, chỉ 20% randomized.

Quan trọng nhất: `dr_family_budget=1`.

Một randomized episode chỉ nhận tối đa MỘT nhóm:
- dynamics, hoặc
- vision/weather, hoặc
- sensor.

Không đổ dynamics + vision + sensor + delay vào cùng một episode.

## DR range đã giảm

Dynamics:
- torque ±3%
- brake ±5%
- steer gain đối xứng ±3%
- extra command delay DR = 0

Vision:
- camera pose nhỏ hơn
- brightness/contrast ±6%
- gamma ±3%
- RGB gain ±1.5%
- Gaussian noise tối đa 1.5/255
- blur 2%
- occlusion OFF
- weather nhẹ hơn

Sensor:
- bias/noise nhỏ hơn đáng kể.

Lý do: camera CARLA 30 FPS và control 50 Hz vốn đã có lag 1–3 tick. Không thêm delay ngẫu nhiên trước khi đo được latency xe thật.

## Failure zones không được tự động oversample

Trainer ghi cluster vị trí failure theo tọa độ CARLA.

Trạng thái:
- `PERSISTENT_CLEAN`: fail lặp cả nominal => nghi geometry/visual/policy.
- `DR_SENSITIVE`: chỉ fail khi DR => range DR hoặc representation quá mạnh.
- `REPEATED`: cần theo dõi.

Failure zone chỉ là DIAGNOSTIC.
Nó KHÔNG tự boost sampling/reward.

Điều này tránh việc một đoạn hỏng kéo gradient và làm policy toàn map bị lệch.

## Cài đặt

Giải nén package rồi:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_v5_singlemap_stable.ps1 `
  -Project "D:\Autonomous-Driving-PPO-V3-Clean"
```

Smoke test:

```powershell
cd D:\Autonomous-Driving-PPO-V3-Clean
powershell -ExecutionPolicy Bypass -File .\run_v5_singlemap_stable.ps1 -SmokeOnly
```

Fresh train:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v5_singlemap_stable.ps1
```

Resume:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_v5_singlemap_stable.ps1 -Resume
```

Checkpoint namespace:

`preTrained_models/ppo/automav5_rgb_mapvuong_stable035/`

## Những file KHÔNG đổi

- `encoder_runtime_rgb_v5.py`
- `observation_builder_rgb_v5.py`
- VAE latent95
- `networks/on_policy/ppo/actor_critic_v5.py`
- `networks/on_policy/ppo/ppo_agent_v5.py`
- `action_controller_v5.py`
- `vehicle_specs_v5.py`

Bản stable dùng file riêng nên không đè V5 hiện tại.

## Để hoàn thiện Sim-to-Real

Cần sửa runtime Jetson để giống training:

1. PPO speed output `[0,1]` phải nhân `speed_cap`, không hard clip.
2. steer deadband giống training.
3. steer slew/rate limit giống training.
4. prevSteer/prevSpeed trong obs100 phải là command thực sự gửi STM32 ở tick trước.
5. Không thêm synthetic sensor/vision noise trên xe thật.
6. Chỉ thêm latency DR sau khi đo latency camera→Jetson→STM32 thật.

Để patch phần real chính xác, gửi bản HIỆN TẠI của:
- `jetson_run_policy_v5_*.py`
- firmware STM32 phần parse `REALCMD` + servo/speed control
- log real có: timestamp, PPO steer, steer gửi STM32, servo feedback, PPO speed, speed command, actual speed, gyro_z.


## Fix v1.1
Fixed missing `CarlaEnvironmentRGBV5SingleMapStable` export used by the trainer import.


## Fix v1.2
Fixed smoke-start `KeyError: 'stable'` in `verify_ready_workers()`.
The STABLE035 reward profile has only `min`, `target`, and `max`; recovery-state
names are diagnostic labels and must not be used as reward-profile keys.


# v1.3 CLEAN MASTER

This version is deliberately a controlled experiment.

Changes from v1.2:
- Fresh namespace: `automav5_rgb_mapvuong_stable035_v13_cleanmaster`
- ONE map only: `/Game/mapvuong/mapvuong`
- Hard max speed remains 0.35 m/s
- Reward structure remains `speed_factor * center_factor * heading_factor`
- Center reward support widened from 0.10 m to 0.18 m
- Heading reward support widened from 20 deg to 45 deg
- DR is HARD OFF for the entire CLEAN MASTER run
- Action std is fixed at 0.15; no exploration decay
- Steering rate limit remains 3.0/s and deadband remains 0.015
- Steering diagnostic history resets at episode boundaries

Do NOT resume the v1.2 checkpoint for this experiment.
Run fresh so the effect of the reward-support change is interpretable.

Recommended:
1. Smoke test.
2. Fresh 100k.
3. Inspect offroad rate and persistent failure zones.
4. Only if clean performance materially improves, continue to 200k.
5. Do not add DR until the nominal map is mastered reliably.

## Fix v1.3.1
Fixed PowerShell parser error caused by an accidental duplicated UTF-8 BOM before `param(`.
`run_v5_singlemap_stable.ps1` now starts with `param(` as the first bytes/tokens.


# v1.4 CORNER LOOKAHEAD

Controlled experiment after v1.3 showed excellent straight-line behavior but
persistent clean offroad failures at the square corners.

Only one task-level change:
- The heading term in reward now compares vehicle yaw with the bearing to a
  Driving waypoint 0.50 m ahead (pure-pursuit style), instead of only the lane
  tangent directly under the vehicle.
- PPO observation remains exactly 100 dimensions and gets NO waypoint,
  lookahead, lateral error, or heading error.
- Lateral reward, wheel offroad termination, progress metric, action space,
  speed cap, steering rate limit, deadband, VAE and PPO architecture remain
  unchanged.
- DR remains HARD OFF.
- action std remains fixed at 0.15.

New diagnostics:
- headingNow vs headingPreview
- raw PPO steer vs applied steer
- raw steering sign flip rate
- steering rate-limiter hit percentage
- applied steering reset-safe delta metrics

Fresh namespace:
`automav5_rgb_mapvuong_stable035_v14_lookahead`

Procedure:
1. Smoke only.
2. Fresh 100k.
3. Do NOT enable DR.
4. Evaluate whether offroad rate drops and whether persistent corner zones
   become less frequent/older.
5. If offroad remains 100%, next experiment is steering authority / lookahead
   distance, not more reward terms or DR.


# v1.4.1 PROBEFIX

The v1.4 training log showed `headingPreview=0`, `lookahead=0`,
`STEER RAW=0`, and `limiterHit=0`. The underlying per-step diagnostics were
being collected inside each worker, but the new v1.4 fields were accidentally
omitted from `RolloutDiagnostics.export()` and `aggregate_diagnostics()`.

This version fixes diagnostics only. It intentionally keeps the SAME model
namespace as v1.4 so the existing 100k checkpoint can be loaded.

Use:
`.\run_v5_singlemap_stable.ps1 -Resume -SmokeOnly -ProbeStepsPerWorker 1500`

This collects 1500 steps per worker from the existing 100k policy, performs
NO PPO update, saves NO checkpoint, and makes NO training-state change.

Decision:
- If `previewValid` is high and headingPreview is nonzero, the v1.4 reward
  lookahead was actually active; diagnose lookahead distance / steering authority.
- If `previewValid` is near 0%, CARLA `Waypoint.next()` is not giving a usable
  path on this custom OpenDRIVE map; build the preview from the map route/topology
  instead.


# v1.4.2 PROBECOMPAT

Fixes resume compatibility for the diagnostic-only probe.

The existing 100k state was created by:
`TRAINER_PPO_V5_SINGLEMAP_STABLE035_1_4_CORNER_LOOKAHEAD`

The v1.4.1 diagnostic patch used a newer trainer-version string and the strict
state-version guard rejected the old state before the probe could start.

v1.4.2 explicitly accepts:
- `..._1_4_CORNER_LOOKAHEAD`
- `..._1_4_1_PROBEFIX`
- the current v1.4.2 probe version

The model namespace is unchanged, so `-Resume -SmokeOnly` loads the exact
existing v1.4 100k policy/state.

No reward, action, controller, PPO update, or DR behavior is changed.


# v1.4.3 LONGPROBE

Fixes the PowerShell runner so `-ProbeStepsPerWorker` is actually passed to
the trainer as `--smoke-steps-per-worker`.

Previously the runner printed `Probe/worker: 1500`, but the trainer still used
its default 32 samples/worker, which is why the output showed `rollout total: 64`.

This patch does not change reward, PPO, controller, DR, model namespace, or
training state. It is probe-only.

Use:
`.\run_v5_singlemap_stable.ps1 -Resume -SmokeOnly -ProbeStepsPerWorker 1500`


# v1.4.4 DETERMINISTIC PROBE

Purpose: distinguish learned actor-mean behavior from stochastic exploration
noise before changing the steering rate limiter.

The existing v1.4 policy was trained with action std 0.15. The long probe
showed:
- previewValid = 100%
- lookahead ~= 0.54 m
- rawFlip ~= 7.35%
- limiterHit ~= 71.9%
- applied signFlip = 0%

Those raw metrics include exploration sampling. This patch adds a smoke-only
`--probe-action-std` override. Default PowerShell probe value is 0.0001,
which approximates deterministic actor-mean behavior while preserving the
existing PPO API.

It does NOT modify:
- training state
- checkpoint
- reward
- controller
- DR
- model namespace
- learned weights

Recommended:
`.\run_v5_singlemap_stable.ps1 -Resume -SmokeOnly -ProbeStepsPerWorker 1500 -ProbeActionStd 0.0001`


# v1.5 LOW-NOISE CONTINUE

Decision from the deterministic v1.4 probe:
- lookahead is valid (100%)
- deterministic actor mean is smooth
- deterministic rawFlip = 0%
- deterministic limiterHit ~= 0.2%
- but the policy still goes offroad at corners
- stochastic std=0.15 caused rawFlip ~= 7.35% and limiterHit ~= 71.9%

Interpretation:
The 3.0/s steering slew limiter is NOT the limiting factor for the learned
actor mean. The large limiter-hit rate during training mostly comes from
Gaussian exploration noise.

This experiment changes ONE training variable:
- action std: 0.15 -> 0.05

Why 0.05:
For independent Gaussian samples, the expected absolute sample-to-sample
noise difference is approximately `2*sigma/sqrt(pi)`.
At sigma=0.05 this is about 0.056, close to the physical adapter limit of
0.06 steering units/tick at 50 Hz. This greatly reduces exploration samples
that are immediately clipped by the actuator adapter while retaining useful
exploration.

Unchanged:
- same v1.4 checkpoint / same namespace
- same lookahead = 0.50 m
- same steer rate = 3.0/s
- same deadband = 0.015
- same speed max = 0.35 m/s
- same reward support = center 0.18 m / heading 45 deg
- DR HARD OFF
- recovery OFF

Recommended controlled continuation:
`.\run_v5_singlemap_stable.ps1 -Resume -TotalSteps 150000`

This resumes from step 100000 and trains only +50000 steps.

Then evaluate actor mean without learning:
`.\run_v5_singlemap_stable.ps1 -Resume -SmokeOnly -ProbeStepsPerWorker 1500 -ProbeActionStd 0.0001`

Do not increase steering rate and do not enable DR during this experiment.


# v1.5.1 STARTUPFIX

Fixes a startup-only NameError in v1.5.

Cause:
`print_startup()` tried to read `master.action_std`, but `master` is local to
`main()` and is not in the scope of `print_startup()`.

Fix:
`main()` now passes the actual loaded/overridden action std explicitly as
`current_action_std`.

No PPO update, reward, controller, lookahead, DR, speed, or steering behavior
was changed.

The failed v1.5 run stopped before rollout collection, so the existing 100k
checkpoint/state remains the correct resume source.

Run:
`.\run_v5_singlemap_stable.ps1 -Resume -TotalSteps 150000`


# v1.7 ORACLE-TEACHER

The privileged CARLA pure-pursuit oracle passed all 4/4 mapvuong spawns with:
- speed ~= 0.192-0.193 m/s
- max lateral error ~= 0.041-0.049 m
- required/applied steer max ~= 0.331-0.344
- steer-rate limiter hit ~= 0-0.3%

Therefore:
- CARLA physics is capable
- map topology is usable
- 3.0/s steering slew is capable
- the remaining failure is the learned policy/reward/perception path

Instead of another small reward-threshold tweak, v1.7 adds a direct
TRAIN-ONLY oracle steering teacher.

The PPO observation remains exactly obs100:
`latent95 + speed + yaw + ax + prevSteer + prevSpeed`.

No waypoint, lateral error, heading error, or oracle steer is added to the PPO
observation. The teacher exists only in reward shaping during CARLA training.

Reward:
`speed_factor * center_factor * direction_factor`

where:
`direction_factor = 0.5 * heading_factor + 0.5 * teacher_factor`

and:
`teacher_factor = clip(1 - abs(applied_steer - oracle_steer)/0.35, 0, 1)`

This directly rewards the policy for producing the ~0.33 steering authority
that the successful oracle actually needs around the square corners.

Unchanged:
- resume existing 150k checkpoint
- lookahead = 0.50 m
- action std = 0.05
- steer rate = 3.0/s
- deadband = 0.015
- speed max = 0.35 m/s
- DR HARD OFF
- recovery OFF
- PPO/VAE architecture unchanged

Fast test:
`.\run_v5_singlemap_stable.ps1 -Resume -TotalSteps 170000`

Only +20k transitions.

New diagnostic:
`ORACLE TEACH | |target|=... | err=... | factor=... | strong n=... err=...`

Success criterion:
- offroad rate starts dropping below 100%, OR
- strong-turn teacher error falls materially and deterministic evaluation begins
  surviving corners.

If +20k shows no movement, stop. Do not keep extending training blindly.
