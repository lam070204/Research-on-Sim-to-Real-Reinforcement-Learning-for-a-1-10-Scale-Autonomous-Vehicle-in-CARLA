# MAPVUONG ORACLE CORNER TEST

This is a diagnostic, not PPO training.

It bypasses PPO and drives the exact STABLE035 environment using a privileged
CARLA-waypoint pure-pursuit controller.

Default test:
- map: `/Game/mapvuong/mapvuong`
- spawns: 1,2,3,4
- speed command: 0.20 m/s
- lookahead: 0.50 m
- steer rate: 3.0 /s
- episode: 30 s per spawn
- DR OFF
- recovery OFF

PASS = the oracle survives until time limit.
FAIL = offroad/collision/stuck/overspeed before time limit.

Interpretation:
- 4/4 PASS: physics, map geometry and steering-rate limit are capable; focus on RL/reward/observation.
- any FAIL: fix steering authority/controller geometry/map topology before more PPO training.

Output:
- `oracle_results/mapvuong_oracle_<timestamp>/oracle_summary.csv`
- `oracle_results/mapvuong_oracle_<timestamp>/oracle_steps.csv`

Install:
```powershell
powershell -ExecutionPolicy Bypass -File .\install_oracle_mapvuong_corner_test.ps1 `
  -Project "D:\Autonomous-Driving-PPO-V3-Clean"
```

Run:
```powershell
cd D:\Autonomous-Driving-PPO-V3-Clean

powershell -ExecutionPolicy Bypass -File .\run_oracle_mapvuong_corner_test.ps1
```
