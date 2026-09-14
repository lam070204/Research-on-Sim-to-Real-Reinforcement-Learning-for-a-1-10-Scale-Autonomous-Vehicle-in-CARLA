# MAPVUONG BC -> REAL PPO ACTOR

Use this only after the oracle-student probe succeeds.

Current evidence:
- 9000 oracle demonstrations
- validation strong-turn error ~0.0028
- validation strong-turn sign accuracy 100%
- closed-loop student passed 4/4

This tool behavior-clones the SAME PPO actor architecture.

Steering target:
- successful oracle steer from `oracle_obs100_dataset.npz`

Speed target:
- deterministic speed output of the source PPO checkpoint on each same obs100
- this distills/preserves the learned PPO speed behavior while replacing steering

Default source:
`preTrained_models\ppo\automav5_rgb_mapvuong_stable035_v14_lookahead\ppo_policy_5_.pth`

Why policy_5:
It is the 150k low-noise checkpoint before the failed reward-only
ORACLE-TEACHER fine-tune.

Safety:
- source checkpoint is never overwritten
- old training_state is never modified
- output goes to a new model namespace:
  `automav5_rgb_mapvuong_stable035_bc_oracle_v1`

The script then performs deterministic closed-loop evaluation for 30 s on all
four safe spawns.

Install:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_bc_ppo_actor_mapvuong.ps1 `
  -Project "D:\Autonomous-Driving-PPO-V3-Clean"
```

Run:

```powershell
cd D:\Autonomous-Driving-PPO-V3-Clean

powershell -ExecutionPolicy Bypass -File .\run_bc_ppo_actor_mapvuong.ps1
```

Do not resume PPO training until this prints:

`BC PPO RESULT | passed=4/4`
