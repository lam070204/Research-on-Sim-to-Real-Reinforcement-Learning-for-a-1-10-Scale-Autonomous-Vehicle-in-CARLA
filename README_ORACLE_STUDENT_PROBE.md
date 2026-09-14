# MAPVUONG ORACLE -> STUDENT PROBE

Use this after the oracle controller passes mapvuong 4/4 but reward-only PPO
still fails the corners.

This is **not PPO training**.

It answers one decisive question:

> Can the current `obs100 = latent95 + proprio5` directly predict the successful
> oracle steering action?

The probe:

1. Drives with the successful privileged pure-pursuit oracle.
2. Collects `obs100 -> oracle steer` demonstrations.
3. Trains a small MLP student using only obs100.
4. Runs that student closed-loop on all four safe spawns.

Default runtime is a few minutes.

Interpretation:

- `STUDENT RESULT | passed=4/4`
  - latent/observation contains enough corner information.
  - Next: behavior-clone the real PPO actor from oracle data, then short PPO fine-tune.

- Low supervised error but closed-loop fails
  - covariate shift.
  - Next: DAgger / teacher correction.

- High strong-turn validation error
  - latent95/current observation does not expose the corner sufficiently.
  - Next: inspect VAE corner separability, temporal/frame stacking, or encoder training.

Install:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_oracle_student_probe.ps1 `
  -Project "D:\Autonomous-Driving-PPO-V3-Clean"
```

Run:

```powershell
cd D:\Autonomous-Driving-PPO-V3-Clean

powershell -ExecutionPolicy Bypass -File .\run_oracle_student_probe.ps1
```

No PPO checkpoint or training-state file is modified.
