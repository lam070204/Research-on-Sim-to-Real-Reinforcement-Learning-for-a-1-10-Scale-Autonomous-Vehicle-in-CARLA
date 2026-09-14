param(
    [int]$TestSteps = 3500,
    [string]$ModelName = "automav3_rightturn35_multilightvae_v1"
)

$ErrorActionPreference = "Stop"
$presets = 0..15
$spawns = 1..4

foreach ($preset in $presets) {
    Write-Host ""
    Write-Host "===== LIGHT PRESET $preset ====="

    python .\set_carla_lighting.py --preset $preset
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to set lighting preset $preset"
    }

    Start-Sleep -Seconds 1

    foreach ($spawn in $spawns) {
        Write-Host ""
        Write-Host "--- preset=$preset | spawn=$spawn ---"

        python .\train_ppo_rgb_v4_rightturn35_safe.py `
          --train false `
          --load-checkpoint true `
          --test-timesteps $TestSteps `
          --curriculum false `
          --desired-speed 0.35 `
          --policy-speed-cap 0.35 `
          --dynamics-dr false `
          --camera-pose-dr false `
          --weather-dr false `
          --image-dr false `
          --sensor-noise-dr false `
          --safe-spawns $spawn `
          --model-name $ModelName `
          --print-every-steps 100

        if ($LASTEXITCODE -ne 0) {
            throw "Eval failed at preset=$preset spawn=$spawn"
        }
    }
}

Write-Host ""
Write-Host "MULTI-LIGHT EVAL COMPLETE"
