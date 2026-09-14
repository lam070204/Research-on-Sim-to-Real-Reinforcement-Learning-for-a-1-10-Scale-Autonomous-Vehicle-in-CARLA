param(
    [string]$Project = "D:\Autonomous-Driving-PPO-V3-Clean"
)

$ErrorActionPreference = "Stop"
$Here = $PSScriptRoot

if (-not (Test-Path $Project)) {
    throw "Project not found: $Project"
}

$rootFiles = @(
    "train_ppo_rgb_v5_singlemap_stable.py",
    "reward_function_v5_singlemap_stable.py",
    "reward_manager_v5_singlemap_stable.py",
    "domain_randomization_v5_singlemap_stable.py",
    "stable_action_adapter_v5.py",
    "run_v5_singlemap_stable.ps1"
)

foreach ($name in $rootFiles) {
    $src = Join-Path $Here $name
    $dst = Join-Path $Project $name

    if (-not (Test-Path $src)) {
        throw "Package file missing: $src"
    }

    Copy-Item $src $dst -Force
    Write-Host "COPIED | $dst"
}

$simDir = Join-Path $Project "simulation"
if (-not (Test-Path $simDir)) {
    throw "simulation folder not found: $simDir"
}

$envSrc = Join-Path $Here "simulation\carla_environment_rgb_v5_singlemap_stable.py"
$envDst = Join-Path $simDir "carla_environment_rgb_v5_singlemap_stable.py"

Copy-Item $envSrc $envDst -Force
Write-Host "COPIED | $envDst"

Write-Host ""
Write-Host "Syntax check..."
Push-Location $Project
try {
    python -m py_compile .\train_ppo_rgb_v5_singlemap_stable.py
    python -m py_compile .\reward_function_v5_singlemap_stable.py
    python -m py_compile .\reward_manager_v5_singlemap_stable.py
    python -m py_compile .\domain_randomization_v5_singlemap_stable.py
    python -m py_compile .\stable_action_adapter_v5.py
    python -m py_compile .\simulation\carla_environment_rgb_v5_singlemap_stable.py

    if ($LASTEXITCODE -ne 0) {
        throw "Python syntax check failed."
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "INSTALL PASS"
Write-Host "Run:"
Write-Host "  cd $Project"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_v5_singlemap_stable.ps1 -SmokeOnly"
Write-Host "Then fresh train:"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_v5_singlemap_stable.ps1"
