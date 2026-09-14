param(
    [string]$Project = "D:\Autonomous-Driving-PPO-V3-Clean",
    [string]$CarlaExe = "D:\CARLA_AUTOMAV3_MAPS\WindowsNoEditor\CarlaUE4.exe",
    [string]$SourceCheckpoint = "preTrained_models\ppo\automav5_rgb_mapvuong_stable035_v14_lookahead\ppo_policy_5_.pth",
    [string]$DatasetPath = "",
    [int]$Port = 2000
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Project)) {
    throw "Project not found: $Project"
}

$scriptPath = Join-Path $Project "bc_ppo_actor_mapvuong.py"
if (-not (Test-Path $scriptPath)) {
    throw "Missing bc_ppo_actor_mapvuong.py in project root."
}

$sourceFull = Join-Path $Project $SourceCheckpoint
if (-not (Test-Path $sourceFull)) {
    throw "Source checkpoint not found: $sourceFull"
}

if ([string]::IsNullOrWhiteSpace($DatasetPath)) {
    $candidate = Get-ChildItem `
        -Path (Join-Path $Project "oracle_student_results") `
        -Filter "oracle_obs100_dataset.npz" `
        -Recurse `
        -File `
        -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if ($null -eq $candidate) {
        throw "Could not auto-find oracle_obs100_dataset.npz"
    }

    $datasetFull = $candidate.FullName
}
else {
    if ([System.IO.Path]::IsPathRooted($DatasetPath)) {
        $datasetFull = $DatasetPath
    }
    else {
        $datasetFull = Join-Path $Project $DatasetPath
    }
}

if (-not (Test-Path $datasetFull)) {
    throw "Dataset not found: $datasetFull"
}

Write-Host "================================================================================"
Write-Host "MAPVUONG BEHAVIOR CLONE -> REAL PPO ACTOR"
Write-Host "================================================================================"
Write-Host "Source checkpoint : $sourceFull"
Write-Host "Dataset           : $datasetFull"
Write-Host "Destination model : automav5_rgb_mapvuong_stable035_bc_oracle_v1"
Write-Host "Source is modified: NO"
Write-Host "PPO training      : NO"
Write-Host "================================================================================"

Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Start-Process $CarlaExe `
    -ArgumentList "-carla-rpc-port=$Port -dx11 -windowed -ResX=480 -ResY=270 -nosound -NoVSync"

Start-Sleep -Seconds 12

Push-Location $Project
try {
    python .\bc_ppo_actor_mapvuong.py `
        --project-root "$Project" `
        --dataset "$datasetFull" `
        --source-checkpoint "$sourceFull" `
        --dest-model-name "automav5_rgb_mapvuong_stable035_bc_oracle_v1" `
        --host localhost `
        --port $Port `
        --map "/Game/mapvuong/mapvuong" `
        --epochs 100 `
        --batch-size 512 `
        --lr 0.0002 `
        --strong-weight 4.0 `
        --speed-distill-weight 2.0 `
        --eval-seconds-per-spawn 30 `
        --device cpu

    if ($LASTEXITCODE -ne 0) {
        throw "BC PPO actor tool failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    Write-Host "Stopping CarlaUE4 instances..."
    Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
}
