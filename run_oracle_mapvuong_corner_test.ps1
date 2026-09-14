param(
    [string]$Project = "D:\Autonomous-Driving-PPO-V3-Clean",
    [string]$CarlaExe = "D:\CARLA_AUTOMAV3_MAPS\WindowsNoEditor\CarlaUE4.exe",
    [int]$Port = 2000,
    [double]$LookaheadM = 0.50,
    [double]$SpeedMps = 0.20,
    [double]$EpisodeSeconds = 30.0,
    [double]$SteerRate = 3.0
)

$ErrorActionPreference = "Stop"

Write-Host "======================================================================"
Write-Host "MAPVUONG ORACLE CORNER TEST"
Write-Host "======================================================================"
Write-Host "PPO             : BYPASSED"
Write-Host "Map             : /Game/mapvuong/mapvuong"
Write-Host "Spawns          : 1,2,3,4"
Write-Host "Lookahead       : $LookaheadM m"
Write-Host "Speed           : $SpeedMps m/s"
Write-Host "Steer rate      : $SteerRate /s"
Write-Host "Episode         : $EpisodeSeconds s/spawn"
Write-Host "DR / recovery   : OFF / OFF"
Write-Host "======================================================================"

if (-not (Test-Path $Project)) {
    throw "Project not found: $Project"
}
if (-not (Test-Path $CarlaExe)) {
    throw "CarlaUE4.exe not found: $CarlaExe"
}

$scriptPath = Join-Path $Project "oracle_mapvuong_corner_test.py"
if (-not (Test-Path $scriptPath)) {
    throw "Missing oracle_mapvuong_corner_test.py in project root."
}

Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "START CARLA | port=$Port"
Start-Process $CarlaExe `
    -ArgumentList "-carla-rpc-port=$Port -dx11 -windowed -ResX=480 -ResY=270 -nosound -NoVSync"

Start-Sleep -Seconds 12

Push-Location $Project
try {
    python .\oracle_mapvuong_corner_test.py `
        --host localhost `
        --port $Port `
        --map "/Game/mapvuong/mapvuong" `
        --lookahead-m $LookaheadM `
        --speed-mps $SpeedMps `
        --episode-seconds $EpisodeSeconds `
        --steer-rate $SteerRate

    if ($LASTEXITCODE -ne 0) {
        throw "Oracle test failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    Write-Host "Stopping CarlaUE4 instances..."
    Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
}
