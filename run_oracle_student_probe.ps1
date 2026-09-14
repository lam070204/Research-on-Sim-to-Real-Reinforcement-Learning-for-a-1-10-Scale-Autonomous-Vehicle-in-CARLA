param(
    [string]$Project = "D:\Autonomous-Driving-PPO-V3-Clean",
    [string]$CarlaExe = "D:\CARLA_AUTOMAV3_MAPS\WindowsNoEditor\CarlaUE4.exe",
    [int]$Port = 2000,
    [double]$LookaheadM = 0.50,
    [double]$SpeedMps = 0.20,
    [double]$CollectSecondsPerSpawn = 45.0,
    [double]$EvalSecondsPerSpawn = 30.0
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================================"
Write-Host "MAPVUONG ORACLE -> STUDENT PROBE"
Write-Host "================================================================================"
Write-Host "PPO training      : OFF"
Write-Host "Student input     : obs100 only"
Write-Host "Map               : /Game/mapvuong/mapvuong"
Write-Host "Spawns            : 1,2,3,4"
Write-Host "Collect           : $CollectSecondsPerSpawn s/spawn"
Write-Host "Evaluate          : $EvalSecondsPerSpawn s/spawn"
Write-Host "Speed             : $SpeedMps m/s"
Write-Host "Lookahead         : $LookaheadM m"
Write-Host "Steer rate        : 3.0 /s"
Write-Host "DR / recovery     : OFF / OFF"
Write-Host "================================================================================"

if (-not (Test-Path $Project)) {
    throw "Project not found: $Project"
}
if (-not (Test-Path $CarlaExe)) {
    throw "CarlaUE4.exe not found: $CarlaExe"
}

$scriptPath = Join-Path $Project "oracle_student_probe.py"
if (-not (Test-Path $scriptPath)) {
    throw "Missing oracle_student_probe.py in project root."
}

Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Start-Process $CarlaExe `
    -ArgumentList "-carla-rpc-port=$Port -dx11 -windowed -ResX=480 -ResY=270 -nosound -NoVSync"

Start-Sleep -Seconds 12

Push-Location $Project
try {
    python .\oracle_student_probe.py `
        --host localhost `
        --port $Port `
        --map "/Game/mapvuong/mapvuong" `
        --lookahead-m $LookaheadM `
        --speed-mps $SpeedMps `
        --collect-seconds-per-spawn $CollectSecondsPerSpawn `
        --eval-seconds-per-spawn $EvalSecondsPerSpawn `
        --steer-rate 3.0 `
        --device cpu `
        --encoder-device cpu

    if ($LASTEXITCODE -ne 0) {
        throw "Oracle student probe failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
    Write-Host "Stopping CarlaUE4 instances..."
    Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
}
