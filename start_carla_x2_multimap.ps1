param(
    [string]$CarlaExe = "D:\CARLA_AUTOMAV3_MAPS\WindowsNoEditor\CarlaUE4.exe",
    [string]$Map0 = "/Game/maptotrai/mapto_trai",
    [string]$Map1 = "/Game/maptophai/mapto_phai"
)

$ErrorActionPreference = "Stop"

# ============================================================
# PROJECT / PYTHON
# ============================================================

$ProjectRoot = $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host "PROJECT ROOT : $ProjectRoot" -ForegroundColor DarkGray
Write-Host "PYTHON       : $PythonExe" -ForegroundColor DarkGray
Write-Host "CARLA        : $CarlaExe" -ForegroundColor DarkGray
Write-Host ""

if (-not (Test-Path $CarlaExe)) {
    throw "CARLA executable not found: $CarlaExe"
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment Python not found: $PythonExe"
}

# Verify Python is actually runnable.
Write-Host "Checking project Python..." -ForegroundColor Yellow
& $PythonExe --version

if ($LASTEXITCODE -ne 0) {
    throw "Project .venv Python is broken. Recreate .venv on this machine."
}

# ============================================================
# CARLA X2
# ============================================================

$ResX = 480
$ResY = 270

Write-Host "Starting CARLA worker 0 on port 2000..." -ForegroundColor Cyan

Start-Process $CarlaExe `
    -ArgumentList "-carla-rpc-port=2000 -dx11 -windowed -ResX=$ResX -ResY=$ResY -WinX=0 -WinY=0 -nosound -NoVSync"

Start-Sleep -Seconds 3

Write-Host "Starting CARLA worker 1 on port 2010..." -ForegroundColor Cyan

Start-Process $CarlaExe `
    -ArgumentList "-carla-rpc-port=2010 -dx11 -windowed -ResX=$ResX -ResY=$ResY -WinX=520 -WinY=0 -nosound -NoVSync"

Write-Host "Waiting for both CARLA servers..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

# ============================================================
# LOAD INITIAL MAPS
# ============================================================

Write-Host "Loading anchor map on worker 0..." -ForegroundColor Cyan

& $PythonExe -c "from simulation.carla_connection_v5 import carla; c=carla.Client('localhost',2000); c.set_timeout(180); w=c.load_world('$Map0'); print('W0 MAP:',w.get_map().name)"

if ($LASTEXITCODE -ne 0) {
    throw "Worker 0 failed to load map: $Map0"
}

Write-Host "Loading anchor map on worker 1..." -ForegroundColor Cyan

& $PythonExe -c "from simulation.carla_connection_v5 import carla; c=carla.Client('localhost',2010); c.set_timeout(180); w=c.load_world('$Map1'); print('W1 MAP:',w.get_map().name)"

if ($LASTEXITCODE -ne 0) {
    throw "Worker 1 failed to load map: $Map1"
}

# ============================================================
# READY
# ============================================================

Write-Host ""
Write-Host "CARLA X2 MULTIMAP READY" -ForegroundColor Green
Write-Host "  worker 0 -> localhost:2000 -> $Map0"
Write-Host "  worker 1 -> localhost:2010 -> $Map1"
Write-Host ""
Write-Host "The VNext trainer will switch maps automatically." -ForegroundColor Green