$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$PythonExe = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { throw "Missing project Python: $PythonExe" }
& $PythonExe .\check_vnext_carla_maps.py
if ($LASTEXITCODE -ne 0) { throw "VNext CARLA map preflight FAILED. See MISSING/RESOLVED lines above." }
