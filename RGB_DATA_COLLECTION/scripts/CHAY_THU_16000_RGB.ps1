param(
    [int]$ImagesPerSpawn = 2000,
    [int]$SaveEvery = 10,
    [int]$EpisodeLength = 3000,
    [double]$MinSpeedKmh = 1.0
)

$ErrorActionPreference = "Stop"

$CollectionRoot = Split-Path $PSScriptRoot -Parent
$ProjectRoot = Split-Path $CollectionRoot -Parent
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$DatasetRoot = Join-Path $CollectionRoot "dataset_new_16000"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Khong tim thay Python: $PythonExe"
}

Push-Location $ProjectRoot
try {
    & $PythonExe ".\continuous_driver_rgb.py" `
        --exp-name ppo `
        --train false `
        --town mapden `
        --load-checkpoint true `
        --test-timesteps 999999999 `
        --episode-length $EpisodeLength `
        --collect-rgb-dataset true `
        --dataset-root $DatasetRoot `
        --dataset-images-per-spawn $ImagesPerSpawn `
        --dataset-save-every $SaveEvery `
        --dataset-test-interval 10 `
        --dataset-min-speed $MinSpeedKmh `
        --dataset-max-environment-steps 500000
}
finally {
    Pop-Location
}
