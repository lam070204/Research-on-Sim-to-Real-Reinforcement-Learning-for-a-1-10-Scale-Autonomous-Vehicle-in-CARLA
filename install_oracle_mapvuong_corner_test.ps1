param(
    [string]$Project = "D:\Autonomous-Driving-PPO-V3-Clean"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Project)) {
    throw "Project not found: $Project"
}

$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

$files = @(
    "oracle_mapvuong_corner_test.py",
    "run_oracle_mapvuong_corner_test.ps1"
)

foreach ($name in $files) {
    $src = Join-Path $Here $name
    $dst = Join-Path $Project $name

    if (-not (Test-Path $src)) {
        throw "Missing package file: $src"
    }

    Copy-Item $src $dst -Force
    Write-Host "INSTALLED: $dst"
}

Write-Host ""
Write-Host "ORACLE TEST INSTALLED."
Write-Host "Next:"
Write-Host "  cd $Project"
Write-Host "  powershell -ExecutionPolicy Bypass -File .\run_oracle_mapvuong_corner_test.ps1"
