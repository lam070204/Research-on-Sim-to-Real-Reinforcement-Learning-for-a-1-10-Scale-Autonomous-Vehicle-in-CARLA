param(
    [int]$TotalSteps = 2000000,
    [double]$TargetSpeed = 0.50,
    [double]$DrEpisodeProb = 0.75,
    [int]$BlockSteps = 50000,

    [string]$CarlaExe = "D:\CARLA_AUTOMAV3_MAPS\WindowsNoEditor\CarlaUE4.exe",

    [int]$Port0 = 2000,
    [int]$Port1 = 2010,

    [int]$StartupTimeoutSec = 240,
    [switch]$KeepCarlaOpen
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-PythonQuiet {
    param(
        [string]$Arguments,
        [int]$TimeoutSec = 10
    )

    $pythonExe = (Get-Command python -ErrorAction Stop).Source

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $pythonExe
    $psi.Arguments = $Arguments
    $psi.WorkingDirectory = $PSScriptRoot
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $p = New-Object System.Diagnostics.Process
    $p.StartInfo = $psi

    try {
        [void]$p.Start()

        if (-not $p.WaitForExit($TimeoutSec * 1000)) {
            try { $p.Kill() } catch {}
            return @{
                Ok = $false
                ExitCode = -999
                StdOut = ""
                StdErr = "timeout"
            }
        }

        $stdout = $p.StandardOutput.ReadToEnd()
        $stderr = $p.StandardError.ReadToEnd()

        return @{
            Ok = ($p.ExitCode -eq 0)
            ExitCode = $p.ExitCode
            StdOut = $stdout.Trim()
            StdErr = $stderr.Trim()
        }
    }
    catch {
        return @{
            Ok = $false
            ExitCode = -998
            StdOut = ""
            StdErr = $_.Exception.Message
        }
    }
    finally {
        if ($null -ne $p) {
            $p.Dispose()
        }
    }
}

function Test-CarlaRpc {
    param([int]$Port)

    $code = "from simulation.carla_connection_v5 import carla; c=carla.Client('127.0.0.1',$Port); c.set_timeout(2.0); w=c.get_world(); print(w.get_map().name)"
    $escaped = $code.Replace('"','\"')

    $r = Invoke-PythonQuiet -Arguments "-c `"$escaped`"" -TimeoutSec 5
    return [bool]$r.Ok
}

function Get-CarlaMapName {
    param([int]$Port)

    $code = "from simulation.carla_connection_v5 import carla; c=carla.Client('127.0.0.1',$Port); c.set_timeout(10.0); print(c.get_world().get_map().name)"
    $escaped = $code.Replace('"','\"')

    $r = Invoke-PythonQuiet -Arguments "-c `"$escaped`"" -TimeoutSec 15
    if ($r.Ok) {
        return $r.StdOut
    }
    return "<unknown>"
}

function Wait-CarlaRpc {
    param(
        [int]$Port,
        [int]$TimeoutSec
    )

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "Waiting CARLA RPC | port=$Port ..."

    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if (Test-CarlaRpc -Port $Port) {
            $mapName = Get-CarlaMapName -Port $Port
            Write-Host "CARLA READY | port=$Port | map=$mapName | waited=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
            return
        }

        Start-Sleep -Seconds 2
    }

    throw "CARLA port $Port did not become ready within $TimeoutSec seconds."
}

function Start-Carla {
    param(
        [string]$Exe,
        [int]$Port
    )

    $args = "-carla-rpc-port=$Port -dx11 -windowed -ResX=480 -ResY=270 -nosound -NoVSync"

    Write-Host "START CARLA | port=$Port"
    Write-Host "  $Exe $args"

    return Start-Process `
        -FilePath $Exe `
        -ArgumentList $args `
        -WorkingDirectory (Split-Path $Exe -Parent) `
        -PassThru
}

function Load-CarlaMap {
    param(
        [int]$Port,
        [string]$MapPath
    )

    Write-Host "LOAD MAP | port=$Port | $MapPath"

    $code = "from simulation.carla_connection_v5 import carla; c=carla.Client('127.0.0.1',$Port); c.set_timeout(120.0); w=c.load_world('$MapPath'); print(w.get_map().name)"
    $escaped = $code.Replace('"','\"')

    $r = Invoke-PythonQuiet -Arguments "-c `"$escaped`"" -TimeoutSec 150

    if (-not $r.Ok) {
        throw "Failed to load map $MapPath on port $Port.`n$($r.StdErr)"
    }

    Write-Host "MAP READY | port=$Port | $($r.StdOut)"
}

Write-Host "============================================================"
Write-Host "PPO V5 SIMPLE FAST - FRESH TRAIN + AUTO 2 CARLA [FIXED]"
Write-Host "============================================================"
Write-Host "CARLA exe    : $CarlaExe"
Write-Host "Worker 0     : port $Port0 -> xanhphai initially"
Write-Host "Worker 1     : port $Port1 -> mapxanhdam initially"
Write-Host "Action       : 2 [steer, speed]"
Write-Host "Obs          : 100 = latent95 + speed + yaw + ax + prevSteer + prevSpeed"
Write-Host "Reward       : original-style speed x center x heading"
Write-Host "Target speed : $TargetSpeed m/s"
Write-Host "DR mix       : $([math]::Round(100*$DrEpisodeProb))% FULL / $([math]::Round(100*(1-$DrEpisodeProb)))% NOMINAL"
Write-Host "Recovery     : OFF"
Write-Host "Map block    : $BlockSteps global transitions"
Write-Host "Total steps  : $TotalSteps"
Write-Host "============================================================"

if (-not (Test-Path $CarlaExe)) {
    throw "CarlaUE4.exe not found: $CarlaExe"
}

$required = @(
    ".\train_ppo_rgb_v5_simple.py",
    ".\reward_function_v5_simple.py",
    ".\reward_manager_v5_simple.py",
    ".\domain_randomization_v5_simple.py",
    ".\simulation\carla_environment_rgb_v5_simple.py"
)

foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        throw "Missing required file: $f"
    }
}

$started = @()

try {
    # Kill stale CARLA processes so ports/world state are deterministic.
    Write-Host "Stopping old CarlaUE4 processes..."
    Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3

    $p0 = Start-Carla -Exe $CarlaExe -Port $Port0
    $started += $p0

    # Stagger Unreal startup to reduce simultaneous GPU/CPU spike.
    Start-Sleep -Seconds 5

    $p1 = Start-Carla -Exe $CarlaExe -Port $Port1
    $started += $p1

    Wait-CarlaRpc -Port $Port0 -TimeoutSec $StartupTimeoutSec
    Wait-CarlaRpc -Port $Port1 -TimeoutSec $StartupTimeoutSec

    # Put each server on the first training pair. The trainer may later change
    # worlds at block boundaries.
    Load-CarlaMap -Port $Port0 -MapPath "/Game/mapxanh/xanhphai"
    Load-CarlaMap -Port $Port1 -MapPath "/Game/xanhtrai/mapxanhdam"

    # load_world restarts the world internally, so verify RPC again.
    Wait-CarlaRpc -Port $Port0 -TimeoutSec 120
    Wait-CarlaRpc -Port $Port1 -TimeoutSec 120

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "BOTH CARLA SERVERS READY - STARTING PPO"
    Write-Host "============================================================"

    & python .\train_ppo_rgb_v5_simple.py `
      --workers 2 `
      --host 127.0.0.1 `
      --port0 $Port0 `
      --port1 $Port1 `
      --carla-timeout 180 `
      --total-timesteps $TotalSteps `
      --rollout-total 1024 `
      --map-block-steps $BlockSteps `
      --safe-spawns 1,2,3,4 `
      --desired-speed $TargetSpeed `
      --dynamics-dr true `
      --vision-dr true `
      --sensor-dr true `
      --dr-episode-prob $DrEpisodeProb `
      --recovery-episode-prob 0.0 `
      --initial-map-validation false `
      --canary-every-steps 100000 `
      --map-validation-every-steps 300000 `
      --robust-validation-every-steps 300000 `
      --failure-report-every-steps 25000 `
      --learner-device cpu `
      --worker-ppo-device cpu `
      --encoder-device cpu `
      --latent-audit true

    if ($LASTEXITCODE -ne 0) {
        throw "PPO V5 SIMPLE FAST training FAILED."
    }
}
finally {
    if (-not $KeepCarlaOpen) {
        Write-Host ""
        Write-Host "Stopping CarlaUE4 instances..."
        Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
    }
    else {
        Write-Host "KeepCarlaOpen enabled - CARLA remains running."
    }
}
