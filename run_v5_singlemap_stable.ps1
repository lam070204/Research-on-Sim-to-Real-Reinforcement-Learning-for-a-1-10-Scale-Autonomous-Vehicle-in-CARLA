param(
    [int]$TotalSteps = 170000,
    [double]$MaxSpeed = 0.35,
    [string]$SafeSpawns = "1,2,3,4",
    [switch]$Resume,
    [switch]$SmokeOnly,
    [int]$ProbeStepsPerWorker = 1500,
    [double]$ProbeActionStd = 0.0001,
    [double]$ResumeActionStd = 0.05,
    [switch]$KeepCarlaOpen,
    [string]$CarlaExe = "D:\CARLA_AUTOMAV3_MAPS\WindowsNoEditor\CarlaUE4.exe",
    [int]$Port0 = 2000,
    [int]$Port1 = 2010
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
                StdOut = ""
                StdErr = "timeout"
            }
        }

        return @{
            Ok = ($p.ExitCode -eq 0)
            StdOut = $p.StandardOutput.ReadToEnd().Trim()
            StdErr = $p.StandardError.ReadToEnd().Trim()
        }
    }
    catch {
        return @{
            Ok = $false
            StdOut = ""
            StdErr = $_.Exception.Message
        }
    }
    finally {
        try { $p.Dispose() } catch {}
    }
}

function Test-CarlaRpc {
    param([int]$Port)

    $code = "from simulation.carla_connection_v5 import carla; c=carla.Client('127.0.0.1',$Port); c.set_timeout(2.0); print(c.get_world().get_map().name)"
    $escaped = $code.Replace('"','\"')
    $r = Invoke-PythonQuiet -Arguments "-c `"$escaped`"" -TimeoutSec 5
    return [bool]$r.Ok
}

function Wait-CarlaRpc {
    param(
        [int]$Port,
        [int]$TimeoutSec = 240
    )

    $sw = [System.Diagnostics.Stopwatch]::StartNew()

    Write-Host "Waiting CARLA RPC | port=$Port ..."

    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSec) {
        if (Test-CarlaRpc -Port $Port) {
            Write-Host "CARLA READY | port=$Port | waited=$([math]::Round($sw.Elapsed.TotalSeconds,1))s"
            return
        }

        Start-Sleep -Seconds 2
    }

    throw "CARLA port $Port did not become ready within $TimeoutSec seconds."
}

function Start-CarlaInstance {
    param(
        [string]$Exe,
        [int]$Port
    )

    $args = "-carla-rpc-port=$Port -dx11 -windowed -ResX=480 -ResY=270 -nosound -NoVSync"

    Write-Host "START CARLA | port=$Port"
    return Start-Process `
        -FilePath $Exe `
        -ArgumentList $args `
        -WorkingDirectory (Split-Path $Exe -Parent) `
        -PassThru
}

function Load-Map {
    param(
        [int]$Port,
        [string]$MapPath
    )

    Write-Host "LOAD MAP | port=$Port | $MapPath"

    $code = "from simulation.carla_connection_v5 import carla; c=carla.Client('127.0.0.1',$Port); c.set_timeout(120.0); w=c.load_world('$MapPath'); print(w.get_map().name)"
    $escaped = $code.Replace('"','\"')
    $r = Invoke-PythonQuiet -Arguments "-c `"$escaped`"" -TimeoutSec 150

    if (-not $r.Ok) {
        throw "Failed to load $MapPath on port $Port`n$($r.StdErr)"
    }

    Write-Host "MAP READY | port=$Port | $($r.StdOut)"
}

Write-Host "======================================================================"
Write-Host "PPO V5 SINGLE-MAP STABLE035 v1.7 ORACLE-TEACHER"
Write-Host "======================================================================"
Write-Host "Map             : /Game/mapvuong/mapvuong"
Write-Host "Workers         : 2, SAME MAP"
Write-Host "Obs             : 100 = latent95 + speed + yaw + ax + prevSteer + prevSpeed"
Write-Host "Action          : 2 = [steer, normalized speed]"
Write-Host "Hard max speed  : $MaxSpeed m/s"
Write-Host "Reward speed    : original style MIN=15/25*MAX, TARGET=22/25*MAX"
Write-Host "Steer rate      : 3.0 /s"
Write-Host "Steer deadband  : 0.015"
Write-Host "Recovery inject : OFF"
Write-Host "DR              : HARD OFF until clean map is mastered"
Write-Host "Reward support  : center 0.18m | heading 45deg"
Write-Host "Heading reward  : 0.50m pure-pursuit lookahead (reward only)"
Write-Host "Action std      : resume-training override = $ResumeActionStd"
Write-Host "Teacher reward  : 50% pure-pursuit steer alignment, TRAIN ONLY"
Write-Host "Teacher scale   : error 0.35 -> zero teacher factor"
Write-Host "Reason          : oracle passed 4/4; directly teach PPO the missing corner steer"
Write-Host "Checkpoint      : every 25,000 transitions"
Write-Host "Resume          : $([bool]$Resume)"
if ($SmokeOnly) { Write-Host "Probe/worker    : $ProbeStepsPerWorker steps (NO LEARNING)" }
if ($SmokeOnly) { Write-Host "Probe actionStd : $ProbeActionStd (near-deterministic)" }
Write-Host "======================================================================"

if (-not (Test-Path $CarlaExe)) {
    throw "CarlaUE4.exe not found: $CarlaExe"
}

$required = @(
    ".\train_ppo_rgb_v5_singlemap_stable.py",
    ".\reward_function_v5_singlemap_stable.py",
    ".\reward_manager_v5_singlemap_stable.py",
    ".\domain_randomization_v5_singlemap_stable.py",
    ".\stable_action_adapter_v5.py",
    ".\simulation\carla_environment_rgb_v5_singlemap_stable.py"
)

foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        throw "Missing stable file: $f"
    }
}

$started = @()

try {
    Write-Host "Stopping old CarlaUE4 processes..."
    Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3

    $p0 = Start-CarlaInstance -Exe $CarlaExe -Port $Port0
    $started += $p0

    Start-Sleep -Seconds 5

    $p1 = Start-CarlaInstance -Exe $CarlaExe -Port $Port1
    $started += $p1

    Wait-CarlaRpc -Port $Port0
    Wait-CarlaRpc -Port $Port1

    Load-Map -Port $Port0 -MapPath "/Game/mapvuong/mapvuong"
    Load-Map -Port $Port1 -MapPath "/Game/mapvuong/mapvuong"

    Wait-CarlaRpc -Port $Port0 -TimeoutSec 120
    Wait-CarlaRpc -Port $Port1 -TimeoutSec 120

    $resumeValue = if ($Resume) { "true" } else { "false" }
    $smokeValue = if ($SmokeOnly) { "true" } else { "false" }

    Write-Host ""
    Write-Host "BOTH CARLA SERVERS READY - STARTING STABLE035 PPO"
    Write-Host ""

    & python .\train_ppo_rgb_v5_singlemap_stable.py `
      --workers 2 `
      --host 127.0.0.1 `
      --port0 $Port0 `
      --port1 $Port1 `
      --expected-map "mapvuong" `
      --resume $resumeValue `
      --total-timesteps $TotalSteps `
      --rollout-total 1024 `
      --checkpoint-every-steps 25000 `
      --safe-spawns $SafeSpawns `
      --max-speed $MaxSpeed `
      --max-episode-seconds 60 `
      --steer-rate-limit 3.0 `
      --steer-deadband 0.015 `
      --speed-rate-limit 0.50 `
      --action-std-init 0.05 `
      --action-std-min 0.05 `
      --action-std-decay 0.0 `
      --action-std-decay-freq 100000 `
      --dynamics-dr false `
      --vision-dr false `
      --sensor-dr false `
      --recovery-scenarios false `
      --disturbance-prob 0.0 `
      --dr-family-budget 1 `
      --phase-clean-until 999999999 `
      --phase-vision-until 999999999 `
      --phase-dynamics-until 999999999 `
      --phase-controlled-until 999999999 `
      --dr-prob-vision 0.0 `
      --dr-prob-dynamics 0.0 `
      --dr-prob-controlled 0.0 `
      --dr-prob-polish 0.0 `
      --learner-device cpu `
      --worker-ppo-device cpu `
      --encoder-device cpu `
      --worker-timeout 1800 `
      --smoke-test $smokeValue `
      --smoke-steps-per-worker $ProbeStepsPerWorker `
      --probe-action-std $ProbeActionStd `
      --resume-action-std $ResumeActionStd

    if ($LASTEXITCODE -ne 0) {
        throw "PPO V5 SINGLE-MAP STABLE035 training FAILED."
    }
}
finally {
    if (-not $KeepCarlaOpen) {
        Write-Host ""
        Write-Host "Stopping CarlaUE4 instances..."
        Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
    }
    else {
        Write-Host "KeepCarlaOpen enabled."
    }
}
