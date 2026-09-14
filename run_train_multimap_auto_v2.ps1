param(
    [int64]$TotalSteps = 3000000,
    [int64]$BlockSteps = 100000,

    [string[]]$Maps = @(
        "/Game/mapxanh/xanhphai",
        "/Game/xanhtrai/mapxanhdam"
    ),

    [string]$ModelName = "automav3_rgb_v4_vision_dr_multimap_fresh",
    [string]$SafeSpawns = "1,2,3,4",

    [string]$Project = "C:\Autonomous-Driving-PPO-V3-Clean",
    [string]$CarlaExe = "C:\CARLA_AUTOMAV3_MAPS\WindowsNoEditor\CarlaUE4.exe",

    [ValidateSet("cpu","cuda")]
    [string]$PpoDevice = "cpu",

    [ValidateSet("cpu","cuda")]
    [string]$EncoderDevice = "cpu",

    [int]$PrintEverySteps = 50
)

$ErrorActionPreference = "Stop"

# ============================================================
# AUTOMAV3 V3/V4 Vision-DR generic MULTI-MAP trainer launcher
#
# - Any number of maps: 2, 3, 4, 5, ...
# - Any TotalSteps
# - Any positive BlockSteps
# - Automatically balances TOTAL steps across maps
# - Round-robin map switching
# - Resume-aware from training_state_v3.json
# - Keeps --total-timesteps CONSTANT for the whole run so
#   PPO_CURRICULUM remains globally consistent.
# ============================================================
# powershell.exe -ExecutionPolicy Bypass -File .\run_train_multimap_auto_v2.ps1 `
#   -Project "C:\Autonomous-Driving-PPO-V4-3old" `
#   -TotalSteps 3000000 `
#   -BlockSteps 100000 `
#   -ModelName "automav3_v4dr_2map_3m" `
#   -Maps "/Game/mapxanh/xanhphai","/Game/xanhtrai/mapxanhdam"

if ($TotalSteps -le 0) {
    throw "TotalSteps must be > 0"
}
if ($BlockSteps -le 0) {
    throw "BlockSteps must be > 0"
}
if ($Maps.Count -lt 1) {
    throw "At least one CARLA map is required"
}

# Normalize map arguments.
# When this script is launched through powershell.exe -File, a command such as
#   -Maps "mapA","mapB"
# may arrive as ONE literal string: "mapA,mapB".
# Support all of these forms:
#   - a real PowerShell string[]
#   - comma-separated text
#   - semicolon-separated text
$NormalizedMaps = New-Object 'System.Collections.Generic.List[String]'

foreach ($RawMap in @($Maps)) {
    foreach ($Part in ([string]$RawMap -split '[,;]')) {
        $Trimmed = $Part.Trim()
        if (-not [string]::IsNullOrWhiteSpace($Trimmed)) {
            $NormalizedMaps.Add($Trimmed)
        }
    }
}

$Maps = @($NormalizedMaps)

if ($Maps.Count -lt 1) {
    throw "No valid CARLA maps were provided"
}

$Trainer = Join-Path $Project "train_ppo_rgb_v4_vision_dr_multimap.py"

$VenvPython = Join-Path $Project ".venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

Set-Location $Project

if (-not (Test-Path $CarlaExe)) {
    throw "CARLA executable not found: $CarlaExe"
}
if (-not (Test-Path $Trainer)) {
    throw "Trainer not found: $Trainer"
}

# ------------------------------------------------------------
# Build equal per-map quotas.
# Example:
# Total=3,000,000, maps=4 -> 750,000 each.
# If Total is not divisible by map count, difference is <= 1 step.
# ------------------------------------------------------------
$MapCount = $Maps.Count
$BaseQuota = [int64][math]::Floor($TotalSteps / $MapCount)
$Remainder = [int64]($TotalSteps % $MapCount)

$Remaining = New-Object 'System.Collections.Generic.List[Int64]'
$Allocated = New-Object 'System.Collections.Generic.List[Int64]'

for ($i = 0; $i -lt $MapCount; $i++) {
    $q = $BaseQuota
    if ($i -lt $Remainder) {
        $q += 1
    }
    $Remaining.Add([int64]$q)
    $Allocated.Add([int64]$q)
}

# ------------------------------------------------------------
# Build deterministic round-robin schedule.
# Each phase is at most BlockSteps.
# Last phase for a map may be shorter, which is intentional.
# ------------------------------------------------------------
$Phases = @()
$GlobalEnd = [int64]0
$PhaseIndex = 0

while ($GlobalEnd -lt $TotalSteps) {
    $MadeProgress = $false

    for ($i = 0; $i -lt $MapCount; $i++) {
        if ($Remaining[$i] -le 0) {
            continue
        }

        $Take = [int64][math]::Min($BlockSteps, $Remaining[$i])
        $StartStep = $GlobalEnd
        $GlobalEnd += $Take
        $Remaining[$i] -= $Take
        $PhaseIndex += 1
        $MadeProgress = $true

        $Phases += [pscustomobject]@{
            Index     = $PhaseIndex
            MapIndex  = $i
            Map       = $Maps[$i]
            StartStep = $StartStep
            EndStep   = $GlobalEnd
            Steps     = $Take
        }
    }

    if (-not $MadeProgress) {
        throw "Internal scheduler error: no progress while GlobalEnd < TotalSteps"
    }
}

if ($GlobalEnd -ne $TotalSteps) {
    throw "Internal scheduler error: schedule ends at $GlobalEnd instead of $TotalSteps"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "MULTI-MAP TRAIN PLAN"
Write-Host "============================================================"
Write-Host "ModelName   : $ModelName"
Write-Host "TotalSteps  : $TotalSteps"
Write-Host "BlockSteps  : $BlockSteps"
Write-Host "Maps        : $MapCount"
Write-Host "Phases      : $($Phases.Count)"
Write-Host "SafeSpawns  : $SafeSpawns"
Write-Host ""

for ($i = 0; $i -lt $MapCount; $i++) {
    Write-Host ("MAP[{0}] quota={1} | {2}" -f $i, $Allocated[$i], $Maps[$i])
}
Write-Host "============================================================"

# ------------------------------------------------------------
# Lock the schedule/config for this ModelName.
# This prevents accidental changes to TotalSteps / map list / block size
# after training has already started, because PPO curriculum depends on
# the full TotalSteps horizon and the map schedule defines the data mix.
# ------------------------------------------------------------
$PlanDir = Join-Path $Project "training_plans"
New-Item -ItemType Directory -Force -Path $PlanDir | Out-Null

$SafeModelName = ($ModelName -replace '[^A-Za-z0-9_.-]', '_')
$PlanPath = Join-Path $PlanDir ($SafeModelName + "_multimap_plan.json")

$CurrentPlan = [ordered]@{
    version = "MULTIMAP_PLAN_V1"
    model_name = $ModelName
    total_steps = [int64]$TotalSteps
    block_steps = [int64]$BlockSteps
    maps = @($Maps)
    safe_spawns = $SafeSpawns
}

if (Test-Path $PlanPath) {
    $SavedPlan = Get-Content $PlanPath -Raw | ConvertFrom-Json

    $SavedMaps = @($SavedPlan.maps)
    $MapsSame = (
        $SavedMaps.Count -eq $Maps.Count
    )

    if ($MapsSame) {
        for ($i = 0; $i -lt $Maps.Count; $i++) {
            if ([string]$SavedMaps[$i] -ne [string]$Maps[$i]) {
                $MapsSame = $false
                break
            }
        }
    }

    $PlanSame = (
        [string]$SavedPlan.version -eq "MULTIMAP_PLAN_V1" -and
        [string]$SavedPlan.model_name -eq [string]$ModelName -and
        [int64]$SavedPlan.total_steps -eq [int64]$TotalSteps -and
        [int64]$SavedPlan.block_steps -eq [int64]$BlockSteps -and
        [string]$SavedPlan.safe_spawns -eq [string]$SafeSpawns -and
        $MapsSame
    )

    if (-not $PlanSame) {
        throw @"
Training plan mismatch for ModelName:
$ModelName

Saved plan:
$PlanPath

Do NOT change TotalSteps / BlockSteps / Maps / SafeSpawns in the middle
of the same run. Use a NEW -ModelName for a new schedule.
"@
    }
} else {
    $CurrentPlan | ConvertTo-Json -Depth 5 | Set-Content -Path $PlanPath -Encoding UTF8
    Write-Host "PLAN LOCKED : $PlanPath"
}

# ------------------------------------------------------------
# Resume detection.
# The trainer already saves:
# preTrained_models\PPO\<ModelName>\training_state_v3.json
# ------------------------------------------------------------
$RunDir = Join-Path $Project ("preTrained_models\PPO\" + $ModelName)
$StatePath = Join-Path $RunDir "training_state_v3.json"

$CurrentStep = [int64]0
$Resume = $false

if (Test-Path $RunDir) {
    if (-not (Test-Path $StatePath)) {
        throw @"
Model directory exists but training_state_v3.json is missing:
$RunDir

For safety, this launcher will not guess whether to overwrite or resume.
Use a new -ModelName for a fresh run.
"@
    }

    $State = Get-Content $StatePath -Raw | ConvertFrom-Json
    $CurrentStep = [int64]$State.global_step
    $Resume = $true

    Write-Host ""
    Write-Host "RESUME DETECTED"
    Write-Host "global_step : $CurrentStep"
    Write-Host "state       : $StatePath"
}

if ($CurrentStep -ge $TotalSteps) {
    Write-Host "Training already reached TotalSteps=$TotalSteps. Nothing to do."
    exit 0
}

function Stop-Carla {
    Get-Process CarlaUE4* -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 2
}

function Start-CarlaMap([string]$MapPath) {
    Stop-Carla

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "START CARLA: $MapPath"
    Write-Host "============================================================"

    Start-Process $CarlaExe `
        -ArgumentList "-carla-rpc-port=2000 -dx11 -windowed -ResX=480 -ResY=270 -nosound -NoVSync"

    Start-Sleep -Seconds 15

    $LoadCode = @"
import carla
c = carla.Client('localhost', 2000)
c.set_timeout(120)
w = c.load_world(r'$MapPath')
print('DA DOI MAP:', w.get_map().name)
"@

    & $Python -c $LoadCode
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to load CARLA map: $MapPath"
    }
}

try {
    foreach ($Phase in $Phases) {
        # Skip phases already fully completed.
        if ($Phase.EndStep -le $CurrentStep) {
            continue
        }

        # If an emergency checkpoint stopped inside a phase,
        # resume the SAME map until that phase's absolute EndStep.
        Write-Host ""
        Write-Host "############################################################"
        Write-Host ("PHASE {0}/{1}" -f $Phase.Index, $Phases.Count)
        Write-Host ("MAP        : {0}" -f $Phase.Map)
        Write-Host ("SCHEDULE   : {0} -> {1}" -f $Phase.StartStep, $Phase.EndStep)
        Write-Host ("CURRENT    : {0}" -f $CurrentStep)
        Write-Host ("THIS TARGET: {0}" -f $Phase.EndStep)
        Write-Host "############################################################"

        Start-CarlaMap -MapPath $Phase.Map

        $LoadFlag = "false"
        if ($Resume -or $CurrentStep -gt 0) {
            $LoadFlag = "true"
        }

        & $Python $Trainer `
            --train true `
            --model-name $ModelName `
            --town current `
            --safe-spawns $SafeSpawns `
            --total-timesteps $TotalSteps `
            --session-end-step $Phase.EndStep `
            --load-checkpoint $LoadFlag `
            --curriculum true `
            --ppo-device $PpoDevice `
            --encoder-device $EncoderDevice `
            --print-every-steps $PrintEverySteps

        if ($LASTEXITCODE -ne 0) {
            throw ("Training failed in phase {0}, target global step {1}" -f $Phase.Index, $Phase.EndStep)
        }

        $CurrentStep = [int64]$Phase.EndStep
        $Resume = $true

        Write-Host ("PHASE PASS | global_step={0}" -f $CurrentStep)
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "MULTI-MAP TRAIN COMPLETE"
    Write-Host "global_step = $TotalSteps"
    for ($i = 0; $i -lt $MapCount; $i++) {
        Write-Host ("MAP[{0}] = {1} steps | {2}" -f $i, $Allocated[$i], $Maps[$i])
    }
    Write-Host "============================================================"
}
finally {
    Stop-Carla
}
