param(
    [string]$ProjectRoot = $PSScriptRoot
)

$ErrorActionPreference = "Continue"

# ============================================================
# KIEM TRA TOAN BO PROJECT CARLA + VAE RGB + PPO
# - Khong xoa dataset
# - Khong train model
# - Khong sua source code
# - Chi tao bao cao va goi source nhe de ban giao
# ============================================================

function Write-Utf8File {
    param(
        [string]$Path,
        [object[]]$Content
    )
    $Content | Out-File -FilePath $Path -Encoding utf8 -Width 500
}

function Add-ReportLine {
    param([string]$Text = "")
    $script:Report.Add($Text) | Out-Null
}

function Add-Section {
    param([string]$Title)
    Add-ReportLine ""
    Add-ReportLine ("=" * 90)
    Add-ReportLine $Title
    Add-ReportLine ("=" * 90)
}

function Format-Bytes {
    param([long]$Bytes)
    if ($Bytes -ge 1GB) { return ("{0:N2} GB" -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ("{0:N2} MB" -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ("{0:N2} KB" -f ($Bytes / 1KB)) }
    return "$Bytes B"
}

function Is-ExcludedPath {
    param(
        [string]$FullName,
        [string]$AuditDir
    )

    $normalized = $FullName.Replace("/", "\")
    $excludedFragments = @(
        "\.venv\",
        "\venv\",
        "\__pycache__\",
        "\.git\",
        "\dataset\",
        "\dataset_rgb_autopilot\",
        "\checkpoints\",
        "\preTrained_models\",
        "\logs\",
        "\runs\",
        "\carla\",
        "\poetry\",
        "\code-nguyeban\",
        "\model\",
        "\reconstructed\",
        "\node_modules\",
        "\dist\",
        "\build\"
    )

    if ($normalized.StartsWith($AuditDir.Replace("/", "\"), [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
    }

    foreach ($fragment in $excludedFragments) {
        if ($normalized.IndexOf($fragment, [System.StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $true
        }
    }

    $leaf = Split-Path $normalized -Leaf
    if ($leaf -like ".env*" -or
        $leaf -like "*.pem" -or
        $leaf -like "*.key" -or
        $leaf -like "*.pfx" -or
        $leaf -like "*.p12") {
        return $true
    }

    return $false
}

# ------------------------------------------------------------
# 1. Xac dinh thu muc project
# ------------------------------------------------------------
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Get-Location).Path
}

try {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}
catch {
    Write-Host "Khong tim thay thu muc project: $ProjectRoot" -ForegroundColor Red
    exit 1
}

$expectedFiles = @(
    "continuous_driver.py",
    "continuous_driver_rgb.py",
    "collect_rgb_autopilot.py",
    "encoder_init.py",
    "encoder_init_rgb.py",
    "parameters.py"
)

$missingExpected = @()
foreach ($name in $expectedFiles) {
    if (-not (Test-Path (Join-Path $ProjectRoot $name))) {
        $missingExpected += $name
    }
}

if ($missingExpected.Count -gt 0) {
    Write-Host "CANH BAO: Co the ban chua dat script trong thu muc goc project." -ForegroundColor Yellow
    Write-Host "Khong tim thay: $($missingExpected -join ', ')" -ForegroundColor Yellow
    Write-Host "Thu muc dang kiem tra: $ProjectRoot" -ForegroundColor Yellow
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$AuditDir = Join-Path $ProjectRoot "PROJECT_AUDIT_$timestamp"
$ReportPath = Join-Path $AuditDir "PROJECT_AUDIT_REPORT.txt"
$TreePath = Join-Path $AuditDir "PROJECT_FILE_TREE.txt"
$PythonFilesCsv = Join-Path $AuditDir "PYTHON_FILES.csv"
$PathConstantsPath = Join-Path $AuditDir "IMPORTANT_PATHS_AND_CONSTANTS.txt"
$DatasetCsv = Join-Path $AuditDir "DATASET_INVENTORY.csv"
$ModelCsv = Join-Path $AuditDir "MODEL_CHECKPOINT_INVENTORY.csv"
$SyntaxPath = Join-Path $AuditDir "PYTHON_SYNTAX_CHECK.txt"
$ImportPath = Join-Path $AuditDir "PYTHON_IMPORT_CHECK.txt"
$PipFreezePath = Join-Path $AuditDir "PIP_FREEZE.txt"
$CarlaPath = Join-Path $AuditDir "CARLA_CONNECTION_CHECK.txt"
$HandoffDir = Join-Path $AuditDir "SOURCE_FOR_DOCUMENTATION"
$BundleZip = Join-Path $ProjectRoot "AUTONOMOUS_PROJECT_HANDOFF_$timestamp.zip"

New-Item -ItemType Directory -Path $AuditDir -Force | Out-Null
New-Item -ItemType Directory -Path $HandoffDir -Force | Out-Null

$script:Report = New-Object System.Collections.Generic.List[string]

Add-ReportLine "AUTONOMOUS DRIVING PROJECT AUDIT"
Add-ReportLine "Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Add-ReportLine "Project root: $ProjectRoot"
Add-ReportLine "Audit folder: $AuditDir"

# ------------------------------------------------------------
# 2. Thong tin he thong
# ------------------------------------------------------------
Add-Section "1. SYSTEM AND POWERSHELL"

Add-ReportLine "Computer name : $env:COMPUTERNAME"
Add-ReportLine "Windows user  : $env:USERNAME"
Add-ReportLine "PowerShell    : $($PSVersionTable.PSVersion)"
Add-ReportLine "OS            : $([System.Environment]::OSVersion.VersionString)"
Add-ReportLine "64-bit OS     : $([System.Environment]::Is64BitOperatingSystem)"
Add-ReportLine "Current path  : $((Get-Location).Path)"

try {
    $os = Get-CimInstance Win32_OperatingSystem
    Add-ReportLine "Windows name  : $($os.Caption)"
    Add-ReportLine "Windows ver   : $($os.Version)"
    Add-ReportLine "RAM total     : $(Format-Bytes ([long]$os.TotalVisibleMemorySize * 1KB))"
    Add-ReportLine "RAM free      : $(Format-Bytes ([long]$os.FreePhysicalMemory * 1KB))"
}
catch {
    Add-ReportLine "Khong doc duoc Win32_OperatingSystem: $($_.Exception.Message)"
}

try {
    $gpu = Get-CimInstance Win32_VideoController |
        Select-Object Name, DriverVersion, AdapterRAM
    foreach ($item in $gpu) {
        Add-ReportLine "GPU           : $($item.Name) | Driver $($item.DriverVersion) | RAM $(Format-Bytes ([long]$item.AdapterRAM))"
    }
}
catch {
    Add-ReportLine "Khong doc duoc GPU: $($_.Exception.Message)"
}

# ------------------------------------------------------------
# 3. Python dang dung
# ------------------------------------------------------------
Add-Section "2. PYTHON ENVIRONMENT"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$PythonExe = $null

if (Test-Path $VenvPython) {
    $PythonExe = $VenvPython
    Add-ReportLine "Selected Python: $PythonExe"
    Add-ReportLine "Reason         : Found project .venv"
}
else {
    try {
        $PythonExe = (Get-Command python -ErrorAction Stop).Source
        Add-ReportLine "Selected Python: $PythonExe"
        Add-ReportLine "Reason         : System PATH"
    }
    catch {
        Add-ReportLine "ERROR: Khong tim thay Python."
    }
}

if ($PythonExe) {
    try {
        $pyVersion = & $PythonExe --version 2>&1
        Add-ReportLine "Python version : $pyVersion"
    }
    catch {
        Add-ReportLine "Python version error: $($_.Exception.Message)"
    }

    try {
        $pyInfo = & $PythonExe -c "import sys,platform; print('executable='+sys.executable); print('version='+sys.version.replace(chr(10),' ')); print('platform='+platform.platform()); print('prefix='+sys.prefix)" 2>&1
        foreach ($line in $pyInfo) { Add-ReportLine $line }
    }
    catch {
        Add-ReportLine "Python info error: $($_.Exception.Message)"
    }

    try {
        & $PythonExe -m pip freeze 2>&1 | Out-File $PipFreezePath -Encoding utf8 -Width 500
        Add-ReportLine "pip freeze     : $PipFreezePath"
    }
    catch {
        Write-Utf8File -Path $PipFreezePath -Content @("pip freeze failed: $($_.Exception.Message)")
        Add-ReportLine "pip freeze failed."
    }
}

# ------------------------------------------------------------
# 4. Danh sach file
# ------------------------------------------------------------
Add-Section "3. PROJECT FILE INVENTORY"

$allFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object {
        -not $_.FullName.StartsWith($AuditDir, [System.StringComparison]::OrdinalIgnoreCase)
    }

$pythonFiles = $allFiles | Where-Object {
    $_.Extension -ieq ".py" -and
    $_.FullName -notmatch "\\\.venv\\" -and
    $_.FullName -notmatch "\\venv\\" -and
    $_.FullName -notmatch "\\__pycache__\\" -and
    $_.FullName -notmatch "\\carla\\"
}

$relativeRows = foreach ($file in $pythonFiles) {
    $relative = $file.FullName.Substring($ProjectRoot.TrimEnd("\").Length).TrimStart("\")
    [PSCustomObject]@{
        RelativePath = $relative
        SizeKB       = [math]::Round($file.Length / 1KB, 2)
        LastWrite    = $file.LastWriteTime
        FullName     = $file.FullName
    }
}

$relativeRows |
    Sort-Object RelativePath |
    Export-Csv -Path $PythonFilesCsv -NoTypeInformation -Encoding UTF8

Add-ReportLine "Total files    : $($allFiles.Count)"
Add-ReportLine "Python files   : $($pythonFiles.Count)"
Add-ReportLine "Python CSV     : $PythonFilesCsv"

$treeLines = New-Object System.Collections.Generic.List[string]
$treeLines.Add("PROJECT ROOT: $ProjectRoot") | Out-Null
$treeLines.Add("") | Out-Null

$treeFiles = $allFiles | Where-Object {
    $_.FullName -notmatch "\\\.venv\\" -and
    $_.FullName -notmatch "\\venv\\" -and
    $_.FullName -notmatch "\\__pycache__\\" -and
    $_.FullName -notmatch "\\dataset_rgb_autopilot\\" -and
    $_.FullName -notmatch "\\dataset\\" -and
    $_.FullName -notmatch "\\logs\\" -and
    $_.FullName -notmatch "\\runs\\" -and
    $_.FullName -notmatch "\\carla\\" -and
    $_.FullName -notmatch "\\\.git\\"
}

foreach ($file in ($treeFiles | Sort-Object FullName)) {
    $relative = $file.FullName.Substring($ProjectRoot.TrimEnd("\").Length).TrimStart("\")
    $treeLines.Add(("{0} | {1} | {2}" -f $relative, (Format-Bytes $file.Length), $file.LastWriteTime)) | Out-Null
}
Write-Utf8File -Path $TreePath -Content $treeLines
Add-ReportLine "Readable tree  : $TreePath"

# ------------------------------------------------------------
# 5. Tim duong dan va hang so quan trong trong code
# ------------------------------------------------------------
Add-Section "4. IMPORTANT PATHS AND CONSTANTS FOUND IN SOURCE"

$patterns = @(
    "SAVE_ROOT",
    "DATA_ROOT",
    "TRAIN_DIR",
    "TEST_DIR",
    "MODEL_DIR",
    "CHECKPOINT",
    "preTrained_models",
    "dataset_rgb_autopilot",
    "var_encoder_rgb",
    "vae_rgb_best",
    "vae_rgb_last",
    "decoder_rgb",
    "mapden_rgb",
    "EXPECTED_MAP_KEYWORD",
    "SAFE_SPAWN",
    "MAX_IMAGES",
    "IMAGES_PER_SPAWN",
    "SAVE_EVERY",
    "LATENT",
    "state_dim",
    "action_dim",
    "FRONT_CAMERA_",
    "PORT",
    "TRAFFIC_MANAGER_PORT"
)

$pathLines = New-Object System.Collections.Generic.List[string]

foreach ($file in ($pythonFiles | Sort-Object FullName)) {
    try {
        $matches = Select-String -Path $file.FullName -Pattern $patterns -SimpleMatch
        foreach ($match in $matches) {
            $relative = $file.FullName.Substring($ProjectRoot.TrimEnd("\").Length).TrimStart("\")
            $pathLines.Add(("{0}:{1}: {2}" -f $relative, $match.LineNumber, $match.Line.Trim())) | Out-Null
        }
    }
    catch {
        $pathLines.Add("ERROR reading $($file.FullName): $($_.Exception.Message)") | Out-Null
    }
}

Write-Utf8File -Path $PathConstantsPath -Content $pathLines
Add-ReportLine "Constants file : $PathConstantsPath"
Add-ReportLine "Matches found  : $($pathLines.Count)"

# ------------------------------------------------------------
# 6. Kiem tra dataset
# ------------------------------------------------------------
Add-Section "5. DATASET INVENTORY"

$datasetRoots = Get-ChildItem -Path $ProjectRoot -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Name -in @("dataset_rgb_autopilot", "dataset") -and
        $_.FullName -notmatch "\\\.venv\\" -and
        $_.FullName -notmatch "\\venv\\" -and
        $_.FullName -notmatch "\\carla\\" -and
        -not $_.FullName.StartsWith($AuditDir, [System.StringComparison]::OrdinalIgnoreCase)
    }

$datasetRows = @()

foreach ($folder in $datasetRoots) {
    $images = Get-ChildItem $folder.FullName -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Extension -match "^\.(png|jpg|jpeg|bmp)$" }

    $trainImages = $images | Where-Object { $_.FullName -match "\\train\\" }
    $testImages = $images | Where-Object { $_.FullName -match "\\test\\" }
    $totalBytes = ($images | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $totalBytes) { $totalBytes = 0 }

    $oldest = $images | Sort-Object LastWriteTime | Select-Object -First 1
    $newest = $images | Sort-Object LastWriteTime -Descending | Select-Object -First 1

    $relative = $folder.FullName.Substring($ProjectRoot.TrimEnd("\").Length).TrimStart("\")

    $datasetRows += [PSCustomObject]@{
        RelativePath = $relative
        TrainImages  = $trainImages.Count
        TestImages   = $testImages.Count
        TotalImages  = $images.Count
        TotalSizeMB  = [math]::Round($totalBytes / 1MB, 2)
        OldestImage  = if ($oldest) { $oldest.LastWriteTime } else { $null }
        NewestImage  = if ($newest) { $newest.LastWriteTime } else { $null }
        NewestFile   = if ($newest) { $newest.FullName } else { "" }
    }

    Add-ReportLine "$relative | Train=$($trainImages.Count) | Test=$($testImages.Count) | Total=$($images.Count) | Size=$(Format-Bytes $totalBytes)"
}

$datasetRows |
    Sort-Object RelativePath |
    Export-Csv -Path $DatasetCsv -NoTypeInformation -Encoding UTF8

Add-ReportLine "Dataset CSV    : $DatasetCsv"

# ------------------------------------------------------------
# 7. Kiem tra model/checkpoint
# ------------------------------------------------------------
Add-Section "6. MODEL AND CHECKPOINT INVENTORY"

$modelExtensions = @(".pth", ".pt", ".pkl", ".pickle", ".onnx", ".engine")
$modelFiles = $allFiles | Where-Object {
    $modelExtensions -contains $_.Extension.ToLower() -and
    $_.FullName -notmatch "\\\.venv\\" -and
    $_.FullName -notmatch "\\venv\\" -and
    $_.FullName -notmatch "\\carla\\"
}

$modelRows = foreach ($file in $modelFiles) {
    $relative = $file.FullName.Substring($ProjectRoot.TrimEnd("\").Length).TrimStart("\")
    [PSCustomObject]@{
        RelativePath = $relative
        Extension    = $file.Extension
        SizeMB       = [math]::Round($file.Length / 1MB, 3)
        LastWrite    = $file.LastWriteTime
        FullName     = $file.FullName
    }
}

$modelRows |
    Sort-Object LastWrite -Descending |
    Export-Csv -Path $ModelCsv -NoTypeInformation -Encoding UTF8

foreach ($row in ($modelRows | Sort-Object LastWrite -Descending)) {
    Add-ReportLine "$($row.RelativePath) | $($row.SizeMB) MB | $($row.LastWrite)"
}

Add-ReportLine "Model CSV      : $ModelCsv"

# ------------------------------------------------------------
# 8. Kiem tra cu phap Python khong tao pyc
# ------------------------------------------------------------
Add-Section "7. PYTHON SYNTAX CHECK"

if ($PythonExe) {
    $fileListPath = Join-Path $AuditDir "python_file_list.txt"
    $syntaxScriptPath = Join-Path $AuditDir "_syntax_check.py"

    $pythonFiles.FullName | Out-File $fileListPath -Encoding utf8 -Width 1000

    $syntaxScript = @'
import io
import os
import sys
import tokenize

file_list = sys.argv[1]
ok = 0
bad = 0

with io.open(file_list, "r", encoding="utf-8-sig") as f:
    paths = [line.strip() for line in f if line.strip()]

for path in paths:
    try:
        with tokenize.open(path) as src:
            text = src.read()
        compile(text, path, "exec")
        print("[OK]   " + path)
        ok += 1
    except Exception as exc:
        print("[FAIL] " + path)
        print("       " + repr(exc))
        bad += 1

print("")
print("SUMMARY: OK={} FAIL={} TOTAL={}".format(ok, bad, ok + bad))
sys.exit(1 if bad else 0)
'@

    Write-Utf8File -Path $syntaxScriptPath -Content @($syntaxScript)

    Push-Location $ProjectRoot
    try {
        & $PythonExe $syntaxScriptPath $fileListPath 2>&1 |
            Out-File $SyntaxPath -Encoding utf8 -Width 1000
        $syntaxExit = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }

    Add-ReportLine "Syntax report  : $SyntaxPath"
    Add-ReportLine "Syntax exit    : $syntaxExit (0 = all Python files compiled)"
}
else {
    Write-Utf8File -Path $SyntaxPath -Content @("Python not found.")
    Add-ReportLine "Syntax test skipped: Python not found."
}

# ------------------------------------------------------------
# 9. Kiem tra import thu vien
# ------------------------------------------------------------
Add-Section "8. PYTHON IMPORT CHECK"

if ($PythonExe) {
    $importScriptPath = Join-Path $AuditDir "_import_check.py"
    $importScript = @'
from __future__ import print_function
import importlib
import os
import sys

project_root = sys.argv[1]
if project_root not in sys.path:
    sys.path.insert(0, project_root)

modules = [
    "numpy",
    "PIL",
    "torch",
    "torchvision",
    "pygame",
    "cv2",
]

for name in modules:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print("[OK]   {} | version={}".format(name, version))
    except Exception as exc:
        print("[FAIL] {} | {}".format(name, repr(exc)))

project_modules = [
    "simulation.connection",
    "simulation.sensors",
    "autoencoder_rgb.encoder_rgb",
    "autoencoder_rgb.decoder_rgb",
]

for name in project_modules:
    try:
        importlib.import_module(name)
        print("[OK]   project module {}".format(name))
    except Exception as exc:
        print("[FAIL] project module {} | {}".format(name, repr(exc)))
'@

    Write-Utf8File -Path $importScriptPath -Content @($importScript)

    Push-Location $ProjectRoot
    try {
        & $PythonExe $importScriptPath $ProjectRoot 2>&1 |
            Out-File $ImportPath -Encoding utf8 -Width 1000
    }
    finally {
        Pop-Location
    }

    Add-ReportLine "Import report  : $ImportPath"
}
else {
    Write-Utf8File -Path $ImportPath -Content @("Python not found.")
    Add-ReportLine "Import test skipped: Python not found."
}

# ------------------------------------------------------------
# 10. Kiem tra CARLA server va map
# ------------------------------------------------------------
Add-Section "9. CARLA SERVER AND MAP CHECK"

$carlaLines = New-Object System.Collections.Generic.List[string]

try {
    $portCheck = Test-NetConnection -ComputerName "127.0.0.1" -Port 2000 -WarningAction SilentlyContinue
    $carlaLines.Add("TCP 127.0.0.1:2000 = $($portCheck.TcpTestSucceeded)") | Out-Null
    Add-ReportLine "CARLA TCP 2000 : $($portCheck.TcpTestSucceeded)"
}
catch {
    $carlaLines.Add("Port check failed: $($_.Exception.Message)") | Out-Null
    Add-ReportLine "CARLA port check failed."
}

if ($PythonExe) {
    $carlaCheckScript = Join-Path $AuditDir "_carla_check.py"
    $carlaCheckCode = @'
from __future__ import print_function
import sys
import time

project_root = sys.argv[1]
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from simulation.connection import carla
    print("[OK] Imported CARLA API from simulation.connection")
except Exception as exc:
    print("[FAIL] Cannot import CARLA API:", repr(exc))
    raise SystemExit(2)

try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    waypoints = carla_map.generate_waypoints(2.0)

    print("[OK] Connected to CARLA")
    print("Map       :", carla_map.name)
    print("Spawns    :", len(spawn_points))
    print("Waypoints :", len(waypoints))

    for index, transform in enumerate(spawn_points):
        print(
            "Spawn {:02d}: x={:.3f}, y={:.3f}, z={:.3f}, pitch={:.2f}, yaw={:.2f}, roll={:.2f}".format(
                index + 1,
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.pitch,
                transform.rotation.yaw,
                transform.rotation.roll,
            )
        )
except Exception as exc:
    print("[FAIL] CARLA connection/query:", repr(exc))
    raise SystemExit(3)
'@

    Write-Utf8File -Path $carlaCheckScript -Content @($carlaCheckCode)

    Push-Location $ProjectRoot
    try {
        $carlaOutput = & $PythonExe $carlaCheckScript $ProjectRoot 2>&1
        foreach ($line in $carlaOutput) {
            $carlaLines.Add([string]$line) | Out-Null
        }
    }
    finally {
        Pop-Location
    }
}

Write-Utf8File -Path $CarlaPath -Content $carlaLines
Add-ReportLine "CARLA report  : $CarlaPath"

# ------------------------------------------------------------
# 11. Git status
# ------------------------------------------------------------
Add-Section "10. GIT STATUS"

if (Test-Path (Join-Path $ProjectRoot ".git")) {
    try {
        Push-Location $ProjectRoot
        $gitBranch = git branch --show-current 2>&1
        $gitStatus = git status --short 2>&1
        $gitLog = git log -1 --oneline 2>&1
        Pop-Location

        Add-ReportLine "Branch        : $gitBranch"
        Add-ReportLine "Latest commit : $gitLog"
        Add-ReportLine "Changes:"
        foreach ($line in $gitStatus) { Add-ReportLine "  $line" }

        $gitStatus | Out-File (Join-Path $AuditDir "GIT_STATUS.txt") -Encoding utf8 -Width 500
    }
    catch {
        try { Pop-Location } catch {}
        Add-ReportLine "Git check error: $($_.Exception.Message)"
    }
}
else {
    Add-ReportLine "No .git folder found."
}

# ------------------------------------------------------------
# 12. Tao goi source nhe de gui kiem tra/tai lieu
# ------------------------------------------------------------
Add-Section "11. SOURCE HANDOFF BUNDLE"

$allowedExtensions = @(
    ".py", ".md", ".txt", ".toml", ".json",
    ".yaml", ".yml", ".ini", ".cfg", ".ps1",
    ".bat", ".cmd", ".xml", ".csv"
)

$sourceCandidates = $allFiles | Where-Object {
    $allowedExtensions -contains $_.Extension.ToLower() -and
    -not (Is-ExcludedPath -FullName $_.FullName -AuditDir $AuditDir)
}

$copiedCount = 0
foreach ($file in $sourceCandidates) {
    try {
        $relative = $file.FullName.Substring($ProjectRoot.TrimEnd("\").Length).TrimStart("\")
        $destination = Join-Path $HandoffDir $relative
        $destinationDir = Split-Path $destination -Parent
        New-Item -ItemType Directory -Path $destinationDir -Force | Out-Null
        Copy-Item -Path $file.FullName -Destination $destination -Force
        $copiedCount++
    }
    catch {
        Add-ReportLine "Copy failed: $($file.FullName) | $($_.Exception.Message)"
    }
}

# Copy audit reports vao bundle
$reportBundleDir = Join-Path $HandoffDir "_PROJECT_AUDIT_REPORTS"
New-Item -ItemType Directory -Path $reportBundleDir -Force | Out-Null

$reportFilesToCopy = @(
    $ReportPath,
    $TreePath,
    $PythonFilesCsv,
    $PathConstantsPath,
    $DatasetCsv,
    $ModelCsv,
    $SyntaxPath,
    $ImportPath,
    $PipFreezePath,
    $CarlaPath
)

# Bao cao chinh duoc ghi o buoc cuoi, nhung duong dan duoc giu de copy lai sau.

Add-ReportLine "Source files copied: $copiedCount"
Add-ReportLine "Handoff folder     : $HandoffDir"
Add-ReportLine "Excluded           : .venv, datasets, model weights, checkpoints, CARLA binaries, logs, runs, secrets"

# ------------------------------------------------------------
# 13. Ghi bao cao tong
# ------------------------------------------------------------
Add-Section "12. FINAL OUTPUT"

Add-ReportLine "Main report     : $ReportPath"
Add-ReportLine "Source bundle   : $BundleZip"
Add-ReportLine ""
Add-ReportLine "Gui file ZIP nay de phan tich va he thong lai project:"
Add-ReportLine "$BundleZip"

Write-Utf8File -Path $ReportPath -Content $script:Report

# Copy reports sau khi bao cao chinh da ton tai
foreach ($reportFile in $reportFilesToCopy) {
    if (Test-Path $reportFile) {
        Copy-Item $reportFile $reportBundleDir -Force
    }
}

# Tao README cho goi ban giao
$handoffReadme = @"
AUTONOMOUS PROJECT HANDOFF
Generated: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

Noi dung:
- Source Python va tai lieu nhe
- Bao cao cau truc project
- Dataset inventory
- Model/checkpoint inventory
- Python syntax check
- Import check
- CARLA map/spawn check
- pip freeze

Khong bao gom:
- .venv
- Anh dataset
- Model .pth/.pt
- Checkpoint .pickle
- CARLA binaries
- Logs/runs
- File bi mat (.env, key, pem)
"@

Write-Utf8File -Path (Join-Path $HandoffDir "README_HANDOFF.txt") -Content @($handoffReadme)

try {
    if (Test-Path $BundleZip) {
        Remove-Item $BundleZip -Force
    }
    Compress-Archive -Path (Join-Path $HandoffDir "*") -DestinationPath $BundleZip -Force
}
catch {
    Add-Content -Path $ReportPath -Value "`nZIP ERROR: $($_.Exception.Message)" -Encoding utf8
    Write-Host "Khong tao duoc ZIP: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host "KIEM TRA PROJECT DA HOAN TAT" -ForegroundColor Green
Write-Host ("=" * 78) -ForegroundColor Cyan
Write-Host "Bao cao:" -ForegroundColor Yellow
Write-Host "  $ReportPath"
Write-Host ""
Write-Host "Goi source de gui:" -ForegroundColor Yellow
Write-Host "  $BundleZip"
Write-Host ""
Write-Host "Mo thu muc ket qua:" -ForegroundColor Yellow
Write-Host "  explorer `"$AuditDir`""
Write-Host ""
Write-Host "Luu y: Script KHONG train, KHONG xoa dataset, KHONG sua source code." -ForegroundColor Cyan
