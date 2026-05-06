param(
    [int]$Port = 8050,
    [string]$Python = "python",
    [string]$ModelsDir = "artifacts/models",
    [string]$Config = "",
    [double]$Lat = [double]::NaN,
    [double]$Lon = [double]::NaN,
    [double]$RadiusNm = [double]::NaN,
    [string]$ReplayInput = "",
    [int]$ReplayLimitFiles = 0,
    [int]$ReplayMaxRecords = 0,
    [double]$ReplayStepSeconds = 15,
    [switch]$ReplayPreload,
    [switch]$NoReplayPreload
)

$ErrorActionPreference = "Stop"
$CommandDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $CommandDir
$RuntimeDir = Join-Path $Root "artifacts\runtime"
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
$PidFile = Join-Path $RuntimeDir "dashboard.pid"
$OutLog = Join-Path $RuntimeDir "dashboard.log"
$ErrLog = Join-Path $RuntimeDir "dashboard.err.log"

if ($Python -eq "python") {
    $MiniforgePython = Join-Path $env:USERPROFILE "miniforge3\python.exe"
    if (Test-Path -LiteralPath $MiniforgePython) {
        $Python = $MiniforgePython
    }
}

if ($Config -eq "") {
    $DefaultConfig = Join-Path $Root "config.real.yaml"
    if (Test-Path -LiteralPath $DefaultConfig) {
        $Config = $DefaultConfig
    }
}

function Get-ListeningPidsByPort([int]$LocalPort) {
    $pids = @()
    $listeners = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $LocalPort -State Listen -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
        if ($listener.OwningProcess) {
            $pids += [int]$listener.OwningProcess
        }
    }
    if ($pids.Count -eq 0) {
        $lines = netstat -ano 2>$null | Select-String "127\.0\.0\.1:$LocalPort\s+.*LISTENING"
        foreach ($line in $lines) {
            $parts = ($line.Line.Trim() -split "\s+")
            if ($parts.Count -ge 5) {
                $pidValue = 0
                if ([int]::TryParse($parts[-1], [ref]$pidValue)) {
                    $pids += $pidValue
                }
            }
        }
    }
    return $pids | Select-Object -Unique
}

$existingPid = $null
if (Test-Path -LiteralPath $PidFile) {
    $pidText = Get-Content -LiteralPath $PidFile -Raw
    $pidValue = 0
    if ([int]::TryParse($pidText.Trim(), [ref]$pidValue)) {
        $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($process) {
            $existingPid = $pidValue
        }
    }
}

$listenerPid = Get-ListeningPidsByPort $Port | Select-Object -First 1

if ($existingPid -or $listenerPid) {
    $id = if ($existingPid) { $existingPid } else { $listenerPid }
    if (-not $existingPid -and $listenerPid) {
        $commandLine = ""
        try {
            $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $id").CommandLine
        } catch {
            $commandLine = ""
        }
        $listenerProcess = Get-Process -Id $id -ErrorAction SilentlyContinue
        $looksLikeDashboard = $commandLine -match "adsb_phase_dbscan\.live_dashboard"
        if (-not $looksLikeDashboard -and $commandLine -eq "" -and $listenerProcess -and $listenerProcess.ProcessName -match "^python") {
            $looksLikeDashboard = $true
        }
        if (-not $looksLikeDashboard) {
            Write-Host "Port $Port is already in use by PID $id, but it does not look like this dashboard."
            Write-Host "Choose another port with -Port, or stop that service yourself."
            exit 1
        }
    }
    Write-Host "Dashboard already appears to be running. PID: $id"
    Write-Host "Open http://127.0.0.1:$Port"
    exit 0
}

$argsList = @("-m", "adsb_phase_dbscan.live_dashboard", "--models-dir", $ModelsDir, "--port", "$Port")
if ($Config -ne "") {
    $argsList += @("--config", "`"$Config`"")
}
if (-not [double]::IsNaN($Lat)) {
    $argsList += @("--lat", "$Lat")
}
if (-not [double]::IsNaN($Lon)) {
    $argsList += @("--lon", "$Lon")
}
if (-not [double]::IsNaN($RadiusNm)) {
    $argsList += @("--radius-nm", "$RadiusNm")
}
$ReplayPath = $null
if ($ReplayInput -ne "") {
    $ReplayPath = Join-Path $Root $ReplayInput
} else {
    $ReplayCandidates = @(
        (Join-Path $Root "..\processed\extracted_days"),
        (Join-Path $Root "..\processed\atl_replay_traffic_segments.jsonl.gz"),
        (Join-Path $Root "..\Data")
    )
    foreach ($candidate in $ReplayCandidates) {
        if (Test-Path -LiteralPath $candidate) {
            $ReplayPath = $candidate
            break
        }
    }
}
if ($ReplayPath -and (Test-Path -LiteralPath $ReplayPath)) {
    $ReplayPath = (Resolve-Path -LiteralPath $ReplayPath).Path
    $RuntimeCacheDir = Join-Path $Root "artifacts\replay_cache\runtime"
    $argsList += @(
        "--replay-input", "`"$ReplayPath`"",
        "--replay-limit-files", "$ReplayLimitFiles",
        "--replay-max-records", "$ReplayMaxRecords",
        "--replay-step-seconds", "$ReplayStepSeconds"
    )
    if ((-not $NoReplayPreload) -and ($ReplayPreload -or (Test-Path -LiteralPath $RuntimeCacheDir))) {
        $argsList += "--replay-preload"
    }
}

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList $argsList `
    -WorkingDirectory $Root `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content -LiteralPath $PidFile -Encoding ASCII
Start-Sleep -Seconds 2
$process.Refresh()
if ($process.HasExited) {
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Dashboard failed to start. Exit code: $($process.ExitCode)"
    if (Test-Path -LiteralPath $ErrLog) {
        Write-Host "Recent errors:"
        Get-Content -LiteralPath $ErrLog -Tail 20
    }
    exit 1
}

Write-Host "Dashboard started. PID: $($process.Id)"
Write-Host "Open http://127.0.0.1:$Port"
Write-Host "Logs: artifacts\runtime\dashboard.err.log"
