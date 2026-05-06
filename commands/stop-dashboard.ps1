param(
    [int]$Port = 8050
)

$ErrorActionPreference = "SilentlyContinue"
$CommandDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $CommandDir
$RuntimeDir = Join-Path $Root "artifacts\runtime"
$PidFile = Join-Path $RuntimeDir "dashboard.pid"
$stopped = @()

function Get-ProcessCommandLine([int]$ProcessId) {
    try {
        return (Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId").CommandLine
    } catch {
        return ""
    }
}

function Stop-DashboardProcess([int]$ProcessId, [bool]$RequireCommandMatch) {
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if (-not $process) {
        return $false
    }
    if ($RequireCommandMatch) {
        $commandLine = Get-ProcessCommandLine $ProcessId
        $looksLikeDashboard = $commandLine -match "adsb_phase_dbscan\.live_dashboard"
        if (-not $looksLikeDashboard -and $commandLine -eq "" -and $process.ProcessName -match "^python") {
            $looksLikeDashboard = $true
        }
        if (-not $looksLikeDashboard) {
            Write-Host "PID $ProcessId is listening on the port but does not look like the dashboard. Not stopping it."
            return $false
        }
    }
    Stop-Process -Id $ProcessId -ErrorAction SilentlyContinue
    return $true
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

if (Test-Path -LiteralPath $PidFile) {
    $pidText = Get-Content -LiteralPath $PidFile -Raw
    $pidValue = 0
    if ([int]::TryParse($pidText.Trim(), [ref]$pidValue)) {
        if (Stop-DashboardProcess $pidValue $false) {
            $stopped += $pidValue
        }
    }
    Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
}

$listenerPids = Get-ListeningPidsByPort $Port
foreach ($listenerPid in $listenerPids) {
    if (Stop-DashboardProcess $listenerPid $true) {
        $stopped += $listenerPid
    }
}

if ($stopped.Count -gt 0) {
    Write-Host "Stopped dashboard process(es): $(($stopped | Select-Object -Unique) -join ', ')"
} else {
    Write-Host "No dashboard process was running."
}
