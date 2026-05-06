param(
    [string]$Python = "",
    [string]$Config = "config.real.yaml",
    [string]$DataDir = "..\Data",
    [string]$Out = "artifacts\replay_cache",
    [double]$RadiusNm = [double]::NaN,
    [string[]]$Day = @(),
    [int]$Workers = 12,
    [int]$DayWorkers = 2,
    [int]$LimitFiles = 0,
    [int]$MaxTracks = 0,
    [switch]$Runtime,
    [switch]$RawCache,
    [switch]$Processes,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$env:PYTHONDONTWRITEBYTECODE = "1"

if ($Python -eq "") {
    $KnownPython = "c:\Users\venro\miniforge3\python.exe"
    if (Test-Path -LiteralPath $KnownPython) {
        $Python = $KnownPython
    } else {
        $Python = "python"
    }
}

$argsList = @(
    "-m", "adsb_phase_dbscan.build_replay_cache",
    "--config", $Config,
    "--data-dir", $DataDir,
    "--out", $Out,
    "--workers", "$Workers",
    "--day-workers", "$DayWorkers"
)

if (-not [double]::IsNaN($RadiusNm)) {
    $argsList += @("--radius-nm", "$RadiusNm")
}
if ($LimitFiles -gt 0) {
    $argsList += @("--limit-files", "$LimitFiles")
}
if ($MaxTracks -gt 0) {
    $argsList += @("--max-tracks", "$MaxTracks")
}
foreach ($item in $Day) {
    if ($item -ne "") {
        $argsList += @("--day", $item)
    }
}
if ($Force) {
    $argsList += "--force"
}
if ($Runtime -or -not $RawCache) {
    $argsList += "--runtime"
}
if ($Processes) {
    $argsList += "--processes"
}

Push-Location $Root
try {
    & $Python @argsList
} finally {
    Pop-Location
}
