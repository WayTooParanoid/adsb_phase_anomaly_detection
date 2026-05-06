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
    [switch]$NoReplayPreload,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$CommandDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $CommandDir
$StopScript = Join-Path $CommandDir "stop-dashboard.ps1"
$StartScript = Join-Path $CommandDir "start-dashboard.ps1"

& $StopScript -Port $Port

$startArgs = @{
    Port = $Port
    Python = $Python
    ModelsDir = $ModelsDir
    ReplayLimitFiles = $ReplayLimitFiles
    ReplayMaxRecords = $ReplayMaxRecords
    ReplayStepSeconds = $ReplayStepSeconds
}
if ($Config -ne "") {
    $startArgs.Config = $Config
}
if (-not [double]::IsNaN($Lat)) {
    $startArgs.Lat = $Lat
}
if (-not [double]::IsNaN($Lon)) {
    $startArgs.Lon = $Lon
}
if (-not [double]::IsNaN($RadiusNm)) {
    $startArgs.RadiusNm = $RadiusNm
}
if ($ReplayInput -ne "") {
    $startArgs.ReplayInput = $ReplayInput
}
if ($ReplayPreload) {
    $startArgs.ReplayPreload = $true
}
if ($NoReplayPreload) {
    $startArgs.NoReplayPreload = $true
}

& $StartScript @startArgs

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$Port"
}
