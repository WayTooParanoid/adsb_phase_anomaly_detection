param(
    [Alias("Input")]
    [string]$InputPath = "artifacts\training_cache\atl_segments.csv.gz",
    [string]$Config = "config.real.yaml",
    [string]$OutputDir = "artifacts\models",
    [string]$Python = "python",
    [int]$MaxSegmentsPerPhase = 5000
)

$ErrorActionPreference = "Stop"
$CommandDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $CommandDir
Set-Location -LiteralPath $Root

if ($Python -eq "python") {
    $MiniforgePython = Join-Path $env:USERPROFILE "miniforge3\python.exe"
    if (Test-Path -LiteralPath $MiniforgePython) {
        $Python = $MiniforgePython
    }
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    Write-Host "Training cache not found: $InputPath" -ForegroundColor Red
    Write-Host "Build it first with: build-training-cache.cmd"
    exit 1
}

if (-not (Test-Path -LiteralPath $Config)) {
    Write-Host "Config not found: $Config" -ForegroundColor Red
    exit 1
}

Write-Host "Training ADS-B DBSCAN models"
Write-Host "Input:  $InputPath"
Write-Host "Config: $Config"
Write-Host "Output: $OutputDir"
Write-Host ""

& $Python -m adsb_phase_dbscan.train `
    --input $InputPath `
    --config $Config `
    --output-dir $OutputDir `
    --max-segments-per-phase $MaxSegmentsPerPhase

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Training complete. Restart the dashboard with: start_dashboard.cmd" -ForegroundColor Green
