param(
    [Alias("Input")]
    [string]$InputPath = "..\processed\extracted_days",
    [string]$Output = "artifacts\training_cache\atl_segments.csv.gz",
    [string]$Config = "config.real.yaml",
    [string]$Python = "python",
    [double]$InputRadiusNm = 120,
    [int]$LimitFiles = 0
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location -LiteralPath $Root

if ($Python -eq "python") {
    $MiniforgePython = Join-Path $env:USERPROFILE "miniforge3\python.exe"
    if (Test-Path -LiteralPath $MiniforgePython) {
        $Python = $MiniforgePython
    }
}

$argsList = @(
    "-m", "adsb_phase_dbscan.build_training_cache",
    "--input", $InputPath,
    "--output", $Output,
    "--config", $Config,
    "--input-radius-nm", "$InputRadiusNm"
)
if ($LimitFiles -gt 0) {
    $argsList += @("--limit-files", "$LimitFiles")
}

& $Python @argsList
