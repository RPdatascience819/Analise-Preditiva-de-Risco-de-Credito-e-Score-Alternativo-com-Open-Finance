param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$KaggleArgs
)

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:KAGGLE_CONFIG_DIR = Join-Path $projectRoot ".kaggle"
$env:KAGGLE_API_TOKEN = Join-Path $env:KAGGLE_CONFIG_DIR "access_token"

& (Join-Path $projectRoot "venv\Scripts\kaggle.exe") @KaggleArgs
