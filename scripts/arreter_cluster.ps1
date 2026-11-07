$ErrorActionPreference="Stop"
$Root=Split-Path -Parent $PSScriptRoot
Push-Location $Root
try { docker compose down --remove-orphans } finally { Pop-Location }
