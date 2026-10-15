# Construye la imagen y reproduce los resultados; las salidas quedan en .\outputs
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
docker build -t precio-bolsa-x .
$cmd = @(if ($args.Count -gt 0) { $args } else { "all" })
docker run --rm -v "${PWD}\outputs:/app/outputs" precio-bolsa-x @cmd
