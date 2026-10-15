#!/usr/bin/env sh
# Construye la imagen y reproduce los resultados; las salidas quedan en ./outputs
set -e
cd "$(dirname "$0")/.."
[ $# -eq 0 ] && set -- all
docker build -t precio-bolsa-x .
docker run --rm -v "$(pwd)/outputs:/app/outputs" precio-bolsa-x "$@"
