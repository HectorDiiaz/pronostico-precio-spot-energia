"""Rutas, configuración y verificación de integridad del conjunto de datos."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(os.environ.get("PRECIO_BOLSA_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
REFERENCE = ROOT / "reference"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

DATE = "Fecha"
TARGET = "precio_promedio_diario_kwh"


def load_yaml(name: str) -> dict[str, Any]:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_path() -> Path:
    """Devuelve la ruta del dataset tras comprobar su huella SHA-256 registrada."""
    config = load_yaml("dataset.yaml")
    path = ROOT / config["path"]
    actual = sha256_file(path)
    if actual != config["sha256"]:
        raise RuntimeError(f"El SHA-256 del dataset no coincide.\n  esperado: {config['sha256']}\n  obtenido: {actual}\n"
                           "Revise que Git no haya convertido los finales de línea (ver .gitattributes).")
    return path
