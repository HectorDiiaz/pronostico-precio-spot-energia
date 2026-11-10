"""Interfaz de línea de comandos.

    python -m precio_bolsa_x all               entrena, evalúa, explica y verifica (por defecto)
    python -m precio_bolsa_x train|evaluate|shap|verify
    python -m precio_bolsa_x predict --fecha 2025-07-31
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from . import pipeline
from .config import DATE, MODELS, OUTPUTS, TARGET


def _load(name: str):
    """Modelo reentrenado en outputs/ si existe; si no, el modelo oficial de models/."""
    path = OUTPUTS / f"{name}.joblib"
    return joblib.load(path if path.exists() else MODELS / f"{name}_oficial.joblib")


def predict(fecha: str) -> None:
    """Pronóstico un día adelante de los dos modelos para una fecha del periodo de estudio."""
    built = pipeline.build_features(); frame = built["frame"]
    row = frame[frame[DATE] == pd.Timestamp(fecha)]
    two_stage, gb = _load("dos_etapas"), _load("gb_huber")
    gb_features = built["feature_sets"][pipeline.CFG["gb_huber"]["feature_set"]]
    if row.empty or row[list(dict.fromkeys(two_stage["variables"] + gb_features))].isna().any(axis=None):
        sys.exit(f"No hay variables completas para {fecha} (rango válido: 2020-03-01 a 2025-07-31).")
    previous = float(row["persistence_prediction"].iloc[0])
    forecasts = {"Modelo de dos etapas": float(pipeline.predict_two_stage(two_stage, row)[0]),
                 "Boosting con pérdida de Huber": float(pipeline.predict_gb_huber(gb, row, gb_features)[0])}
    print(f"Fecha: {fecha}\nPrecio del día previo: {previous:,.2f} COP/kWh")
    for name, value in forecasts.items():
        print(f"Pronóstico, {name}: {value:,.2f} COP/kWh (cambio {value - previous:+,.2f})")
    print(f"Precio real: {float(row[TARGET].iloc[0]):,.2f} COP/kWh")


def main() -> None:
    parser = argparse.ArgumentParser(prog="precio_bolsa_x", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=["all", "train", "evaluate", "shap", "verify", "predict"])
    parser.add_argument("--fecha", default="2025-07-31")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.command in ("all", "train"): pipeline.train(args.n_jobs)
    if args.command in ("all", "evaluate"): pipeline.evaluate()
    if args.command in ("all", "shap"): pipeline.explain()
    if args.command == "predict": predict(args.fecha)
    if args.command in ("all", "verify") and not pipeline.verify(): sys.exit(1)


if __name__ == "__main__":
    main()
