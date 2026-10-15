"""Interfaz de línea de comandos.

    python -m precio_bolsa_x all               entrena, evalúa, explica y verifica (por defecto)
    python -m precio_bolsa_x train|evaluate|shap|verify
    python -m precio_bolsa_x select            repite la selección completa de 36 configuraciones
    python -m precio_bolsa_x predict --fecha 2025-07-31
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from .config import DATE, MODELS, OUTPUTS, TARGET
from . import pipeline


def predict(fecha: str) -> None:
    """Pronóstico un día adelante para una fecha del periodo de estudio, con el modelo oficial o el reentrenado."""
    built = pipeline.build_features(); features = built["feature_sets"][pipeline.CFG["winner"]["feature_set"]]
    path = OUTPUTS / f"gb_huber_{pipeline.CFG['winner']['feature_set']}.joblib"
    model = joblib.load(path if path.exists() else MODELS / "gb_huber_oficial.joblib")
    row = built["frame"][built["frame"][DATE] == pd.Timestamp(fecha)]
    if row.empty or row[features].isna().any(axis=None):
        sys.exit(f"No hay variables completas para {fecha} (rango válido: 2020-01-31 a 2025-07-31).")
    change = float(model.predict(row[features])[0]); previous = float(row["persistence_prediction"].iloc[0])
    print(f"Fecha:                 {fecha}\nPrecio del día previo: {previous:,.2f} COP/kWh\n"
          f"Cambio predicho:       {change:+,.2f} COP/kWh\nPronóstico:            {previous + change:,.2f} COP/kWh\n"
          f"Precio real:           {float(row[TARGET].iloc[0]):,.2f} COP/kWh")


def main() -> None:
    parser = argparse.ArgumentParser(prog="precio_bolsa_x", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="all", choices=["all", "train", "evaluate", "shap", "verify", "select", "predict"])
    parser.add_argument("--fecha", default="2025-07-31")
    parser.add_argument("--n-jobs", type=int, default=2)
    args = parser.parse_args()
    if args.command in ("all", "train"): pipeline.train(args.n_jobs)
    if args.command in ("all", "evaluate"): pipeline.evaluate()
    if args.command in ("all", "shap"): pipeline.explain()
    if args.command == "select": pipeline.select(args.n_jobs)
    if args.command == "predict": predict(args.fecha)
    if args.command in ("all", "verify") and not pipeline.verify(): sys.exit(1)


if __name__ == "__main__":
    main()
