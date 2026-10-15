"""Ejecuta el modelo entrenado sobre los datos de prueba (enero 2024 – julio 2025).

Uso:
    python ejecutar_modelo.py

Carga models/gb_huber_oficial.joblib, pronostica cada día del periodo de prueba y
compara el resultado con la persistencia (repetir el precio del día anterior).
Solo requiere pandas, numpy, scikit-learn y joblib (ver requirements.lock).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
datos = pd.read_csv(ROOT / "data" / "datos_prueba.csv", parse_dates=["Fecha"])
modelo = joblib.load(ROOT / "models" / "gb_huber_oficial.joblib")

# El modelo predice el cambio diario; el pronóstico es el precio del día anterior más ese cambio.
variables = list(modelo.feature_names_in_)
datos["cambio_predicho"] = modelo.predict(datos[variables])
datos["pronostico"] = datos["precio_dia_anterior"] + datos["cambio_predicho"]


def metricas(real: pd.Series, pronostico: pd.Series) -> dict:
    error = real - pronostico
    return {"MAE": np.mean(np.abs(error)), "RMSE": np.sqrt(np.mean(error ** 2)),
            "R2": 1 - np.sum(error ** 2) / np.sum((real - real.mean()) ** 2),
            "sMAPE (%)": np.mean(200 * np.abs(error) / (np.abs(real) + np.abs(pronostico)))}


tabla = pd.DataFrame({"Gradient Boosting Huber": metricas(datos.precio_real, datos.pronostico),
                      "Persistencia (t-1)": metricas(datos.precio_real, datos.precio_dia_anterior)}).T
print(f"Periodo de prueba: {datos.Fecha.min():%Y-%m-%d} a {datos.Fecha.max():%Y-%m-%d} ({len(datos)} días)\n")
print(tabla.round(4).to_string())
mejora = (tabla.MAE.iloc[1] - tabla.MAE.iloc[0]) / tabla.MAE.iloc[1] * 100
print(f"\nMejora del MAE frente a la persistencia: {mejora:.2f} %  (valor de la tesis: MAE 64,0078 COP/kWh)")

salida = ROOT / "outputs" / "pronosticos_prueba.csv"
salida.parent.mkdir(exist_ok=True)
datos[["Fecha", "precio_real", "precio_dia_anterior", "cambio_predicho", "pronostico"]].to_csv(salida, index=False, date_format="%Y-%m-%d")
print(f"Pronósticos guardados en {salida.relative_to(ROOT)}")
