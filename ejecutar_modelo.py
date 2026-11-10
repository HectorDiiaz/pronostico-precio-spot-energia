"""Ejecuta los dos modelos entrenados sobre los datos de prueba (enero 2024 – julio 2025).

Uso:
    python ejecutar_modelo.py

Carga el modelo de dos etapas (mejor modelo del trabajo) y el boosting por gradiente con pérdida de Huber,
pronostica cada día del periodo de prueba y compara ambos con la persistencia (repetir el precio del día
anterior). Solo requiere pandas, numpy, scikit-learn y joblib (ver requirements.lock).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
datos = pd.read_csv(ROOT / "data" / "datos_prueba.csv", parse_dates=["Fecha"])
dos_etapas = joblib.load(ROOT / "models" / "dos_etapas_oficial.joblib")
gb_huber = joblib.load(ROOT / "models" / "gb_huber_oficial.joblib")

# Los dos modelos predicen el cambio diario; el pronóstico es el precio del día anterior más ese cambio.
# Modelo de dos etapas: la etapa 1 estima la probabilidad de un cambio extremo (a la baja, normal o al alza)
# y la etapa 2 predice el cambio con las variables más esas tres probabilidades.
variables = dos_etapas["variables"]
probabilidades = dos_etapas["clasificador"].predict_proba(datos[variables])
cambio = dos_etapas["regresor"].predict(np.column_stack([datos[variables].to_numpy(float), probabilidades]))
datos["pronostico_dos_etapas"] = datos["precio_dia_anterior"] + cambio
datos["pronostico_gb_huber"] = datos["precio_dia_anterior"] + gb_huber.predict(datos[list(gb_huber.feature_names_in_)])


def metricas(real: pd.Series, pronostico: pd.Series) -> dict:
    error = real - pronostico
    return {"MAE": np.mean(np.abs(error)), "RMSE": np.sqrt(np.mean(error ** 2)),
            "R2": 1 - np.sum(error ** 2) / np.sum((real - real.mean()) ** 2),
            "sMAPE (%)": np.mean(200 * np.abs(error) / (np.abs(real) + np.abs(pronostico)))}


tabla = pd.DataFrame({"Modelo de dos etapas": metricas(datos.precio_real, datos.pronostico_dos_etapas),
                      "Boosting con pérdida de Huber": metricas(datos.precio_real, datos.pronostico_gb_huber),
                      "Persistencia (t-1)": metricas(datos.precio_real, datos.precio_dia_anterior)}).T
print(f"Periodo de prueba: {datos.Fecha.min():%Y-%m-%d} a {datos.Fecha.max():%Y-%m-%d} ({len(datos)} días)\n")
print(tabla.round(4).to_string())
for nombre in tabla.index[:2]:
    mejora = (tabla.MAE.iloc[2] - tabla.loc[nombre, "MAE"]) / tabla.MAE.iloc[2] * 100
    print(f"Mejora del MAE frente a la persistencia ({nombre}): {mejora:.2f} %")

salida = ROOT / "outputs" / "pronosticos_prueba.csv"
salida.parent.mkdir(exist_ok=True)
datos[["Fecha", "precio_real", "precio_dia_anterior", "pronostico_dos_etapas", "pronostico_gb_huber"]].to_csv(
    salida, index=False, date_format="%Y-%m-%d")
print(f"\nPronósticos guardados en {salida.relative_to(ROOT)}")
