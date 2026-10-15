# Pronóstico del precio de bolsa de la energía en Colombia con señales de X

Modelo seleccionado en el trabajo de grado *Análisis predictivo del precio spot de la energía en Colombia mediante aprendizaje automático sobre publicaciones gubernamentales en la red social X* (Ingeniería de Sistemas, Universidad del Valle).

- **Autor:** Héctor Luis Díaz
- **Director:** Víctor Manuel Vargas Forero

## ¿Qué es?

Es un modelo que pronostica, con un día de anticipación, el precio promedio diario de bolsa de la energía en Colombia (COP/kWh).

- **Algoritmo:** Gradient Boosting con pérdida de Huber.
- **Qué predice:** el cambio del precio respecto al día anterior. El pronóstico final es `precio de ayer + cambio predicho`.
- **Variables que usa (73 en total):**
  - precios de días anteriores;
  - composición de la generación y nivel de embalses;
  - índice ONI (fenómeno de El Niño);
  - calendario;
  - IPC e IPP;
  - variables construidas a partir de las publicaciones de la cuenta @petrogustavo en X.

## ¿Qué contiene?

| Carpeta o archivo | Contenido |
|---|---|
| `models/gb_huber_oficial.joblib` | Modelo entrenado |
| `data/datos_prueba.csv` | Datos de prueba listos para usar (578 días, de enero de 2024 a julio de 2025) |
| `data/dataset_final_v1.1.csv` | Dataset completo 2020–2025 (2.039 días) |
| `data/DATA_DICTIONARY.md` | Descripción de las columnas del dataset |
| `ejecutar_modelo.py` | Prueba rápida del modelo |
| `src/precio_bolsa_x/` | Código para reentrenar y verificar el modelo |
| `configs/` | Configuración de datos, variables, partición temporal y modelo |
| `reference/` | Resultados oficiales de la tesis, usados para la verificación |
| `Dockerfile`, `requirements.lock` | Entorno con las versiones exactas de las librerías |

## Requisitos

Necesita una de estas dos opciones:

- **Python 3.12** con las librerías de `requirements.lock`, o
- **Docker**, sin instalar nada más.

## Cómo probarlo

### Opción 1: prueba rápida con Python

```bash
pip install -r requirements.lock
python ejecutar_modelo.py
```

### Opción 2: prueba rápida con Docker

```bash
docker build -t precio-bolsa-x .
docker run --rm --entrypoint python precio-bolsa-x ejecutar_modelo.py
```

### Resultado esperado

```
Periodo de prueba: 2024-01-01 a 2025-07-31 (578 días)

                             MAE      RMSE      R2  sMAPE (%)
Gradient Boosting Huber  64.0078  148.6694  0.8766    12.1742
Persistencia (t-1)       64.9125  150.2610  0.8740    12.4394
```

La persistencia es la referencia básica: pronosticar que mañana el precio será igual al de hoy. Los pronósticos de cada día quedan en `outputs/pronosticos_prueba.csv`.

## Reproducción completa (opcional)

Reentrena el modelo desde el dataset, lo evalúa, calcula la importancia de las variables (SHAP) y compara todo con los resultados oficiales de la tesis.

**Con Docker:**

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" precio-bolsa-x
```

En Windows también puede ejecutarse `.\scripts\reproducir.ps1`.

**Sin Docker, en Linux o macOS:**

```bash
PYTHONPATH=src python -m precio_bolsa_x
```

En PowerShell, antes de ese comando, se ejecuta `$env:PYTHONPATH="src"`.

La ejecución tarda menos de un minuto y debe terminar con:

```
RESULTADO: REPRODUCIDO (39/39 verificaciones)
```

Otros comandos disponibles:

| Comando | Qué hace |
|---|---|
| `python -m precio_bolsa_x predict --fecha 2024-10-15` | Pronóstico para una fecha específica |
| `python -m precio_bolsa_x select` | Repite la selección entre las 36 configuraciones evaluadas |

## Resultados principales

| Periodo | MAE del modelo | MAE de la persistencia |
|---|---|---|
| Validación (2023) | 47,48 | 48,44 |
| Prueba (enero 2024 – julio 2025) | 64,01 | 64,91 |

Valores en COP/kWh.

- **Frente a la persistencia:** el modelo fue el único que la superó en validación y en prueba.
- **Aporte de X:** incluir las variables de X redujo el error en prueba en 0,11 COP/kWh, con un intervalo de confianza del 95 % de [0,04; 0,18].

## Notas

- **Integridad de los datos:** el dataset se verifica con su huella SHA-256 antes de cada ejecución. Si el archivo cambia, el programa se detiene.
- **Contenido de X:** el dataset contiene solo agregados diarios de las publicaciones (conteos y sentimiento), no sus textos.
- **Interpretación:** los resultados miden capacidad predictiva, no relaciones de causalidad.
