# Pronóstico del precio de bolsa de la energía en Colombia con publicaciones de X

Conjunto de datos y modelos del trabajo de grado *Análisis predictivo del precio spot de la energía en Colombia mediante aprendizaje automático sobre publicaciones gubernamentales en la red social X* (Ingeniería de Sistemas, Universidad del Valle).

- **Autor:** Héctor Luis Díaz
- **Director:** Víctor Manuel Vargas Forero

## Qué contiene

El repositorio reúne el conjunto de datos final usado en los experimentos y **dos modelos** que pronostican, con un día de anticipación, el precio promedio diario de bolsa (COP/kWh). Los dos predicen el cambio del precio respecto al día anterior; el pronóstico es `precio de ayer + cambio predicho`.

## Los dos modelos

1. **Modelo de dos etapas: el mejor modelo del trabajo** (menor MAE de prueba entre los 17 modelos evaluados).
2. **Boosting por gradiente con pérdida de Huber: el modelo sobre el que se construyó el anterior.**

**Cómo se relacionan:** el modelo de dos etapas *incluye* un boosting por gradiente con pérdida de Huber como su segunda etapa y le agrega una etapa previa que anticipa los cambios extremos del precio:

```
                       ┌──────────────────────────────────────────┐
  variables del día ──►│ Etapa 1: clasificador de cambios extremos│──► probabilidad de cambio
                       │ (boosting por gradiente)                 │    a la baja, normal o al alza
                       └──────────────────────────────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────────┐
  variables del día ──►│ Etapa 2: boosting por gradiente con      │──► cambio del precio
  + 3 probabilidades   │ pérdida de Huber                         │    precio de ayer + cambio
                       └──────────────────────────────────────────┘
```

El boosting con pérdida de Huber se incluye también **por separado**, tal como se evaluó en el trabajo, para comparar los dos: así se ve cuánto aporta la etapa que anticipa los cambios extremos.

| | Modelo de dos etapas | Boosting con pérdida de Huber |
|---|---|---|
| Papel en el trabajo | Mejor modelo | Modelo de base de la etapa 2 |
| Etapa 1 | Clasificador de boosting por gradiente (`HistGradientBoostingClassifier`) que estima la probabilidad de un cambio extremo: umbrales en los percentiles 5 y 95 del cambio diario de entrenamiento (−42,88 y +40,87 COP/kWh) | No tiene |
| Etapa 2 o modelo | Boosting por gradiente con pérdida de Huber, con las variables más las tres probabilidades de la etapa 1 | Boosting por gradiente con pérdida de Huber |
| Variables | 45: precio rezagado, generación y embalses, ONI, calendario y sumas móviles de las publicaciones de X (combinación PGOC con X ventanas) | 73: precio rezagado, generación y embalses, ONI, calendario, IPC e IPP, y todas las variables de X (combinación PGOCM con X completa) |
| Hiperparámetros del boosting | 100 árboles de profundidad 3, tasa de aprendizaje 0,01, mínimo 15 observaciones por hoja y submuestreo 0,8 | Los mismos |
| MAE de prueba | 63,80 COP/kWh | 64,01 COP/kWh |
| Archivo | `models/dos_etapas_oficial.joblib` | `models/gb_huber_oficial.joblib` |

Cada modelo conserva la combinación de variables con que ganó en el trabajo; por eso no usan las mismas variables.

## Archivos

| Carpeta o archivo | Contenido |
|---|---|
| `data/dataset_final_v1.1.csv` | Conjunto de datos diario integrado: 2.039 días, del 1 de enero de 2020 al 31 de julio de 2025, y 33 columnas |
| `data/DATA_DICTIONARY.md` | Descripción de cada columna: significado, unidad, fuente y transformación |
| `data/datos_prueba.csv` | Variables de los 578 días de prueba (enero de 2024 a julio de 2025), listas para la prueba rápida |
| `models/dos_etapas_oficial.joblib` | Modelo de dos etapas entrenado (clasificador, regresor, variables y umbrales) |
| `models/gb_huber_oficial.joblib` | Boosting por gradiente con pérdida de Huber entrenado |
| `ejecutar_modelo.py` | Prueba rápida de los dos modelos |
| `src/precio_bolsa_x/` | Código para reentrenar, evaluar y verificar los dos modelos |
| `configs/` | Configuración de datos, variables, partición temporal y modelos |
| `reference/` | Cifras oficiales del trabajo de grado, usadas en la verificación |
| `Dockerfile`, `requirements.lock` | Entorno con las versiones exactas de las librerías |

## Requisitos

Una de estas dos opciones:

- **Python 3.12** con las librerías de `requirements.lock`, o
- **Docker**, sin instalar nada más.

## Prueba rápida

Con Python:

```bash
pip install -r requirements.lock
python ejecutar_modelo.py
```

Con Docker:

```bash
docker build -t precio-bolsa-x .
docker run --rm --entrypoint python precio-bolsa-x ejecutar_modelo.py
```

Resultado esperado:

```
Periodo de prueba: 2024-01-01 a 2025-07-31 (578 días)

                                   MAE      RMSE      R2  sMAPE (%)
Modelo de dos etapas           63.8032  148.4463  0.8770    12.0801
Boosting con pérdida de Huber  64.0078  148.6694  0.8766    12.1742
Persistencia (t-1)             64.9125  150.2610  0.8740    12.4394
```

La persistencia es la referencia básica: pronosticar que el precio de mañana será igual al de hoy. Los pronósticos de cada día quedan en `outputs/pronosticos_prueba.csv`.

## Reproducción completa

Reentrena los dos modelos desde el conjunto de datos, con la misma búsqueda de hiperparámetros del trabajo, y compara todo con las cifras oficiales:

- métricas de entrenamiento, validación y prueba, hiperparámetros elegidos y umbrales de cambio extremo;
- pronósticos diarios del modelo de dos etapas en el periodo de prueba;
- intervalo de confianza de su diferencia con la persistencia (bootstrap por bloques de 7 días);
- contraste con y sin las variables de X, con hiperparámetros idénticos, y según si hubo publicaciones en los 7 días previos;
- participación SHAP de cada grupo de variables en las dos etapas.

Con Docker:

```bash
docker run --rm -v "$(pwd)/outputs:/app/outputs" precio-bolsa-x
```

En Windows también puede ejecutarse `.\scripts\reproducir.ps1`.

Sin Docker, en Linux o macOS:

```bash
PYTHONPATH=src python -m precio_bolsa_x
```

En PowerShell, antes de ese comando, se ejecuta `$env:PYTHONPATH="src"`.

La ejecución tarda alrededor de un minuto y debe terminar con:

```
RESULTADO: REPRODUCIDO (95/95 verificaciones)
```

Para un pronóstico de una fecha específica con los dos modelos:

```bash
python -m precio_bolsa_x predict --fecha 2024-10-15
```

## Resultados principales

| Modelo | MAE de validación (2023) | MAE de prueba (enero 2024 – julio 2025) |
|---|---|---|
| Modelo de dos etapas | 47,35 | 63,80 |
| Boosting con pérdida de Huber | 47,48 | 64,01 |
| Persistencia | 48,44 | 64,91 |

Valores en COP/kWh.

- **Frente a la persistencia:** el modelo de dos etapas redujo el MAE de prueba en 1,11 COP/kWh, con un intervalo de confianza del 95 % de 0,15 a 2,06.
- **Aporte de X:** sin las variables de X, el MAE de prueba del modelo de dos etapas aumentó de 63,80 a 64,12 COP/kWh (intervalo de confianza del 95 % de la diferencia: 0,08 a 0,55). La mejora se concentró en los días con publicaciones en la semana previa.

## Alcance

En el trabajo de grado se evaluaron 17 modelos en 2.074 experimentos. Este paquete no repite esa evaluación completa: reproduce las configuraciones ganadoras de los dos modelos incluidos, con su búsqueda de hiperparámetros, y los análisis del mejor modelo.

## Conjunto de datos

Cada fila corresponde a un día e integra:

- el precio promedio diario de bolsa (COP/kWh), que es la variable objetivo;
- la participación de cada tecnología en la generación y el nivel de los embalses del Sistema Interconectado Nacional (XM);
- el Índice Oceánico de El Niño (ONI) y la categoría ENSO derivada de él (NOAA);
- el Índice de Precios al Consumidor (IPC) y el Índice de Precios del Productor (IPP) (DANE);
- variables de calendario: día de la semana, fin de semana, festivo y días no laborales consecutivos;
- variables diarias de las publicaciones de la cuenta @petrogustavo en X y de los mensajes que citó o respondió: conteos de publicaciones seleccionadas por categoría (positiva, neutra o negativa según la dirección del precio con la que se asocian), sus porcentajes y el balance diario.

El conjunto de datos no tiene valores faltantes. Los días sin publicaciones seleccionadas toman valor cero en las variables de X.

Huella SHA-256 del archivo:

```
8bd796daad2c70764c6401af64746ec84f8c1b2a80ab24df060a1bed9bef30ce
```

El código comprueba esta huella antes de cada ejecución y se detiene si el archivo cambió. El archivo `.gitattributes` evita que Git altere los finales de línea del CSV, para que la huella se conserve.

## Notas

- **Textos de X:** el conjunto de datos contiene solo agregados diarios de las publicaciones, no sus textos.
- **Uso como predictores:** cada columna guarda el valor del propio día. En los modelos, las variables que no se conocen antes del pronóstico se usan rezagadas, y las ventanas de X suman solo días anteriores al pronosticado. Las columnas de control (`check_pct_sum` y `check_pct_sentimientos`) y `precio_log` no se usan como predictores.
- **Interpretación:** los resultados miden capacidad predictiva, no relaciones de causalidad.
