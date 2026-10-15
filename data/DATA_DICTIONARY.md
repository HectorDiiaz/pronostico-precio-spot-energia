# Diccionario de datos — dataset oficial v1.0

Este diccionario describe las 33 columnas de `dataset_final_v1.0.csv`. “Condicional”
en Predictor exige respetar disponibilidad real, horizonte y causalidad; un valor
rezagado puede ser válido aunque el contemporáneo no lo sea.

| # | Nombre | Clase | Descripción | Tipo conceptual | Unidad | Fuente | Transformación | Periodicidad | Predictor | Observaciones metodológicas |
|---:|---|---|---|---|---|---|---|---|---|---|
| 1 | `Fecha` | CALENDARIO | Fecha civil de la observación. | Fecha | YYYY-MM-DD | Calendario generado | Secuencia diaria oficial. | Diaria | No | Índice temporal y clave de unión; no aleatorizar. |
| 2 | `precio_promedio_diario_kwh` | TARGET | Precio spot promedio diario de bolsa. | Continua | COP/kWh | Precio procesado | Promedio diario consolidado. | Diaria | No | Objetivo oficial; como entrada, solo rezagado. |
| 3 | `precio_log` | PRECIO DERIVADO | Transformación logarítmica del objetivo. | Continua derivada | log(1 + COP/kWh) | Derivada del target | `numpy.log1p(precio_promedio_diario_kwh)`. | Diaria | No | Admisible como objetivo transformado o rezagado; contemporáneo causa leakage. |
| 4 | `pct_hidraulica` | GENERACIÓN | Participación hidráulica en la generación diaria. | Proporción | % | Generación procesada | Energía hidráulica / total × 100. | Diaria | Condicional | Verificar disponibilidad al instante de pronóstico. |
| 5 | `pct_termica` | GENERACIÓN | Participación térmica en la generación diaria. | Proporción | % | Generación procesada | Energía térmica / total × 100. | Diaria | Condicional | Variable composicional. |
| 6 | `pct_solar` | GENERACIÓN | Participación solar en la generación diaria. | Proporción | % | Generación procesada | Energía solar / total × 100. | Diaria | Condicional | Evitar observación contemporánea si se conoce tras el cierre. |
| 7 | `pct_eolica` | GENERACIÓN | Participación eólica en la generación diaria. | Proporción | % | Generación procesada | Energía eólica / total × 100. | Diaria | Condicional | Sujeta a disponibilidad temporal. |
| 8 | `pct_cogenerador` | GENERACIÓN | Participación de cogeneradores en la generación diaria. | Proporción | % | Generación procesada | Energía de cogeneradores / total × 100. | Diaria | Condicional | Sujeta a disponibilidad temporal. |
| 9 | `kwh_hidraulica` | GENERACIÓN | Energía diaria de fuente hidráulica. | Continua | kWh/día | Generación procesada | Suma diaria por tecnología. | Diaria | Condicional | Confirmar escala/unidad del proveedor y disponibilidad ex ante. |
| 10 | `kwh_termica` | GENERACIÓN | Energía diaria de fuente térmica. | Continua | kWh/día | Generación procesada | Suma diaria por tecnología. | Diaria | Condicional | Confirmar escala/unidad del proveedor y disponibilidad ex ante. |
| 11 | `kwh_solar` | GENERACIÓN | Energía diaria de fuente solar. | Continua | kWh/día | Generación procesada | Suma diaria por tecnología. | Diaria | Condicional | Confirmar escala/unidad del proveedor y disponibilidad ex ante. |
| 12 | `kwh_eolica` | GENERACIÓN | Energía diaria de fuente eólica. | Continua | kWh/día | Generación procesada | Suma diaria por tecnología. | Diaria | Condicional | Confirmar escala/unidad del proveedor y disponibilidad ex ante. |
| 13 | `kwh_cogenerador` | GENERACIÓN | Energía diaria de cogeneradores. | Continua | kWh/día | Generación procesada | Suma diaria por tecnología. | Diaria | Condicional | Confirmar escala/unidad del proveedor y disponibilidad ex ante. |
| 14 | `check_pct_sum` | CONTROL / VALIDACIÓN | Suma de las cinco participaciones de generación. | Continua de control | % | Generación procesada | Suma de columnas `pct_*`; esperado ≈100. | Diaria | No | Control de calidad; excluir por defecto. |
| 15 | `n_negativo` | TWITTER/X | Publicaciones clasificadas como negativas. | Conteo | publicaciones/día | Twitter/X procesado | Conteo diario; cero sin publicaciones. | Diaria | Condicional | Solo información disponible antes del corte. |
| 16 | `n_neutro` | TWITTER/X | Publicaciones clasificadas como neutras. | Conteo | publicaciones/día | Twitter/X procesado | Conteo diario; cero sin publicaciones. | Diaria | Condicional | Cero significa ausencia, no dato perdido. |
| 17 | `n_positivo` | TWITTER/X | Publicaciones clasificadas como positivas. | Conteo | publicaciones/día | Twitter/X procesado | Conteo diario; cero sin publicaciones. | Diaria | Condicional | Solo información conocida al corte. |
| 18 | `n_total_tweets` | TWITTER/X | Total diario de publicaciones clasificadas. | Conteo | publicaciones/día | Twitter/X procesado | Suma de los tres conteos. | Diaria | Condicional | Indicador de volumen y de ausencia de publicaciones. |
| 19 | `pct_negativo` | TWITTER/X | Porcentaje diario negativo. | Proporción | % | Twitter/X procesado | `n_negativo / n_total_tweets × 100`; cero sin publicaciones. | Diaria | Condicional | Interpretar junto con `n_total_tweets`. |
| 20 | `pct_neutro` | TWITTER/X | Porcentaje diario neutro. | Proporción | % | Twitter/X procesado | `n_neutro / n_total_tweets × 100`; cero sin publicaciones. | Diaria | Condicional | Interpretar junto con el volumen. |
| 21 | `pct_positivo` | TWITTER/X | Porcentaje diario positivo. | Proporción | % | Twitter/X procesado | `n_positivo / n_total_tweets × 100`; cero sin publicaciones. | Diaria | Condicional | Interpretar junto con el volumen. |
| 22 | `sentiment_index` | TWITTER/X | Balance positivo menos negativo. | Índice continuo | puntos porcentuales | Twitter/X procesado | `pct_positivo - pct_negativo`; rango −100 a 100. | Diaria | Condicional | Cero puede ser balance/neutralidad o ausencia de publicaciones. |
| 23 | `check_pct_sentimientos` | CONTROL / VALIDACIÓN | Suma de los tres porcentajes de sentimiento. | Continua de control | % | Twitter/X procesado | Suma de porcentajes; cero sin publicaciones. | Diaria | No | Control de calidad; excluir por defecto. |
| 24 | `oni` | CLIMA / ENSO | Oceanic Niño Index asociado a la fecha. | Continua | °C de anomalía | ENSO procesado | Valor mensual expandido diariamente. | Diaria desde mensual | Sí | Respetar fecha real de publicación si el escenario lo exige. |
| 25 | `enso_index` | CLIMA / ENSO | Código ordinal de la categoría ENSO. | Categórica ordinal codificada | entero | ENSO procesado | Clasificación de `oni` en niveles. | Diaria desde mensual | Condicional | Validar tabla código–categoría; tratar como categórica si no se justifica distancia ordinal. |
| 26 | `enso_categoria` | CLIMA / ENSO | Etiqueta textual del estado e intensidad ENSO. | Categórica | categoría | ENSO procesado | Clasificación textual derivada de `oni`. | Diaria desde mensual | Sí | Codificación aprendida exclusivamente con TRAIN. |
| 27 | `nivel_embalses_pct` | EMBALSES | Volumen útil agregado diario de embalses del SIN. | Continua | % | Reservas procesadas V2 | Lectura anual, fechas explícitas y concatenación. | Diaria | Condicional | V2 corrige 132 asociaciones día/mes de 2023; verificar disponibilidad. |
| 28 | `ipp_var` | MACROECONÓMICAS | Nivel mensual de un índice denominado IPP; el nombre físico se conserva por compatibilidad y no significa variación porcentual. | Índice continuo | puntos de índice; base no confirmada | `data/processed/ipc_ipp/IPP_Data_v2.xlsx`, reconstruido desde los 67 valores de `GenerateIPP.ipynb` | Cada valor mensual se repite en todas las fechas del mismo `year` y `month`, sin interpolación. | Original mensual; almacenada diariamente | Condicional | Fuente, cobertura y base exactas requieren confirmación documental externa; respetar el rezago real de publicación. |
| 29 | `ipc` | MACROECONÓMICAS | Serie diaria derivada del IPC. | Índice continuo | puntos; base no confirmada | IPC procesado | Valor mensual expandido diariamente. | Diaria desde mensual | Condicional | Respetar rezago de publicación; falta confirmar año/base. |
| 30 | `dia_semana` | CALENDARIO | Día de semana, lunes=0 y domingo=6. | Categórica cíclica codificada | entero 0–6 | Calendario generado | `Fecha.dayofweek`. | Diaria | Sí | Preferir codificación categórica o seno/coseno. |
| 31 | `es_fin_semana` | CALENDARIO | Indicador de sábado o domingo. | Binaria | 0/1 | Calendario generado | 1 si `dia_semana >= 5`. | Diaria | Sí | Conocida ex ante. |
| 32 | `es_festivo` | CALENDARIO | Indicador de festivo nacional colombiano. | Binaria | 0/1 | Calendario generado | Fechas fijas, Ley Emiliani y días relativos a Pascua. | Diaria | Sí | No incluye necesariamente festividades regionales. |
| 33 | `dias_no_laborales_consecutivos` | CALENDARIO | Longitud del bloque continuo de fin de semana/festivo. | Entera derivada | días | Calendario generado | Tamaño del bloque no laboral; cero en días laborales. | Diaria | Condicional | Conocida ex ante si se genera solo con reglas de calendario. |

## Exclusiones y cautelas predeterminadas

- `check_pct_sum` y `check_pct_sentimientos` son controles, no predictores.
- `precio_log` es derivada del objetivo: solo objetivo transformado o valor rezagado.
- El objetivo contemporáneo tampoco puede entrar como predictor.
- Generación, Twitter/X, embalses y macroeconómicas deben alinearse con su instante
  real de disponibilidad.
- Toda transformación aprendida se ajusta exclusivamente con TRAIN, según
  `docs/thesis/EXPERIMENTAL_PROTOCOL.md`.

## Semánticas pendientes de confirmación

1. `ipp_var`: ya se confirmó que es un nivel y no una variación; siguen pendientes
   la fuente externa exacta, cobertura, periodo base y calendario de publicación.
2. `ipc`: confirmar año/base y definición oficial exacta.
3. `enso_index`: formalizar la tabla completa código–`enso_categoria`.
4. `kwh_*`: confirmar con metadatos del proveedor la unidad y escala exactas.
