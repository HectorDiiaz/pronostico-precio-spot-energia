# Diccionario de datos: `dataset_final_v1.1.csv`

El archivo tiene 2.039 filas, una por día del 1 de enero de 2020 al 31 de julio de 2025, y 33 columnas. No tiene valores faltantes.

| # | Columna | Grupo | Descripción | Unidad | Fuente | Transformación |
|---:|---|---|---|---|---|---|
| 1 | `Fecha` | Calendario | Fecha de la observación | AAAA-MM-DD | Construcción propia | Secuencia diaria continua |
| 2 | `precio_promedio_diario_kwh` | Precio | Precio promedio diario de bolsa; variable objetivo | COP/kWh | XM | Promedio de los precios horarios del día |
| 3 | `precio_log` | Precio | Logaritmo del precio | log(1 + COP/kWh) | Derivada del precio | `log1p(precio_promedio_diario_kwh)` |
| 4 | `pct_hidraulica` | Generación | Participación hidráulica en la generación del día | % | XM | Energía hidráulica / total × 100 |
| 5 | `pct_termica` | Generación | Participación térmica | % | XM | Energía térmica / total × 100 |
| 6 | `pct_solar` | Generación | Participación solar | % | XM | Energía solar / total × 100 |
| 7 | `pct_eolica` | Generación | Participación eólica | % | XM | Energía eólica / total × 100 |
| 8 | `pct_cogenerador` | Generación | Participación de la cogeneración | % | XM | Energía de cogeneración / total × 100 |
| 9 | `kwh_hidraulica` | Generación | Energía hidráulica del día | kWh | XM | Suma diaria de las plantas de la tecnología |
| 10 | `kwh_termica` | Generación | Energía térmica del día | kWh | XM | Suma diaria |
| 11 | `kwh_solar` | Generación | Energía solar del día | kWh | XM | Suma diaria |
| 12 | `kwh_eolica` | Generación | Energía eólica del día | kWh | XM | Suma diaria |
| 13 | `kwh_cogenerador` | Generación | Energía de cogeneración del día | kWh | XM | Suma diaria |
| 14 | `check_pct_sum` | Control | Suma de las cinco participaciones de generación (cercana a 100) | % | Derivada | Suma de las columnas `pct_*` de generación |
| 15 | `n_negativo` | X | Publicaciones seleccionadas clasificadas como negativas (asociadas con un posible aumento del precio) | publicaciones | X | Conteo diario |
| 16 | `n_neutro` | X | Publicaciones seleccionadas clasificadas como neutras (sin dirección definida respecto del precio) | publicaciones | X | Conteo diario |
| 17 | `n_positivo` | X | Publicaciones seleccionadas clasificadas como positivas (asociadas con una posible disminución del precio) | publicaciones | X | Conteo diario |
| 18 | `n_total_tweets` | X | Total de publicaciones seleccionadas del día | publicaciones | X | Suma de los tres conteos |
| 19 | `pct_negativo` | X | Porcentaje de publicaciones negativas | % | X | `n_negativo / n_total_tweets × 100` |
| 20 | `pct_neutro` | X | Porcentaje de publicaciones neutras | % | X | `n_neutro / n_total_tweets × 100` |
| 21 | `pct_positivo` | X | Porcentaje de publicaciones positivas | % | X | `n_positivo / n_total_tweets × 100` |
| 22 | `sentiment_index` | X | Balance diario de publicaciones positivas y negativas | puntos porcentuales (−100 a 100) | X | `pct_positivo − pct_negativo` |
| 23 | `check_pct_sentimientos` | Control | Suma de los tres porcentajes de categoría (100, salvo redondeo, en días con publicaciones; 0 en los demás) | % | Derivada | Suma de las columnas `pct_*` de X |
| 24 | `oni` | ENSO | Índice Oceánico de El Niño | °C de anomalía | NOAA | Valor mensual asignado a cada día del mes correspondiente |
| 25 | `enso_index` | ENSO | Código numérico de la categoría ENSO (columna 26) | entero 2–8 | Derivada del ONI | 2 = La Niña moderada; 3 = La Niña débil; 4 = sin evento; 5 = El Niño débil; 6 = El Niño moderado; 7 = El Niño fuerte; 8 = El Niño muy fuerte |
| 26 | `enso_categoria` | ENSO | Estado e intensidad del fenómeno ENSO | categoría | Derivada del ONI | Clasificación del ONI por intensidad |
| 27 | `nivel_embalses_pct` | Embalses | Volumen útil diario de los embalses del Sistema Interconectado Nacional | % | XM | Lectura de los archivos anuales |
| 28 | `ipp_var` | Macroeconómico | Nivel del Índice de Precios del Productor; a pesar del nombre, no es una variación | puntos de índice | DANE | Valor mensual repetido en todos los días del mes, sin interpolar |
| 29 | `ipc` | Macroeconómico | Nivel del Índice de Precios al Consumidor | puntos de índice | DANE | Valor mensual repetido en todos los días del mes, sin interpolar |
| 30 | `dia_semana` | Calendario | Día de la semana (lunes = 0, domingo = 6) | entero 0–6 | Construcción propia | A partir de la fecha |
| 31 | `es_fin_semana` | Calendario | Indica sábado o domingo | 0/1 | Construcción propia | 1 si `dia_semana` ≥ 5 |
| 32 | `es_festivo` | Calendario | Indica festivo nacional en Colombia | 0/1 | Construcción propia | Fechas fijas, Ley Emiliani y fechas relativas a la Pascua |
| 33 | `dias_no_laborales_consecutivos` | Calendario | Longitud del bloque de fin de semana y festivos al que pertenece el día | días | Construcción propia | Cero en días laborales |

## Notas

- **Publicaciones de X:** se contabilizaron las publicaciones de la cuenta @petrogustavo y los mensajes que citó o respondió que resultaron seleccionados por su relación con el sector eléctrico. Las categorías indican la dirección del precio con la que se asocia el contenido, no la emoción del mensaje. En los días sin publicaciones seleccionadas, todas las variables de X valen cero, de modo que un balance de cero puede indicar equilibrio entre categorías o ausencia de publicaciones; `n_total_tweets` permite distinguir ambos casos.
- **Nombres de columnas:** se conservan los nombres originales, como `sentiment_index` o `ipp_var`, para no alterar el archivo ni su huella SHA-256.
- **Columnas de control:** `check_pct_sum` y `check_pct_sentimientos` sirven para verificar la coherencia de los datos y no son predictores.
- **Uso como predictores:** cada columna guarda el valor del propio día. En los experimentos, el precio, la generación, los embalses, el ONI y las variables de X entraron rezagados; el IPC y el IPP, con el valor del mes anterior; y el calendario, con el valor del propio día, porque se conoce de antemano.
