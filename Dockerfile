# Entorno reproducible del modelo seleccionado (Gradient Boosting Huber con corrección por persistencia).
FROM python:3.12.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=42 \
    PYTHONIOENCODING=utf-8 \
    OMP_NUM_THREADS=1 \
    PYTHONPATH=/app/src \
    PRECIO_BOLSA_ROOT=/app

WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY configs ./configs
COPY data ./data
COPY models ./models
COPY reference ./reference
COPY src ./src
COPY ejecutar_modelo.py .

ENTRYPOINT ["python", "-m", "precio_bolsa_x"]
CMD ["all"]
