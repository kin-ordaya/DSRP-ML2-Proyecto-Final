# Reto ML1 - Contenedor del pipeline de ML (preprocesamiento, entrenamiento y
# predicción) para el proyecto de predicción de churn en telecomunicaciones.
#
# Construye una imagen Python 3.12 con todas las dependencias del proyecto
# (ver requirements.txt) y el código fuente. La ejecución de cada etapa
# (preprocess / train / predict) queda a cargo de docker-compose.

FROM python:3.12-slim

# Evita buffers y archivos .pyc
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 1) Dependencias primero para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 2) Código fuente del proyecto
COPY pyproject.toml .
COPY src/ ./src/
COPY references/ ./references/

# Nota: los datos crudos (data/raw/telco_customer_churn.csv) NO se versionan en
# git (ver .gitignore). Se montan como volúmenes en docker-compose para que el
# pipeline los lea sin depender de Kaggle en tiempo de ejecución. Los artefactos
# generados (data/processed/ y models/) también se montan como volúmenes para
# no ensuciar la imagen.

# Punto de entrada por defecto es trivial; cada servicio de compose define su
# propio comando (python -m src.xxx).
CMD ["python", "--version"]
