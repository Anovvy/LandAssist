FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd -r landassist && useradd -r -g landassist landassist

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY frontend/dist ./frontend/dist
COPY outputs/tangerang_v1.0.0.geojson ./outputs/tangerang_v1.0.0.geojson

RUN chown -R landassist:landassist /app

USER landassist

EXPOSE 8000 8080

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
