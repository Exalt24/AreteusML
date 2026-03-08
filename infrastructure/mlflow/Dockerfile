FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==2.21.0 psycopg2-binary==2.9.10

RUN mkdir -p /mlflow/artifacts

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "${MLFLOW_BACKEND_STORE_URI}", \
     "--default-artifact-root", "${MLFLOW_DEFAULT_ARTIFACT_ROOT}"]
