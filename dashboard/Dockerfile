FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    streamlit==1.41.* \
    streamlit-shadcn-ui==0.1.* \
    httpx==0.28.* \
    plotly==6.0.* \
    pandas==2.2.* \
    loguru==0.7.*

COPY dashboard/ ./dashboard/

RUN mkdir -p /app/dashboard/logs

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
