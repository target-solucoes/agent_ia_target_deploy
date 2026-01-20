# =============================================================================
# Analytics Chatbot - Dockerfile Multi-stage
# Target: ghcr.io/target-solucoes/analytics-chatbot
# =============================================================================

# Stage 1: Builder - compila dependencias
FROM python:3.12-slim AS builder

WORKDIR /build

# Instala dependencias de build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de dependencias
COPY pyproject.toml ./

# Instala dependencias em diretorio isolado
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/install .

# Stage 2: Runtime - imagem final otimizada
FROM python:3.12-slim AS runtime

# Dependencias do sistema:
# - curl: para healthcheck
# - libs*: para kaleido (exportacao PNG de graficos Plotly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libglib2.0-0 libnss3 libnspr4 libdbus-1-3 libatk1.0-0 \
    libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libpango-1.0-0 \
    libcairo2 libasound2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copia dependencias instaladas do builder
COPY --from=builder /install /usr/local/lib/python3.12/site-packages

# Copia codigo fonte da aplicacao
COPY src/ ./src/
COPY streamlit_app/ ./streamlit_app/
COPY data/mappings/ ./data/mappings/
COPY app.py ./
COPY pyproject.toml ./

# Copia dataset embutido na imagem (autossuficiente)
COPY data/datasets/ ./data/datasets/

# Cria diretorios necessarios com permissoes
RUN mkdir -p logs data/output/graphics

# Cria usuario nao-root para seguranca
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Configuracao Streamlit para container
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Configuracao Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

# Dataset path (embutido na imagem)
ENV DATASET_PATH=data/datasets/DadosComercial_resumido_v02.parquet

EXPOSE 8501

# Healthcheck usando endpoint nativo do Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de inicializacao
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
