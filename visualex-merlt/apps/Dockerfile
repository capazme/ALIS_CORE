# VisuaLex-MERLT - Streamlit Dashboard Apps
# Python + Streamlit for ML debugging/monitoring

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8503 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements (if exists) or install common dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install streamlit pandas plotly httpx pyyaml

# Copy application code
COPY . ./apps/

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose Streamlit port
EXPOSE 8503

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8503/_stcore/health || exit 1

# Default: run RLCF dashboard
CMD ["streamlit", "run", "apps/rlcf_dashboard.py", "--server.port=8503", "--server.address=0.0.0.0"]
