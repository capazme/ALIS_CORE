# MERL-T Models - Proprietary Trained Model Weights
# This container serves model weights and configurations
# PROPRIETARY - Internal use only

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/models

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml README.md ./

# Install base dependencies (lightweight)
RUN pip install --upgrade pip setuptools wheel && \
    pip install pyyaml fastapi uvicorn

# Copy package code
COPY merlt_models/ ./merlt_models/
COPY weights/ ./weights/
COPY checkpoints/ ./checkpoints/
COPY config/ ./config/

# Create simple model server
RUN cat > /app/model_server.py << 'EOF'
"""
Simple Model Server
Serves model weights and configurations via HTTP.
"""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

app = FastAPI(
    title="MERL-T Model Server",
    description="Serves trained model weights and configurations",
    version="0.1.0",
)

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app"))

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/models")
async def list_models():
    """List available models and checkpoints."""
    models = []

    # Check weights directory
    weights_dir = MODELS_DIR / "weights"
    if weights_dir.exists():
        for item in weights_dir.iterdir():
            if item.is_dir():
                models.append({"type": "weights", "name": item.name})

    # Check checkpoints directory
    checkpoints_dir = MODELS_DIR / "checkpoints"
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir():
                models.append({"type": "checkpoint", "name": item.name})

    return {"models": models}

@app.get("/config/{config_name}")
async def get_config(config_name: str):
    """Get model configuration by name."""
    config_file = MODELS_DIR / "config" / f"{config_name}.yaml"
    if not config_file.exists():
        config_file = MODELS_DIR / "config" / f"{config_name}.yml"

    if not config_file.exists():
        raise HTTPException(status_code=404, detail=f"Config {config_name} not found")

    return FileResponse(config_file, media_type="application/x-yaml")

@app.get("/weights/{model_name}/{filename}")
async def get_weight_file(model_name: str, filename: str):
    """Download model weight file."""
    file_path = MODELS_DIR / "weights" / model_name / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Weight file not found")

    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
EOF

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose model server port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run model server
CMD ["uvicorn", "model_server:app", "--host", "0.0.0.0", "--port", "8001"]
