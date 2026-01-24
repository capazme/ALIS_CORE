# VisuaLex API - Italian Legal Text Scraper & API
# Open Source - MIT License

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_BASE_PATH=/app \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for caching
COPY pyproject.toml README.md ./

# Install Python dependencies (including browser support)
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e ".[browser]" && \
    playwright install chromium && \
    playwright install-deps chromium && \
    mkdir -p /ms-playwright && \
    chmod -R 755 /ms-playwright

# Copy application code
COPY visualex/ ./visualex/

# Create data directories for cache and history
RUN mkdir -p /app/data /app/download/cache && \
    chmod -R 777 /app/data /app/download

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app /ms-playwright
USER appuser

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "-m", "visualex.app"]
