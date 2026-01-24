# VisuaLex Platform - Frontend
# React + Vite + TypeScript + Tailwind

# Build stage
FROM node:20-alpine AS builder

ARG VITE_MERLT_ENABLED=false

WORKDIR /app

# Copy MERL-T frontend package and build it
COPY visualex-merlt/frontend/package*.json ./visualex-merlt/frontend/
COPY visualex-merlt/frontend/tsconfig*.json ./visualex-merlt/frontend/
COPY visualex-merlt/frontend/tsup.config.ts ./visualex-merlt/frontend/
COPY visualex-merlt/frontend/src ./visualex-merlt/frontend/src

WORKDIR /app/visualex-merlt/frontend
RUN if [ "$VITE_MERLT_ENABLED" = "true" ]; then npm install && npm run build:lib; else echo "Skipping MERL-T frontend build"; fi

WORKDIR /app/visualex-platform/frontend

# Copy package files
COPY visualex-platform/frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY visualex-platform/frontend . .

# Build arguments for API URLs
ARG VITE_API_URL=http://localhost:3001
ARG VITE_PYTHON_API_URL=http://localhost:5000

ENV VITE_API_URL=$VITE_API_URL
ENV VITE_PYTHON_API_URL=$VITE_PYTHON_API_URL
ENV VITE_MERLT_ENABLED=$VITE_MERLT_ENABLED

# Build the application (skip type checking in Docker)
RUN npm run build:docker

# Production stage
FROM nginx:alpine

# Add curl for healthcheck
RUN apk add --no-cache curl

# Copy custom nginx config
COPY <<EOF /etc/nginx/conf.d/default.conf
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # Gzip compression
    gzip on;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

    # Backend API proxy (same-origin for browser)
    location /api/ {
        proxy_pass http://backend:3001;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # Python API proxy (visualex scrapers)
    location ~ ^/(fetch_|stream_|export_pdf|history|version) {
        proxy_pass http://python-api:5000;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    # SPA routing - serve index.html for all routes
    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 'healthy';
        add_header Content-Type text/plain;
    }
}
EOF

# Copy built assets from builder
COPY --from=builder /app/visualex-platform/frontend/dist /usr/share/nginx/html

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost/health || exit 1

CMD ["nginx", "-g", "daemon off;"]
