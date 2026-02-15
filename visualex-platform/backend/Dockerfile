# ============================================================================
# VisuaLex Platform - Backend
# Express + TypeScript + Prisma
# ============================================================================

# Build stage
FROM node:20-bullseye-slim AS builder

ARG MERLT_ENABLED=false

WORKDIR /app

# Install OpenSSL and ca-certificates for Prisma
RUN apt-get update && apt-get install -y openssl ca-certificates wget && rm -rf /var/lib/apt/lists/*

# Copy and build MERL-T backend package (if enabled and exists)
COPY visualex-merlt/backend/package*.json ./visualex-merlt/backend/
COPY visualex-merlt/backend/tsconfig.json ./visualex-merlt/backend/
COPY visualex-merlt/backend/tsup.config.ts ./visualex-merlt/backend/
COPY visualex-merlt/backend/src ./visualex-merlt/backend/src

RUN if [ -f "./visualex-merlt/backend/package.json" ]; then \
      cd visualex-merlt/backend && npm install && npm run build; \
    else \
      echo "MERL-T backend not found, skipping..."; \
    fi

WORKDIR /app/visualex-platform/backend

# Copy package files
COPY visualex-platform/backend/package*.json ./

# Install dependencies
RUN npm install

# Copy Prisma schema
COPY visualex-platform/backend/prisma ./prisma/

# Generate Prisma client
RUN npx prisma generate

# Copy source code
COPY visualex-platform/backend/src ./src/
COPY visualex-platform/backend/tsconfig.json ./

# Build TypeScript
RUN npm run build

# ============================================================================
# Production stage
# ============================================================================
FROM node:20-bullseye-slim

WORKDIR /app

# Install OpenSSL and ca-certificates for Prisma
RUN apt-get update && apt-get install -y openssl ca-certificates wget && rm -rf /var/lib/apt/lists/*

# Copy and build MERL-T in production stage (if exists)
COPY visualex-merlt/backend/package*.json ./visualex-merlt/backend/
COPY visualex-merlt/backend/tsconfig.json ./visualex-merlt/backend/
COPY visualex-merlt/backend/tsup.config.ts ./visualex-merlt/backend/
COPY visualex-merlt/backend/src ./visualex-merlt/backend/src

RUN if [ -f "./visualex-merlt/backend/package.json" ]; then \
      cd visualex-merlt/backend && npm install && npm run build; \
    else \
      echo "MERL-T backend not found, skipping..."; \
    fi

WORKDIR /app/visualex-platform/backend

# Copy package files and install production dependencies
COPY visualex-platform/backend/package*.json ./
RUN npm install --only=production

# Copy Prisma client from builder (already generated)
COPY --from=builder /app/visualex-platform/backend/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/visualex-platform/backend/node_modules/@prisma ./node_modules/@prisma
COPY visualex-platform/backend/prisma ./prisma/

# Copy built application
COPY --from=builder /app/visualex-platform/backend/dist ./dist

# Create non-root user
RUN groupadd -r -g 1001 nodejs && \
    useradd -r -u 1001 -g nodejs -m -d /home/appuser appuser && \
    chown -R appuser:nodejs /app /home/appuser
USER appuser

# Ensure npm cache is writable for npx/prisma
ENV HOME=/home/appuser
ENV npm_config_cache=/home/appuser/.npm

# Expose port
EXPOSE 3001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:3001/api/health || exit 1

# Run migrations and start
CMD ["sh", "-c", "npx prisma migrate deploy --schema ./prisma/schema.prisma && node dist/index.js"]
