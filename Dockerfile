# ============================================================
# DeepGuard AI — Production Dockerfile
# Optimized for Render, Railway, Hugging Face Spaces, and VPS
# ============================================================
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Install minimal OS dependencies for computer vision and video processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy codebase
COPY . .

# Create persistent storage directories
RUN mkdir -p data checkpoints

# Expose container port
EXPOSE 8000

# Start server (binds to cloud provider's $PORT or default 8000)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

