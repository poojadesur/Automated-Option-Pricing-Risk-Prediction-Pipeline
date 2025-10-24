# Dockerfile for Option Pricing Pipeline
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p data/raw data/processed models/saved logs

# Expose MLflow port
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=./mlruns

# Default command
CMD ["python", "-m", "mlflow", "ui", "--host", "0.0.0.0"]
