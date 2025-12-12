
# Institutional v25 Reproducible Environment
# Base Image: Python 3.10 Slim
FROM python:3.10-slim

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROJECT_DIR=/app

# System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create Project Directory
WORKDIR ${PROJECT_DIR}

# Install Python Dependencies
# Copy requirements first for cache optimization
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Additional Institutional Dependencies
RUN pip install --no-cache-dir \
    mlflow \
    shap \
    arch \
    statsmodels \
    pytest \
    pytest-cov \
    black \
    flake8

# Copy Source Code
COPY . .

# Create Directories for Reports and Logs
RUN mkdir -p reports/science reports/ml logs data/models

# Default Command: Run Audit Check (to verify environment)
CMD ["python", "audit_check.py"]
