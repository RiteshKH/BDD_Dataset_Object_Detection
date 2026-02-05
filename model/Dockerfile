# BDD100K Model Training Container
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training scripts
COPY dataset.py .
COPY model_architecture.py .
COPY train.py .

# Create necessary directories
RUN mkdir -p /checkpoints /data /logs

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command runs training
ENTRYPOINT ["python", "train.py"]
CMD ["--help"]
