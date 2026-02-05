# BDD100K Evaluation and Visualization Container
FROM python:3.10-slim

# Install system dependencies for OpenCV and plotting
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

# Copy evaluation scripts
COPY evaluate.py .
COPY visualize.py .

# Create output directories
RUN mkdir -p /metrics /visualizations /data /checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command runs evaluation
ENTRYPOINT ["python", "evaluate.py"]
CMD ["--help"]
