# BDD100K Data Analysis Container
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy analysis scripts
COPY parser.py .
COPY analyze.py .
COPY dashboard.py .

# Create output directory
RUN mkdir -p /app/results

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command runs the analysis
CMD ["python", "analyze.py"]
