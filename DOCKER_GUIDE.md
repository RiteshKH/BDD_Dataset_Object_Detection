# Docker Containerization Guide

This document provides a comprehensive guide to the Docker containerization strategy for the BDD100K Object Detection project.

## 🏗️ Architecture Overview

The project is fully containerized across three phases:

```
┌─────────────────────────────────────────────────────────────────┐
│                    BDD100K ML Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Phase 1: Data Analysis          Phase 2: Model Training       │
│  ┌──────────────────┐            ┌──────────────────┐          │
│  │  bdd-analysis    │            │  bdd-training    │          │
│  │  Container       │            │  Container       │          │
│  │  (Python 3.10)   │            │  (Python 3.10)   │          │
│  │                  │            │  (GPU Support)   │          │
│  │  - parser.py     │            │  - dataset.py    │          │
│  │  - analyze.py    │            │  - train.py      │          │
│  │  - dashboard.py  │            │  - model.py      │          │
│  └──────────────────┘            └──────────────────┘          │
│         │                               │                      │
│         ▼                               ▼                      │
│  results/analysis/              results/checkpoints/          │
│                                                                 │
│  Phase 3: Evaluation                                           │
│  ┌──────────────────┐                                          │
│  │  bdd-evaluation  │                                          │
│  │  Container       │                                          │
│  │  (Python 3.10)   │                                          │
│  │                  │                                          │
│  │  - evaluate.py   │                                          │
│  │  - visualize.py  │                                          │
│  └──────────────────┘                                          │
│         │                                                      │
│         ▼                                                      │
│  results/metrics/                                              │
│  results/visualizations/                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Container Specifications

### 1. Data Analysis Container

**Image**: `bdd-analysis`  
**Base**: `python:3.10-slim`  
**Purpose**: Parse BDD100K annotations, perform EDA, generate interactive dashboard

**Dependencies**:
- numpy >= 1.26.0
- pillow >= 10.2.0
- matplotlib >= 3.8.0
- plotly >= 5.18.0
- streamlit >= 1.29.0

**Entry Point**: `python analyze.py`

**Volume Mounts**:
- `/data/images`: BDD100K images (read-only)
- `/data/labels`: BDD100K labels (read-only)
- `/app/results`: Analysis outputs (read-write)

**Ports**: 8501 (for Streamlit dashboard)

### 2. Model Training Container

**Image**: `bdd-training`  
**Base**: `python:3.10-slim`  
**Purpose**: Train object detection model with PyTorch

**Dependencies**:
- torch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.26.0
- opencv-python >= 4.8.0
- pycocotools >= 2.0.7

**Entry Point**: `python train.py`

**Volume Mounts**:
- `/data/images`: BDD100K images (read-only)
- `/data/labels`: BDD100K labels (read-only)
- `/checkpoints`: Model checkpoints (read-write)
- `/logs`: Training logs (read-write)

**GPU Support**: Optional with `--gpus all`

### 3. Evaluation Container

**Image**: `bdd-evaluation`  
**Base**: `python:3.10-slim`  
**Purpose**: Evaluate model performance and generate visualizations

**Dependencies**:
- torch >= 2.0.0
- torchvision >= 0.15.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- pandas >= 2.1.0
- pycocotools >= 2.0.7

**Entry Point**: `python evaluate.py`

**Volume Mounts**:
- `/data/images`: BDD100K images (read-only)
- `/data/labels`: BDD100K labels (read-only)
- `/checkpoints`: Model checkpoints (read-only)
- `/metrics`: Evaluation metrics (read-write)
- `/visualizations`: Prediction visualizations (read-write)

## 🚀 Usage Patterns

### Pattern 1: Full Pipeline with Docker Compose

**Use case**: Run complete ML pipeline from start to finish

```bash
# Run all phases sequentially
docker-compose up

# Or run phases individually
docker-compose up analysis
docker-compose up training
docker-compose up evaluation
```

**Advantages**:
- Single command execution
- Automatic dependency management
- Consistent volume paths
- Easy to reproduce

### Pattern 2: Individual Containers

**Use case**: Fine-grained control, custom arguments

```bash
# Phase 1: Analysis
docker run -v $(pwd)/data:/data:ro \
           -v $(pwd)/results/analysis:/app/results \
           bdd-analysis python analyze.py --data-dir /data --output-dir /app/results

# Phase 2: Training
docker run --gpus all \
           -v $(pwd)/data:/data:ro \
           -v $(pwd)/results/checkpoints:/checkpoints \
           bdd-training --epochs 10 --batch-size 4

# Phase 3: Evaluation
docker run -v $(pwd)/data:/data:ro \
           -v $(pwd)/results/checkpoints:/checkpoints:ro \
           -v $(pwd)/results/metrics:/metrics \
           bdd-evaluation --checkpoint /checkpoints/best_model.pth
```

**Advantages**:
- Flexible command-line arguments
- Independent execution
- Easy debugging
- Resource control per container


## 🎯 GPU Configuration

### Enable GPU in Docker Compose

Edit `docker-compose.yml`:

```yaml
training:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### Enable GPU in Docker Run

```bash
docker run --gpus all \
           --runtime=nvidia \
           -e NVIDIA_VISIBLE_DEVICES=0 \
           bdd-training
```

### Verify GPU Access

```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check GPU in training container
docker run --gpus all bdd-training python -c "import torch; print(torch.cuda.is_available())"
```

## 🛠️ Building Containers

### Build All Containers

```bash
docker-compose build
```

### Build Individual Containers

```bash
# Phase 1
cd data_analysis
docker build -t bdd-analysis .

# Phase 2
cd model
docker build -t bdd-training .

# Phase 3
cd evaluation
docker build -t bdd-evaluation .
```

## 📊 Monitoring and Debugging

### View Container Logs

```bash
# Real-time logs
docker-compose logs -f training

# Last 100 lines
docker logs --tail 100 <container-id>

# Logs from specific service
docker-compose logs analysis
```

### Inspect Running Containers

```bash
# List running containers
docker ps

# Container resource usage
docker stats

# Inspect container details
docker inspect <container-id>
```

### Execute Commands in Running Container

```bash
# Open shell
docker exec -it <container-id> /bin/bash

# Run specific command
docker exec <container-id> ls /app/results

# Check Python environment
docker exec <container-id> pip list
```

## 🧪 Testing Containers

### Quick Smoke Tests

```bash
# Test analysis container
docker run bdd-analysis python -c "import numpy; print('OK')"

# Test training container
docker run bdd-training python -c "import torch; print(torch.__version__)"

# Test evaluation container
docker run bdd-evaluation python -c "import matplotlib; print('OK')"
```

### Full Integration Test

```bash
# Create test data directory
mkdir -p test_data/images/100k/train test_data/labels/100k/train

# Run analysis on test data
docker run -v $(pwd)/test_data:/data:ro \
           -v $(pwd)/test_results:/app/results \
           bdd-analysis
```

## 📝 Container Configuration Files

### docker-compose.yml

Central orchestration file defining all services, volumes, networks, and dependencies.

### Dockerfiles

- `data_analysis/Dockerfile`: Analysis container
- `model/Dockerfile`: Training container
- `evaluation/Dockerfile`: Evaluation container

### .dockerignore

Add to each module:
```
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.git/
.venv/
*.egg-info/
.pytest_cache/
```

## 🌐 Environment Variables

### Analysis Container

```bash
DATA_PATH=/data
OUTPUT_PATH=/app/results
PYTHONUNBUFFERED=1
```

### Training Container

```bash
CUDA_VISIBLE_DEVICES=0
TORCH_HOME=/checkpoints
PYTHONUNBUFFERED=1
```

### Evaluation Container

```bash
OUTPUT_DIR=/metrics
VIZ_DIR=/visualizations
PYTHONUNBUFFERED=1
```

## 📚 References

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)

---
