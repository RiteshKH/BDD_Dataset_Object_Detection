# BDD100K Object Detection Project

A complete machine learning pipeline for object detection on the BDD100K (Berkeley DeepDrive) dataset, featuring containerized data analysis, model training, and comprehensive evaluation.

## 📋 Project Overview

This project implements a full production-ready object detection system for autonomous driving scenarios:

- **Phase 1**: Dockerized data analysis with interactive dashboard
- **Phase 2**: PyTorch-based model training with Faster R-CNN
- **Phase 3**: Quantitative and qualitative evaluation with failure analysis

**Dataset**: BDD100K - 100K driving images with 10 object detection classes:
- Vehicles: car, truck, bus, train
- Two-wheelers: bike, motor
- Pedestrians: person, rider
- Traffic infrastructure: traffic light, traffic sign

## 🎯 Key Features

- ✅ **PEP8 compliant** code with comprehensive docstrings
- ✅ **Full Docker containerization** for all phases (analysis, training, evaluation)
- ✅ **Docker Compose orchestration** for complete pipeline automation
- ✅ **Modular architecture** with clear separation of concerns
- ✅ **Interactive dashboard** for dataset visualization (Streamlit)
- ✅ **Production-ready training** pipeline with checkpointing
- ✅ **COCO-style evaluation** metrics (mAP@0.5, mAP@0.75, mAP avg)
- ✅ **Failure pattern analysis** by size, occlusion, scene conditions
- ✅ **Comprehensive documentation** of design decisions

## 📁 Repository Structure

```
BBD_Detection_Task/
├── README.md                          # This file
├── docker-compose.yml                 # Multi-container orchestration
├── .github/
│   └── copilot-instructions.md        # AI agent guidance
├── data_analysis/                     # Phase 1: Data Analysis (DOCKERIZED)
│   ├── Dockerfile                     # Self-contained analysis environment
│   ├── requirements.txt               # Python dependencies
│   ├── parser.py                      # BDD100K JSON parser
│   ├── analyze.py                     # Analysis pipeline
│   ├── dashboard.py                   # Interactive Streamlit dashboard
│   ├── ANALYSIS.md                    # Detailed analysis findings
│   └── README.md                      # Module documentation
├── model/                             # Phase 2: Model Training (DOCKERIZED)
│   ├── Dockerfile                     # Training container
│   ├── requirements.txt               # PyTorch dependencies
│   ├── dataset.py                     # PyTorch Dataset/DataLoader
│   ├── model_architecture.py          # Model loading and configuration
│   ├── train.py                       # Training pipeline
│   └── MODEL_SELECTION.md             # Architecture rationale
├── evaluation/                        # Phase 3: Evaluation (DOCKERIZED)
│   ├── Dockerfile                     # Evaluation container
│   ├── requirements.txt               # Evaluation dependencies
│   ├── evaluate.py                    # Quantitative metrics (mAP)
│   ├── visualize.py                   # Qualitative visualization
│   └── EVALUATION.md                  # Performance analysis
├── notebooks/                         # Exploratory notebooks
│   └── BBD_Object_Detection.ipynb     # Development notebook
├── results/                           # Output directory (Docker volumes)
│   ├── analysis/                      # Phase 1 outputs
│   ├── checkpoints/                   # Phase 2 model checkpoints
│   ├── logs/                          # Training logs
│   ├── metrics/                       # Phase 3 evaluation metrics
│   └── visualizations/                # Phase 3 prediction visualizations
├── bdd100k_images_100k/               # Dataset images
│   └── 100k/
│       ├── train/                     # Training images
│       ├── val/                       # Validation images
│       └── test/                      # Test images
└── bdd100k_labels/                    # Annotation files (per-image JSON)
    └── 100k/
        ├── train/                     # Training labels (1 JSON per image)
        ├── val/                       # Validation labels
        └── test/                      # Test labels
```

## 🚀 Quick Start

> **TL;DR**: See [QUICKSTART.md](QUICKSTART.md) for a quick reference card with all essential commands.

### Option 1: Docker Compose (Recommended - Full Pipeline)

Run all phases with a single command:

```bash
# Run complete pipeline (analysis → training → evaluation)
docker-compose up

# Run specific phase only
docker-compose up analysis         # Phase 1: Data analysis
docker-compose up training         # Phase 2: Model training
docker-compose up evaluation       # Phase 3: Evaluation

# Run dashboard (interactive visualization)
docker-compose up dashboard
# Open browser: http://localhost:8501
```

### Option 2: Individual Containers

Build and run each phase independently:

```bash
# Phase 1: Data Analysis
cd data_analysis
docker build -t bdd-analysis .
docker run -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
           -v $(pwd)/../bdd100k_labels:/data/labels:ro \
           -v $(pwd)/../results/analysis:/app/results \
           bdd-analysis

# Phase 2: Model Training  
cd model
docker build -t bdd-training .
docker run --gpus all \
           -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
           -v $(pwd)/../bdd100k_labels:/data/labels:ro \
           -v $(pwd)/../results/checkpoints:/checkpoints \
           bdd-training --epochs 10 --batch-size 4

# Phase 3: Evaluation
cd evaluation
docker build -t bdd-evaluation .
docker run -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
           -v $(pwd)/../bdd100k_labels:/data/labels:ro \
           -v $(pwd)/../results/checkpoints:/checkpoints:ro \
           -v $(pwd)/../results/metrics:/metrics \
           bdd-evaluation --checkpoint /checkpoints/best_model.pth
```

### Option 3: Local Python (Development)

For active development without Docker:

```bash
# Install dependencies per phase
pip install -r data_analysis/requirements.txt
pip install -r model/requirements.txt
pip install -r evaluation/requirements.txt

# Run scripts directly (see individual phase sections below)
```

---

## 🛠️ Setup Instructions

### Prerequisites

- **Docker** (required for all phases)
- **Docker Compose** (optional, for orchestration)
- **NVIDIA Docker** (optional, for GPU training)
- Python 3.10+ (if running locally without Docker)
- 50GB+ disk space
- 16GB+ RAM (32GB recommended for training)

### 1. Install Docker

**Ubuntu/Linux:**
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
# Add your user to docker group (avoids sudo)
sudo usermod -aG docker $USER
# Log out and back in for group changes
```

**macOS/Windows:**
- Download [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Install and start Docker Desktop

Verify installation:
```bash
docker --version
docker-compose --version
```

### 2. Install NVIDIA Docker (Optional - For GPU Training)

**Ubuntu/Linux only:**
```bash
# Add NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 3. Dataset Setup

Ensure your dataset follows this structure:
```
BDD_Detection_Task/
├── bdd100k_images_100k/100k/
│   ├── train/  (*.jpg files)
│   ├── val/    (*.jpg files)
│   └── test/   (*.jpg files)
└── bdd100k_labels/100k/
    ├── train/  (*.json files, one per image)
    ├── val/    (*.json files)
    └── test/   (*.json files)
```

### 4. Clone Repository

```bash
git clone <repository-url>
cd BDD_Detection_Task
```

---

## 📊 Phase 1: Data Analysis (Dockerized)

### Method 1: Using Docker Compose

```bash
# Run analysis
docker-compose up analysis

# View results
ls results/analysis/
```

### Method 2: Individual Container

**Build the container:**
```bash
cd data_analysis
docker build -t bdd-analysis .
```

**Run data analysis:**
```bash
docker run \
  -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
  -v $(pwd)/../bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/analysis:/app/results \
  bdd-analysis python analyze.py --data-dir /data --output-dir /app/results
```

**Windows PowerShell:**
```powershell
docker run `
  -v ${PWD}\..\bdd100k_images_100k:/data/images:ro `
  -v ${PWD}\..\bdd100k_labels:/data/labels:ro `
  -v ${PWD}\..\results\analysis:/app/results `
  bdd-analysis python analyze.py --data-dir /data --output-dir /app/results
```

### Method 3: Local Python (Development Only)

```bash
cd data_analysis
pip install -r requirements.txt
python analyze.py \
  --data-dir ../bdd100k_labels \
  --output-dir ../results/analysis
```

### Interactive Dashboard

**Using Docker Compose:**
```bash
docker-compose up dashboard
# Open browser: http://localhost:8501
```

**Using Docker:**
```bash
cd data_analysis
docker run -p 8501:8501 \
  -v $(pwd)/../results/analysis:/app/results:ro \
  bdd-analysis \
  streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
```

**Locally:**
```bash
cd data_analysis
streamlit run dashboard.py
```

### Analysis Outputs

The container generates the following files in `results/analysis/`:

- `class_distribution.json`: Object counts per class
- `split_comparison.json`: Train/val distribution comparison
- `image_statistics.json`: Image-level statistics
- `bbox_statistics.json`: Bounding box size/shape stats
- `occlusion_statistics.json`: Occlusion and truncation rates
- `scene_attributes.json`: Weather, time of day distributions
- `anomalies.json`: Detected dataset anomalies

**See [data_analysis/ANALYSIS.md](data_analysis/ANALYSIS.md) for detailed findings.**

---

### Training Outputs

- `checkpoints/checkpoint_epoch_N.pth`: Checkpoint after each epoch
- `checkpoints/best_model.pth`: Best model based on validation loss
- `logs/training.log`: Training progress and loss values

**See [model/MODEL_SELECTION.md](model/MODEL_SELECTION.md) for architecture details.**

---

### Evaluation Outputs

**Metrics (`results/metrics/`):**
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **mAP@0.75**: Mean Average Precision at IoU 0.75
- **mAP (0.5:0.95)**: Average across IoU thresholds (COCO metric)
- **Per-class AP**: Individual class performance
- `evaluation_results.json`: Complete metrics

**Visualizations (`results/visualizations/`):**
- `viz_XXX_<image_name>.jpg`: Side-by-side ground truth vs predictions
- `failures_by_size.json`: Missed detections categorized by object size
- `confusion_matrix.png`: Class confusion visualization

**See [evaluation/EVALUATION.md](evaluation/EVALUATION.md) for performance analysis.**

## 🤖 Phase 2: Model Training (Dockerized)

### Method 1: Using Docker Compose

```bash
# Run training (default: 1 epoch, batch-size 2)
docker-compose up training

# Enable GPU support: Edit docker-compose.yml and uncomment GPU section
```

### Method 2: Individual Container with GPU

**Build the container:**
```bash
cd model
docker build -t bdd-training .
```

**Run training with GPU:**
```bash
docker run --gpus all \
  -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
  -v $(pwd)/../bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/checkpoints:/checkpoints \
  -v $(pwd)/../results/logs:/logs \
  bdd-training \
  --data-dir /data \
  --output-dir /checkpoints \
  --log-dir /logs \
  --epochs 10 \
  --batch-size 4 \
  --lr 0.005
```

**Run training CPU-only (no --gpus flag):**
```bash
docker run \
  -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
  -v $(pwd)/../bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/checkpoints:/checkpoints \
  bdd-training \
  --data-dir /data \
  --output-dir /checkpoints \
  --epochs 1 \
  --batch-size 2
```

**Windows PowerShell:**
```powershell
docker run --gpus all `
  -v ${PWD}\..\bdd100k_images_100k:/data/images:ro `
  -v ${PWD}\..\bdd100k_labels:/data/labels:ro `
  -v ${PWD}\..\results\checkpoints:/checkpoints `
  bdd-training --data-dir /data --output-dir /checkpoints --epochs 10
```

### Method 3: Local Python

```bash
cd model
pip install -r requirements.txt

# Quick test (1 epoch, 100 images)
python train.py \
  --train-img-dir ../bdd100k_images_100k/100k/train \
  --train-ann-dir ../bdd100k_labels/100k/train \
  --epochs 1 \
  --batch-size 4 \
  --early-stopping-patience 10 \
  --output-dir ../results/checkpoints

# Full training
python train.py \
  --train-img-dir ../bdd100k_images_100k/100k/train \
  --train-ann-dir ../bdd100k_labels/100k/train \
  --val-img-dir ../bdd100k_images_100k/100k/val \
  --val-ann-dir ../bdd100k_labels/100k/val \
  --epochs 10 \
  --batch-size 4 \
  --early-stopping-patience 10 \
  --output-dir ../results/checkpoints
```

---

## 📊 Complete Pipeline Execution

### Full Automated Pipeline with Docker Compose

Run the complete ML pipeline from start to finish:

```bash
# 1. Data analysis
docker-compose up analysis

# 2. Review analysis results
cat results/analysis/class_distribution.json

# 3. Train model (adjust epochs/batch-size as needed)
docker-compose up training

# 4. Evaluate model
docker-compose up evaluation

# 5. Generate visualizations
docker-compose up visualize

# 6. (Optional) Launch dashboard
docker-compose up dashboard
```

### GPU Training Configuration

To enable GPU support in Docker Compose, edit `docker-compose.yml`:

```yaml
training:
  # Uncomment these lines:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

Then run:
```bash
docker-compose up training
```

---

## 🔧 Docker Management

### Useful Commands

```bash
# Build all containers
docker-compose build

# Rebuild specific service
docker-compose build training

# View running containers
docker ps

# Stop all containers
docker-compose down

# Remove all containers and volumes
docker-compose down -v

# View logs from specific service
docker-compose logs training

# Follow logs in real-time
docker-compose logs -f evaluation

# Execute command in running container
docker-compose exec training bash

# Clean up Docker images
docker system prune -a
```

### Storage Management

Docker volumes store results persistently:

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect bdd_detection_task_results

# Remove unused volumes
docker volume prune
```

---

## 🎓 Dataset Information

### BDD100K Structure

**Important**: BDD100K labels are stored as **individual JSON files per image** (not a single consolidated file):

```
bdd100k_images_100k/100k/
├── train/              # Training images (*.jpg)
├── val/                # Validation images (*.jpg)
└── test/               # Test images (*.jpg)

bdd100k_labels/100k/
├── train/              # Training labels (one .json file per image)
│   ├── 0000f77c-6257a9fd.json
│   ├── 000a1249-33b0dcf5.json
│   └── ...
├── val/                # Validation labels
│   └── ...
└── test/               # Test labels
    └── ...
```

Each JSON file contains annotations for a single image with the same base name.

### Detection Classes (10 classes)

- **Vehicles**: car, truck, bus, train
- **Two-wheelers**: bike, motor  
- **Pedestrians**: person, rider
- **Traffic infrastructure**: traffic light, traffic sign

---

## 📈 Phase 3: Evaluation & Visualization (Dockerized)

### Method 1: Using Docker Compose

```bash
# Run evaluation (requires trained model)
docker-compose up evaluation

# Generate visualizations only
docker-compose up visualize
```

### Method 2: Individual Container

**Build the container:**
```bash
cd evaluation
docker build -t bdd-evaluation .
```

**Run quantitative evaluation:**
```bash
docker run \
  -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
  -v $(pwd)/../bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/checkpoints:/checkpoints:ro \
  -v $(pwd)/../results/metrics:/metrics \
  bdd-evaluation \
  --data-dir /data \
  --checkpoint /checkpoints/best_model.pth \
  --output-dir /metrics \
  --multi-iou
```

**Run visualization:**
```bash
docker run \
  -v $(pwd)/../bdd100k_images_100k:/data/images:ro \
  -v $(pwd)/../bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/checkpoints:/checkpoints:ro \
  -v $(pwd)/../results/visualizations:/visualizations \
  bdd-evaluation python visualize.py \
  --data-dir /data \
  --checkpoint /checkpoints/best_model.pth \
  --output-dir /visualizations \
  --num-samples 50 \
  --analyze-failures
```

**Windows PowerShell:**
```powershell
docker run `
  -v ${PWD}\..\bdd100k_images_100k:/data/images:ro `
  -v ${PWD}\..\bdd100k_labels:/data/labels:ro `
  -v ${PWD}\..\results\checkpoints:/checkpoints:ro `
  -v ${PWD}\..\results\metrics:/metrics `
  bdd-evaluation --data-dir /data --checkpoint /checkpoints/best_model.pth --output-dir /metrics
```

### Method 3: Local Python

**Install dependencies:**
```bash
cd evaluation
pip install -r requirements.txt
```

**Run evaluation:**
```bash
python evaluate.py \
  --val-img-dir ../bdd100k_images_100k/100k/val \
  --val-ann-dir ../bdd100k_labels/100k/val \
  --checkpoint ../results/checkpoints/best_model.pth \
  --output-dir ../results/metrics \
  --multi-iou
```

**Generate visualizations:**
```bash
python visualize.py \
  --val-img-dir ../bdd100k_images_100k/100k/val \
  --val-ann-dir ../bdd100k_labels/100k/val \
  --checkpoint ../results/checkpoints/best_model.pth \
  --output-dir ../results/visualizations \
  --num-images 50 \
  --score-threshold 0.5 \
  --analyze-failures \
  --confusion-matrix
```

### Known Challenges

1. **Small objects**: Traffic lights and signs (<2000 pixels²)
2. **Rare classes**: Train class (<0.5% of dataset)
3. **Occlusion**: ~25% of objects partially occluded
4. **Weather diversity**: Limited fog/snow examples

### Proposed Improvements

- Focal loss for class imbalance
- Higher resolution training (1600x900)
- Weather augmentation (synthetic rain/fog)
- Occlusion-aware training

## 🛠 Development Workflow

### Development Mode (Without Docker)

For rapid iteration during development:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install all dependencies
pip install -r data_analysis/requirements.txt
pip install -r model/requirements.txt
pip install -r evaluation/requirements.txt

# Run scripts directly
cd data_analysis
python analyze.py --data-dir ../bdd100k_labels --output-dir ../results/analysis

cd ../model
python train.py --epochs 1 --subset-size 100

cd ../evaluation
python evaluate.py --checkpoint ../results/checkpoints/best_model.pth
```

### Testing Individual Modules

```bash
# Test data parser
cd data_analysis
python parser.py

# Test dataset loader
cd model
python -c "from dataset import BDD100KDataset; print('Dataset OK')"

# Test model architecture
python model_architecture.py

# Test training (1 epoch, small subset)
python train.py --subset-size 50 --epochs 1 --batch-size 2
```

### Debugging Docker Containers

```bash
# Run container interactively
docker run -it --entrypoint /bin/bash bdd-training

# Inside container, test commands
python -c "import torch; print(torch.cuda.is_available())"
ls /data/images/100k/train/ | head
python train.py --help

# Inspect container filesystem
docker run --rm bdd-training ls -la /app

# Check container logs
docker logs <container-id>
```

## 📚 Documentation

- **[DOCKER_GUIDE.md](DOCKER_GUIDE.md)**: Comprehensive Docker containerization guide
- **[data_analysis/README.md](data_analysis/README.md)**: Data analysis module documentation
- **[data_analysis/ANALYSIS.md](data_analysis/ANALYSIS.md)**: Dataset analysis findings
- **[model/README.md](model/README.md)**: Model training module documentation
- **[model/MODEL_SELECTION.md](model/MODEL_SELECTION.md)**: Model architecture rationale
- **[evaluation/README.md](evaluation/README.md)**: Evaluation module documentation
- **[evaluation/EVALUATION.md](evaluation/EVALUATION.md)**: Performance analysis and improvements
- **[RESULTS_REPORT.md](RESULTS_REPORT.md)**: Final report

## 🙏 Acknowledgments

- **BDD100K Dataset**: Berkeley DeepDrive team
- **PyTorch/Torchvision**: Model implementations
- **Faster R-CNN**: Ren et al., "Faster R-CNN: Towards Real-Time Object Detection"
- **Github Copilot**: Basic Code framework generation
