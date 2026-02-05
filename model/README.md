# Phase 2: Model Training

This module implements a complete PyTorch training pipeline for object detection on BDD100K dataset.

## 📋 Module Contents

- `train.py`: Training loop with checkpointing
- `dataset.py`: PyTorch Dataset and DataLoader implementation
- `model_architecture.py`: Model definition and loading
- `Dockerfile`: Containerized training environment
- `requirements.txt`: Python dependencies
- `MODEL_SELECTION.md`: Architecture rationale

## 🚀 Quick Start

### Option 1: Docker 

```bash
# Build container
docker build -t bdd-training .

# Train with GPU
docker run --gpus all \
  -v /path/to/bdd100k_images_100k:/data/images:ro \
  -v /path/to/bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/checkpoints:/checkpoints \
  bdd-training \
  --data-dir /data \
  --output-dir /checkpoints \
  --epochs 10 \
  --batch-size 4

# Train on CPU (no --gpus flag)
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

### Option 2: Local Python

```bash
# Install dependencies
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
  --lr 0.005 \
  --early-stopping-patience 10 \
  --output-dir ../results/checkpoints
```

## 🔧 Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | required | Base directory containing images/ and labels/ |
| `--train-img-dir` | str | None | Training images directory |
| `--train-ann-dir` | str | None | Training annotations directory |
| `--val-img-dir` | str | None | Validation images directory |
| `--val-ann-dir` | str | None | Validation annotations directory |
| `--output-dir` | str | `./checkpoints` | Output directory for checkpoints |
| `--log-dir` | str | `./logs` | Logging directory |
| `--epochs` | int | 10 | Number of training epochs |
| `--batch-size` | int | 4 | Batch size for training |
| `--lr` | float | 0.005 | Learning rate |
| `--momentum` | float | 0.9 | SGD momentum |
| `--weight-decay` | float | 0.0005 | Weight decay |
| `--subset-size` | int | None | Train on subset (for testing) |
| `--num-workers` | int | 4 | DataLoader workers |
| `--print-freq` | int | 10 | Print frequency |

## 📦 Model Architecture

**Default**: Faster R-CNN with ResNet-50 FPN backbone

- Pre-trained on COCO dataset
- 10 output classes (BDD100K detection classes)
- Multi-scale Feature Pyramid Network (FPN)
- Region Proposal Network (RPN) + ROI head

See [MODEL_SELECTION.md](MODEL_SELECTION.md) for detailed rationale.

## 📊 Training Outputs

### Checkpoints

Saved to `--output-dir`:
- `checkpoint_epoch_N.pth`: Checkpoint after each epoch
- `best_model.pth`: Best model based on validation loss
- `last_model.pth`: Latest checkpoint

### Logs

Training progress logged to console and `--log-dir`:
- Epoch number
- Loss values (classifier, box_reg, objectness, rpn_box_reg)
- Learning rate
- Validation metrics (if validation set provided)

## 🧪 Testing the Module

```bash
# Test dataset loading
python dataset.py

# Test model loading
python model_architecture.py

# Quick training test (1 epoch, 50 images)
python train.py \
  --train-img-dir ../bdd100k_images_100k/100k/train \
  --train-ann-dir ../bdd100k_labels/100k/train \
  --subset-size 50 \
  --epochs 1 \
  --batch-size 2 \
  --output-dir ./test_output
```

## 🔗 Related Documentation

- [../README.md](../README.md): Main project documentation
- [MODEL_SELECTION.md](MODEL_SELECTION.md): Architecture selection rationale
- [../evaluation/README.md](../evaluation/README.md): Evaluation module
