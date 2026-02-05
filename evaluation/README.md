# Phase 3: Evaluation & Visualization

This module provides comprehensive evaluation and visualization tools for the trained object detection model.

## 📋 Module Contents

- `evaluate.py`: Quantitative evaluation (mAP metrics)
- `visualize.py`: Qualitative visualization and failure analysis
- `Dockerfile`: Containerized evaluation environment
- `requirements.txt`: Python dependencies
- `EVALUATION.md`: Performance analysis and insights

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build container
docker build -t bdd-evaluation .

# Run evaluation
docker run \
  -v /path/to/bdd100k_images_100k:/data/images:ro \
  -v /path/to/bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/checkpoints:/checkpoints:ro \
  -v $(pwd)/../results/metrics:/metrics \
  bdd-evaluation \
  --data-dir /data \
  --checkpoint /checkpoints/best_model.pth \
  --output-dir /metrics \
  --multi-iou

# Run visualization
docker run \
  -v /path/to/bdd100k_images_100k:/data/images:ro \
  -v /path/to/bdd100k_labels:/data/labels:ro \
  -v $(pwd)/../results/checkpoints:/checkpoints:ro \
  -v $(pwd)/../results/visualizations:/visualizations \
  bdd-evaluation python visualize.py \
  --data-dir /data \
  --checkpoint /checkpoints/best_model.pth \
  --output-dir /visualizations \
  --num-samples 50 \
  --analyze-failures
```

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Quantitative evaluation
python evaluate.py \
  --val-img-dir ../bdd100k_images_100k/100k/val \
  --val-ann-dir ../bdd100k_labels/100k/val \
  --checkpoint ../results/checkpoints/best_model.pth \
  --output-dir ../results/metrics \
  --multi-iou

# Generate visualizations
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

## 🔧 Command Line Arguments

### evaluate.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | required | Base directory with images/ and labels/ |
| `--val-img-dir` | str | None | Validation images directory |
| `--val-ann-dir` | str | None | Validation annotations directory |
| `--checkpoint` | str | required | Path to trained model checkpoint |
| `--output-dir` | str | `./metrics` | Output directory for metrics |
| `--batch-size` | int | 8 | Batch size for evaluation |
| `--num-workers` | int | 4 | DataLoader workers |
| `--multi-iou` | flag | False | Compute mAP at multiple IoU thresholds |
| `--iou-threshold` | float | 0.5 | IoU threshold for AP calculation |
| `--score-threshold` | float | 0.5 | Confidence threshold for detections |

### visualize.py

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data-dir` | str | required | Base directory with images/ and labels/ |
| `--val-img-dir` | str | None | Validation images directory |
| `--val-ann-dir` | str | None | Validation annotations directory |
| `--checkpoint` | str | required | Path to trained model checkpoint |
| `--output-dir` | str | `./visualizations` | Output directory |
| `--num-samples` | int | 50 | Number of images to visualize |
| `--score-threshold` | float | 0.5 | Confidence threshold |
| `--analyze-failures` | flag | False | Generate failure analysis |
| `--confusion-matrix` | flag | False | Generate confusion matrix |
| `--by-size` | flag | False | Group by object size |
| `--by-occlusion` | flag | False | Group by occlusion level |

## 📊 Metrics Computed

### Quantitative (evaluate.py)

- **mAP@0.5**: Mean Average Precision at IoU 0.5 (standard)
- **mAP@0.75**: Mean Average Precision at IoU 0.75 (strict)
- **mAP@[0.5:0.95]**: COCO-style average over IoU thresholds
- **Per-class AP**: Average Precision for each of 10 classes
- **Recall**: Detection rate per class
- **Precision**: Positive predictive value per class

Output: `metrics/evaluation_results.json`

### Qualitative (visualize.py)

#### Visualization Types

1. **Side-by-Side Comparisons**: Ground truth vs predictions
2. **Failure Analysis**: Missed detections categorized by:
   - Object size (small, medium, large)
   - Occlusion level (none, partial, heavy)
   - Scene conditions (weather, time of day)
3. **Confusion Matrix**: Class confusion patterns
4. **Success Cases**: Best predictions per class

#### Output Files

- `viz_XXX_<image_name>.jpg`: Annotated images
- `failures_by_size.json`: Categorized failures
- `failures_by_occlusion.json`: Occlusion-based failures
- `confusion_matrix.png`: Class confusion heatmap
- `per_class_examples/`: Best/worst examples per class

## 🧪 Testing the Module

```bash
# Quick evaluation test (10 images)
python evaluate.py \
  --val-img-dir ../bdd100k_images_100k/100k/val \
  --val-ann-dir ../bdd100k_labels/100k/val \
  --checkpoint ../results/checkpoints/best_model.pth \
  --output-dir ./test_metrics \
  --batch-size 2

# Quick visualization test (5 images)
python visualize.py \
  --val-img-dir ../bdd100k_images_100k/100k/val \
  --val-ann-dir ../bdd100k_labels/100k/val \
  --checkpoint ../results/checkpoints/best_model.pth \
  --output-dir ./test_viz \
  --num-samples 5
```

## 📈 Understanding Results

### mAP Interpretation

| mAP@0.5 | Quality |
|---------|---------|
| > 50% | Excellent |
| 40-50% | Good |
| 30-40% | Acceptable |
| < 30% | Needs improvement |

### Common Failure Patterns

1. **Small objects** (traffic lights, signs): Hardest to detect
2. **Occluded objects**: Partially visible objects missed
3. **Class confusion**: Similar classes (truck vs bus)
4. **Weather conditions**: Performance drops in adverse weather

See [EVALUATION.md](EVALUATION.md) for detailed analysis.

## 🎨 Visualization Examples

### Color Coding

- **Green boxes**: Ground truth annotations
- **Blue boxes**: Correct predictions (TP)
- **Red boxes**: False positives (FP)
- **Orange boxes**: Missed detections (FN)

### Confidence Scores

Each prediction includes confidence score (0-1):
- > 0.7: High confidence
- 0.5-0.7: Medium confidence
- < 0.5: Low confidence (filtered by default)

## 🐛 Troubleshooting

**Checkpoint not found**:
- Ensure training completed: `ls ../results/checkpoints/`
- Check volume mounts in Docker

**Out of memory**:
- Reduce `--batch-size` to 1
- Reduce `--num-samples` for visualization

**Poor performance**:
- Check training logs for convergence
- Verify correct checkpoint loaded
- Adjust `--score-threshold` (lower = more detections)

**No visualizations generated**:
- Check output directory permissions
- Verify images directory path correct
- Ensure checkpoint compatible with model

## 📚 Dependencies

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy >= 1.26.0
- pillow >= 10.2.0
- opencv-python >= 4.8.0
- matplotlib >= 3.8.0
- seaborn >= 0.13.0
- pandas >= 2.1.0
- pycocotools >= 2.0.7

See [requirements.txt](requirements.txt) for complete list.

## 🔗 Related Documentation

- [../README.md](../README.md): Main project documentation
- [EVALUATION.md](EVALUATION.md): Detailed performance analysis
- [../model/README.md](../model/README.md): Training module
- [../data_analysis/README.md](../data_analysis/README.md): Data analysis module
