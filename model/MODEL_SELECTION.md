# Model Selection and Architecture

## Chosen Model: Faster R-CNN with ResNet-50 FPN Backbone

### Executive Summary

For the BDD100K object detection task, **Faster R-CNN with ResNet-50 Feature Pyramid Network (FPN)** backbone is selected as the primary model architecture. This choice is based on comprehensive analysis of the dataset characteristics and the specific requirements of autonomous driving object detection.

## Rationale for Model Selection

### 1. Dataset Characteristics Alignment

Based on the data analysis findings, the BDD100K dataset presents several key challenges:

**Multi-Scale Objects**:
- Small objects: traffic lights and signs (often <10,000 pixels²)
- Medium objects: persons, riders, motorcycles
- Large objects: cars, trucks, buses

**Faster R-CNN with FPN** is specifically designed to handle multi-scale detection through:
- **Feature Pyramid Network**: Extracts features at multiple scales
- **Multi-level RPN**: Proposes regions at different resolutions
- **RoI pooling at different scales**: Processes objects of varying sizes effectively

### 2. Class Imbalance Handling

The dataset shows severe class imbalance (cars dominate, trains are rare):

- Faster R-CNN's **two-stage approach** provides more stable training compared to one-stage detectors
- The region proposal stage helps focus on potential object locations before classification
- Easier to apply class-specific thresholds during inference

### 3. Accuracy vs Speed Trade-offs

For autonomous driving evaluation and development:

| Aspect | Priority | Faster R-CNN Performance |
|--------|----------|--------------------------|
| Accuracy | High | Excellent (state-of-the-art on COCO) |
| Recall | Critical | High recall through RPN proposals |
| Inference Speed | Medium | Moderate (~5-15 FPS on GPU) |
| Training Stability | High | Stable two-stage training |

While real-time deployment might require faster models (YOLO, EfficientDet), **Faster R-CNN provides the accuracy baseline** needed for thorough evaluation and analysis.

### 4. Pre-trained Weights Availability

- **COCO pre-trained weights available**: 80-class COCO dataset includes most BDD classes
- **Transfer learning effective**: Fine-tuning from COCO provides strong initialization
- **Well-documented**: Extensive research and implementation resources

## Architecture Deep Dive

### Overall Architecture

```
Input Image (1280x720)
         ↓
ResNet-50 Backbone (C2, C3, C4, C5 feature maps)
         ↓
Feature Pyramid Network (P2, P3, P4, P5, P6)
         ↓
Region Proposal Network (RPN)
         ↓
RoI Align (extract fixed-size features)
         ↓
Box Head (classification + regression)
         ↓
Final Predictions (boxes, scores, classes)
```

### Component Breakdown

#### 1. **ResNet-50 Backbone**

**Architecture**:
- 50 layers deep convolutional network
- Residual connections enable training deep networks
- Outputs multi-scale feature maps: C2 (1/4), C3 (1/8), C4 (1/16), C5 (1/32)

**Why ResNet-50 over alternatives?**:
- **vs ResNet-101**: Better speed-accuracy tradeoff (lighter than 101)
- **vs MobileNet**: Higher accuracy critical for safety-critical detection
- **vs VGG**: More efficient with residual connections
- **Pretrained on ImageNet**: Strong visual feature representations

#### 2. **Feature Pyramid Network (FPN)**

**Architecture**:
- Bottom-up pathway: ResNet backbone
- Top-down pathway: Upsampling higher-level features
- Lateral connections: Merge features across scales

**Benefits for BDD100K**:
- **P2 (1/4 scale)**: Detects small objects (traffic lights, distant signs)
- **P3-P4 (1/8-1/16)**: Medium objects (persons, motorcycles)
- **P5-P6 (1/32-1/64)**: Large objects (trucks, buses)

This directly addresses the multi-scale challenge identified in data analysis.

#### 3. **Region Proposal Network (RPN)**

**Architecture**:
- Sliding window over FPN feature maps
- Predicts objectness score + box regression for each anchor
- Anchors at multiple scales (32², 64², 128², 256², 512²) and aspect ratios (0.5, 1.0, 2.0)

**Custom Anchor Design for BDD100K**:
Based on bbox statistics from data analysis:
- **Vertical anchors (0.4-0.6)**: Capture persons, riders
- **Horizontal anchors (1.5-2.5)**: Capture vehicles
- **Square anchors (0.8-1.2)**: Capture traffic signs

#### 4. **RoI Align**

**Functionality**:
- Extracts fixed-size (7x7) feature maps from proposals
- Uses bilinear interpolation (avoids quantization errors of RoI Pooling)
- Processes ~2000 proposals per image

**Advantage**:
- Precise alignment crucial for small objects (traffic lights)
- Preserves spatial information better than RoI Pooling

#### 5. **Detection Head**

**Architecture**:
- **Box classifier**: 2-layer MLP → softmax over 11 classes (10 + background)
- **Box regressor**: 2-layer MLP → 4 coordinates (refined bbox)
- **Loss functions**:
  - Cross-entropy loss for classification
  - Smooth L1 loss for box regression

**Customization for BDD100K**:
- Replace final layer to output 11 classes (from COCO's 80)
- Fine-tune with BDD-specific data

### Model Configuration

```python
# Key hyperparameters
num_classes = 11  # 10 detection classes + background
min_size = 800    # Min image dimension during training
max_size = 1333   # Max image dimension

# Anchor generation
anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

# NMS parameters
nms_thresh = 0.7  # IoU threshold for non-maximum suppression
score_thresh = 0.05  # Minimum confidence for detections
```

## Alternative Models Considered

### 1. **YOLO (You Only Look Once) v8**

**Pros**:
- Fast inference (~30-60 FPS)
- Single-stage simplicity
- Good for real-time deployment

**Cons**:
- Lower accuracy on small objects (critical for BDD100K traffic infrastructure)
- Less stable training with severe class imbalance
- Grid-based detection can miss small objects

**Decision**: Rejected for initial development; consider for deployment phase

### 2. **RetinaNet**

**Pros**:
- Focal loss handles class imbalance well
- One-stage efficiency
- FPN for multi-scale detection

**Cons**:
- Slightly lower accuracy than Faster R-CNN
- More sensitive to hyperparameters
- Less extensive pre-trained options

**Decision**: Viable alternative; implemented as secondary option

## Training Strategy

### Transfer Learning Approach

1. **Load COCO pre-trained weights**:
   - Backbone: Fully pretrained
   - RPN: Pretrained on COCO
   - Detection head: Replace and reinitialize for 11 classes

2. **Fine-tuning schedule**:
   - **Stage 1 (epochs 1-3)**: Train only detection head (lr=0.005)
   - **Stage 2 (epochs 4-10)**: Train full model (lr=0.001)
   - **Stage 3 (epochs 11+)**: Fine-tune with lower lr (lr=0.0001)

3. **Data augmentation**:
   - Horizontal flipping
   - Color jittering (for weather variations)
   - Random cropping (for different scales)

### Optimization

- **Optimizer**: SGD with momentum (0.9)
- **Weight decay**: 0.0005
- **Learning rate**: 0.005 initial, decay by 0.1 every 3 epochs
- **Batch size**: 4 images (limited by GPU memory with 1280x720 images)

## Expected Performance

Based on literature and COCO benchmarks:

| Metric | Expected Range | Notes |
|--------|----------------|-------|
| mAP@0.5 | 45-55% | COCO-style metric |
| mAP@0.75 | 30-40% | Stricter threshold |
| mAP (avg) | 35-45% | Average over IoU 0.5:0.95 |
| Inference time | 100-200ms/img | On GPU (RTX 3080) |

**Per-class expectations**:
- High performance: car, person (abundant data)
- Medium performance: truck, bus, traffic sign
- Lower performance: train (rare), traffic light (small)

## Model Limitations and Mitigation

### Limitations

1. **Small object detection**: May miss distant traffic lights
   - **Mitigation**: FPN helps; consider separate small object head

2. **Dense scenes**: Many overlapping objects slow inference
   - **Mitigation**: Adjust NMS threshold; use soft-NMS

3. **Class imbalance**: Bias toward dominant classes
   - **Mitigation**: Monitor per-class metrics; adjust loss weights

4. **Speed**: Not real-time capable
   - **Mitigation**: For deployment, consider knowledge distillation to YOLO


## Implementation Details

### Code Structure

```python
# Load model
from model_architecture import get_faster_rcnn_model

model = get_faster_rcnn_model(
    num_classes=11,
    pretrained=True,  # COCO weights
    min_size=800,
    max_size=1333
)

# Model is ready for training
# Detection head already replaced for 11 classes
```

### Computational Requirements

- **GPU memory**: ~8-10GB for batch size 4
- **Training time**: ~10-15 hours for 10 epochs (on single RTX 3080)
- **Disk space**: ~500MB per checkpoint

## Conclusion

**Faster R-CNN with ResNet-50 FPN** is the optimal choice for BDD100K object detection given:
- Multi-scale object requirements
- Need for high accuracy and recall
- Class imbalance challenges
- Availability of pre-trained weights
- Well-established training procedures

This architecture provides a strong baseline for evaluation, with clear paths for optimization if real-time inference becomes a requirement.
