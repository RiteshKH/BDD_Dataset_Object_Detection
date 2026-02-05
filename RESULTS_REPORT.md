# BDD100K Object Detection Results Report

## Model Checkpoint: `checkpoint_epoch_1.pth`

### 1. Quantitative Evaluation (from `results/metrics/evaluation_metrics.json`)

#### mAP (mean Average Precision)
- **mAP@0.5:** 0.113
- **mAP@0.75:** 0.051
- **mAP@[0.5:0.95] (COCO avg):** 0.059

#### Per-Class AP (at IoU=0.5)
| Class          | AP     |
|----------------|--------|
| car            | 0.573  |
| truck          | 0.002  |
| bus            | 0.000  |
| person         | 0.175  |
| bike           | 0.004  |
| motor          | 0.000  |
| rider          | 0.000  |
| traffic light  | 0.188  |
| traffic sign   | 0.191  |
| train          | 0.000  |

#### Observations
- The model performs best on **cars** (AP=0.573), with moderate performance on **person**, **traffic light**, and **traffic sign**.
- Performance is very low or zero for **bus**, **motor**, **rider**, and **train**.
- mAP drops significantly at higher IoU thresholds, indicating localization challenges.

### 2. Failure Analysis (from `results/visualizations/failures_by_size.json`)

#### Missed Detections by Object Size
| Size   | Total Failures |
|--------|----------------|
| Small  | 56,116         |
| Medium | 16,532         |
| Large  | 4,115          |

#### Top Failure Classes (Small Objects)
- **Traffic sign:** 17,605 missed
- **Traffic light:** 13,264 missed
- **Car:** 17,397 missed
- **Person:** 6,213 missed

#### Top Failure Classes (Medium Objects)
- **Car:** 4,344 missed
- **Person:** 4,075 missed
- **Traffic sign:** 3,424 missed
- **Truck:** 1,807 missed

#### Top Failure Classes (Large Objects)
- **Bus:** 672 missed
- **Truck:** 1,634 missed
- **Car:** 656 missed

#### Insights
- The majority of missed detections are for **small objects**, especially traffic signs, traffic lights, and cars.
- **Large objects** are missed less frequently, but trucks and buses still show notable failures.
- Classes with very low AP (bus, motor, rider, train) also have high miss rates, indicating the need for more data or model improvements for these categories.

### 3. Recommendations
- **Improve small object detection:** Consider data augmentation, higher resolution inputs, or specialized loss functions (e.g., focal loss).
- **Class imbalance:** Address underrepresented classes (bus, motor, rider, train) with oversampling or class-balanced loss.
- **Localization:** mAP drop at higher IoU suggests improving bounding box regression.
- **Further analysis:** Investigate scene conditions (night, weather) and occlusion for additional failure patterns.

---

_Report generated automatically from evaluation and failure analysis outputs._
