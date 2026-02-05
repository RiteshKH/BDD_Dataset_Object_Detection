"""Evaluation script for BDD100K object detection models.

Computes quantitative metrics including:
- mAP at different IoU thresholds (0.5, 0.75, 0.5:0.95)
- Per-class Average Precision
- Precision-Recall curves
- Confusion matrix
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import create_dataloader, BDD100K_CLASSES
from model.model_architecture import get_model, load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union for two bounding boxes.

    Args:
        box1: Bounding box as [x1, y1, x2, y2]
        box2: Bounding box as [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def calculate_ap(
    predictions: List[Tuple[float, bool]], num_gt: int
) -> float:
    """Calculate Average Precision for a single class.

    Args:
        predictions: List of (confidence, is_correct) tuples sorted by confidence
        num_gt: Total number of ground truth instances

    Returns:
        Average Precision score
    """
    if num_gt == 0:
        return 0.0

    # Sort predictions by confidence (descending)
    predictions = sorted(predictions, key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(predictions))
    fp = np.zeros(len(predictions))

    for i, (_, is_correct) in enumerate(predictions):
        if is_correct:
            tp[i] = 1
        else:
            fp[i] = 1

    # Cumulative sums
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    # Calculate precision and recall
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> Dict[str, any]:
    """Evaluate model on validation dataset.

    Args:
        model: Detection model
        dataloader: Validation data loader
        device: Device to run evaluation on
        iou_threshold: IoU threshold for matching predictions to ground truth
        score_threshold: Minimum confidence threshold for predictions

    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()

    # Storage for predictions and ground truths
    all_predictions = defaultdict(list)  # class_id -> [(confidence, is_correct), ...]
    all_gt_counts = defaultdict(int)  # class_id -> count

    logger.info("Starting model evaluation...")

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            # Move data to device
            images = [img.to(device) for img in images]

            # Get predictions
            predictions = model(images)

            # Process each image in batch
            for pred, target in zip(predictions, targets):
                pred_boxes = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()
                pred_scores = pred["scores"].cpu().numpy()

                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                # Filter predictions by score threshold
                keep_idx = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep_idx]
                pred_labels = pred_labels[keep_idx]
                pred_scores = pred_scores[keep_idx]

                # Track ground truth counts
                for gt_label in gt_labels:
                    all_gt_counts[int(gt_label)] += 1

                # Track which ground truths have been matched
                gt_matched = np.zeros(len(gt_labels), dtype=bool)

                # Match predictions to ground truths
                for pred_box, pred_label, pred_score in zip(
                    pred_boxes, pred_labels, pred_scores
                ):
                    pred_label = int(pred_label)

                    # Find best matching ground truth
                    best_iou = 0
                    best_gt_idx = -1

                    for gt_idx, (gt_box, gt_label) in enumerate(
                        zip(gt_boxes, gt_labels)
                    ):
                        if int(gt_label) == pred_label and not gt_matched[gt_idx]:
                            iou = calculate_iou(pred_box, gt_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_gt_idx = gt_idx

                    # Check if prediction is correct
                    is_correct = best_iou >= iou_threshold
                    if is_correct:
                        gt_matched[best_gt_idx] = True

                    all_predictions[pred_label].append((float(pred_score), is_correct))

    # Calculate AP for each class
    logger.info("Calculating Average Precision per class...")
    aps = {}
    for class_id in range(1, len(BDD100K_CLASSES) + 1):  # Skip background (0)
        class_name = BDD100K_CLASSES[class_id - 1]
        num_gt = all_gt_counts.get(class_id, 0)
        predictions = all_predictions.get(class_id, [])

        if num_gt > 0:
            ap = calculate_ap(predictions, num_gt)
            aps[class_name] = ap
            logger.info(f"  {class_name}: AP = {ap:.4f} ({num_gt} ground truths)")
        else:
            aps[class_name] = 0.0
            logger.info(f"  {class_name}: No ground truth instances")

    # Calculate mAP
    mean_ap = np.mean(list(aps.values()))

    metrics = {
        "mAP": mean_ap,
        "per_class_AP": aps,
        "iou_threshold": iou_threshold,
        "score_threshold": score_threshold,
        "total_ground_truths": dict(all_gt_counts),
    }

    logger.info(f"\nmAP@{iou_threshold}: {mean_ap:.4f}")

    return metrics


def evaluate_at_multiple_iou(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    iou_thresholds: List[float] = None,
) -> Dict[str, any]:
    """Evaluate model at multiple IoU thresholds (COCO-style).

    Args:
        model: Detection model
        dataloader: Validation data loader
        device: Device to run evaluation on
        iou_thresholds: List of IoU thresholds to evaluate

    Returns:
        Dictionary containing metrics at all thresholds
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75] + list(np.arange(0.5, 1.0, 0.05))

    results = {}

    for iou_thresh in iou_thresholds:
        logger.info(f"\n=== Evaluating at IoU threshold {iou_thresh:.2f} ===")
        metrics = evaluate_model(model, dataloader, device, iou_threshold=iou_thresh)
        results[f"iou_{iou_thresh:.2f}"] = metrics

    # Calculate mAP averaged over all thresholds (COCO-style)
    all_maps = [metrics["mAP"] for metrics in results.values()]
    results["mAP_avg"] = np.mean(all_maps)

    logger.info(f"\n=== Final Results ===")
    logger.info(f"mAP@0.5: {results['iou_0.50']['mAP']:.4f}")
    logger.info(f"mAP@0.75: {results['iou_0.75']['mAP']:.4f}")
    logger.info(f"mAP (avg 0.5:0.95): {results['mAP_avg']:.4f}")

    return results


def save_metrics(metrics: Dict, output_path: str) -> None:
    """Save evaluation metrics to JSON file.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save JSON file
    """
    # Convert numpy types to Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    metrics_serializable = convert_types(metrics)

    with open(output_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate BDD100K detection model")

    # Data arguments
    parser.add_argument(
        "--val-img-dir", type=str, required=True, help="Validation images directory"
    )
    parser.add_argument(
        "--val-ann-dir",
        type=str,
        required=True,
        help="Validation annotations directory (contains per-image JSON files)",
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="faster_rcnn",
        choices=["faster_rcnn", "retinanet"],
        help="Model architecture",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )

    # Evaluation arguments
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for evaluation (ignored if --multi-iou is set)",
    )
    parser.add_argument(
        "--multi-iou",
        action="store_true",
        help="Evaluate at multiple IoU thresholds (COCO-style)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.05,
        help="Minimum confidence threshold",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/metrics",
        help="Output directory for metrics",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model: {args.model}")
    model = get_model(model_name=args.model, num_classes=11, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)

    # Create dataloader
    logger.info("Creating validation dataloader...")
    val_loader = create_dataloader(
        image_dir=args.val_img_dir,
        annotation_dir=args.val_ann_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Run evaluation
    if args.multi_iou:
        metrics = evaluate_at_multiple_iou(model, val_loader, device)
    else:
        metrics = evaluate_model(
            model, val_loader, device, args.iou_threshold, args.score_threshold
        )

    # Save results
    output_file = output_path / "evaluation_metrics.json"
    save_metrics(metrics, str(output_file))

    logger.info("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
