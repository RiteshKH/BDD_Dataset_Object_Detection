"""Visualization script for qualitative analysis of object detection results.

Provides tools for:
- Visualizing predictions vs ground truth
- Analyzing failure patterns by scene conditions
- Clustering errors by object size, occlusion, etc.
- Generating comparison images
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dataset import BDD100K_CLASSES, create_dataloader
from model.model_architecture import get_model, load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Color palette for classes (BGR for OpenCV)
CLASS_COLORS = {
    "car": (0, 255, 0),  # Green
    "truck": (255, 165, 0),  # Orange
    "bus": (255, 0, 255),  # Magenta
    "person": (0, 0, 255),  # Red
    "bike": (255, 255, 0),  # Cyan
    "motor": (128, 0, 128),  # Purple
    "rider": (255, 0, 0),  # Blue
    "traffic light": (0, 255, 255),  # Yellow
    "traffic sign": (255, 192, 203),  # Pink
    "train": (165, 42, 42),  # Brown
}


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: List[str],
    scores: Optional[np.ndarray] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes on image.

    Args:
        image: Image array (BGR format)
        boxes: Bounding boxes as Nx4 array [x1, y1, x2, y2]
        labels: List of class labels
        scores: Optional confidence scores
        color: Box color (BGR)
        thickness: Line thickness

    Returns:
        Image with drawn boxes
    """
    image = image.copy()

    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)

        # Get color for this class
        box_color = CLASS_COLORS.get(label, color)

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, thickness)

        # Draw label
        label_text = label
        if scores is not None:
            label_text = f"{label} {scores[i]:.2f}"

        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            box_color,
            -1,
        )

        # Draw text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return image


def visualize_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    num_images: int = 20,
    score_threshold: float = 0.5,
) -> None:
    """Visualize model predictions on validation images.

    Creates side-by-side comparisons of ground truth and predictions.

    Args:
        model: Detection model
        dataloader: Validation data loader
        device: Device to run inference on
        output_dir: Directory to save visualizations
        num_images: Number of images to visualize
        score_threshold: Minimum confidence for displaying predictions
    """
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating visualizations for {num_images} images...")

    images_processed = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Visualizing"):
            if images_processed >= num_images:
                break

            # Move data to device
            img_tensors = [img.to(device) for img in images]

            # Get predictions
            predictions = model(img_tensors)

            # Process each image in batch
            for img_tensor, pred, target in zip(img_tensors, predictions, targets):
                if images_processed >= num_images:
                    break

                # Convert tensor to numpy (CHW -> HWC, denormalize)
                img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Ground truth
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                gt_label_names = [BDD100K_CLASSES[int(label) - 1] for label in gt_labels]

                # Predictions
                pred_boxes = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()
                pred_scores = pred["scores"].cpu().numpy()

                # Filter predictions by score
                keep_idx = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep_idx]
                pred_labels = pred_labels[keep_idx]
                pred_scores = pred_scores[keep_idx]
                pred_label_names = [
                    BDD100K_CLASSES[int(label) - 1] for label in pred_labels
                ]

                # Draw ground truth (left image)
                img_gt = draw_boxes(
                    img_np, gt_boxes, gt_label_names, color=(0, 255, 0), thickness=2
                )

                # Draw predictions (right image)
                img_pred = draw_boxes(
                    img_np,
                    pred_boxes,
                    pred_label_names,
                    pred_scores,
                    color=(255, 0, 0),
                    thickness=2,
                )

                # Combine images side by side
                combined = np.hstack([img_gt, img_pred])

                # Add title
                h, w = combined.shape[:2]
                title_img = np.zeros((50, w, 3), dtype=np.uint8)
                cv2.putText(
                    title_img,
                    "Ground Truth",
                    (w // 4 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    title_img,
                    "Predictions",
                    (3 * w // 4 - 50, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                final_img = np.vstack([title_img, combined])

                # Save image
                img_name = target["image_id"]
                # Ensure output filename has a valid image extension
                output_file = output_path / f"viz_{images_processed:03d}_{img_name}.png"
                cv2.imwrite(str(output_file), final_img)

                images_processed += 1

    logger.info(f"Saved {images_processed} visualizations to {output_dir}")


def analyze_failures_by_size(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    iou_threshold: float = 0.5,
) -> Dict[str, List]:
    """Analyze model failures by object size.

    Categorizes missed detections into small, medium, large objects.

    Args:
        model: Detection model
        dataloader: Validation data loader
        device: Device to run inference on
        output_dir: Directory to save analysis
        iou_threshold: IoU threshold for considering a match

    Returns:
        Dictionary with failure statistics by size
    """
    model.eval()

    failures = {"small": [], "medium": [], "large": []}
    size_thresholds = {"small": 32**2, "medium": 96**2}  # COCO definitions

    logger.info("Analyzing failures by object size...")

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Analyzing"):
            img_tensors = [img.to(device) for img in images]
            predictions = model(img_tensors)

            for pred, target in zip(predictions, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()
                gt_areas = target["area"].cpu().numpy()

                pred_boxes = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()

                # Track which GTs are detected
                gt_detected = np.zeros(len(gt_boxes), dtype=bool)

                # Match predictions to GTs
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if pred_label == gt_label and not gt_detected[i]:
                            # Calculate IoU
                            iou = calculate_iou(pred_box, gt_box)
                            if iou >= iou_threshold:
                                gt_detected[i] = True
                                break

                # Categorize missed detections
                for i, (detected, area, label) in enumerate(
                    zip(gt_detected, gt_areas, gt_labels)
                ):
                    if not detected:
                        class_name = BDD100K_CLASSES[int(label) - 1]

                        if area < size_thresholds["small"]:
                            category = "small"
                        elif area < size_thresholds["medium"]:
                            category = "medium"
                        else:
                            category = "large"

                        failures[category].append(
                            {
                                "class": str(class_name),
                                "area": float(area) if hasattr(area, 'item') else area,
                                "image": str(target["image_id"]),
                            }
                        )

    # Generate summary statistics
    summary = {}
    for category, fails in failures.items():
        class_counts = defaultdict(int)
        for fail in fails:
            class_counts[fail["class"]] += 1

        summary[category] = {
            "total_failures": len(fails),
            "by_class": dict(class_counts),
        }

    logger.info("\n=== Failures by Object Size ===")
    for category, stats in summary.items():
        logger.info(f"{category.capitalize()}: {stats['total_failures']} failures")
        for cls, count in sorted(
            stats["by_class"].items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"  {cls}: {count}")

    # Save detailed analysis
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "failures_by_size.json", "w") as f:
        json.dump({"summary": summary, "details": failures}, f, indent=2)

    return failures


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def generate_confusion_matrix(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> np.ndarray:
    """Generate confusion matrix for class predictions.

    Args:
        model: Detection model
        dataloader: Validation data loader
        device: Device to run inference on
        output_dir: Directory to save confusion matrix
        iou_threshold: IoU threshold for matching
        score_threshold: Confidence threshold for predictions

    Returns:
        Confusion matrix as numpy array
    """
    model.eval()
    num_classes = len(BDD100K_CLASSES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    logger.info("Generating confusion matrix...")

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Processing"):
            img_tensors = [img.to(device) for img in images]
            predictions = model(img_tensors)

            for pred, target in zip(predictions, targets):
                gt_boxes = target["boxes"].cpu().numpy()
                gt_labels = target["labels"].cpu().numpy()

                pred_boxes = pred["boxes"].cpu().numpy()
                pred_labels = pred["labels"].cpu().numpy()
                pred_scores = pred["scores"].cpu().numpy()

                # Filter predictions
                keep_idx = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep_idx]
                pred_labels = pred_labels[keep_idx]

                # Match predictions to ground truths
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_gt_label = None

                    for gt_box, gt_label in zip(gt_boxes, gt_labels):
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_label = gt_label

                    if best_iou >= iou_threshold and best_gt_label is not None:
                        # Matched detection
                        gt_idx = int(best_gt_label) - 1
                        pred_idx = int(pred_label) - 1
                        confusion_matrix[gt_idx, pred_idx] += 1

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, BDD100K_CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, BDD100K_CLASSES)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

    return confusion_matrix


def main():
    """Main entry point for visualization script."""
    parser = argparse.ArgumentParser(
        description="Visualize BDD100K detection results"
    )

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

    # Visualization arguments
    parser.add_argument(
        "--num-images", type=int, default=20, help="Number of images to visualize"
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for visualization",
    )
    parser.add_argument(
        "--analyze-failures", action="store_true", help="Analyze failure patterns"
    )
    parser.add_argument(
        "--confusion-matrix", action="store_true", help="Generate confusion matrix"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/visualizations",
        help="Output directory for visualizations",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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
        batch_size=1,  # Process one image at a time for visualization
        shuffle=False,
        num_workers=2,
    )

    # Generate visualizations
    visualize_predictions(
        model,
        val_loader,
        device,
        args.output_dir,
        args.num_images,
        args.score_threshold,
    )

    # Optional: Analyze failures
    if args.analyze_failures:
        analyze_failures_by_size(model, val_loader, device, args.output_dir)

    # Optional: Generate confusion matrix
    if args.confusion_matrix:
        generate_confusion_matrix(model, val_loader, device, args.output_dir)

    logger.info("\nVisualization completed successfully!")


if __name__ == "__main__":
    main()
