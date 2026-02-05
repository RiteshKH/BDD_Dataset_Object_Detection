"""BDD100K Dataset Analysis Script.

Performs comprehensive analysis of the BDD100K dataset including:
- Class distribution analysis
- Train/val split comparison
- Bounding box statistics
- Anomaly detection
- Scene attribute analysis
"""

import argparse
import json
import logging
from pathlib import Path

from parser import BDD100KParser, compare_splits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def analyze_dataset(
    train_json: str,
    val_json: str,
    output_dir: str,
) -> None:
    """Run complete dataset analysis pipeline.

    Args:
        train_json: Path to training annotations JSON
        val_json: Path to validation annotations JSON
        output_dir: Directory to save analysis results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting BDD100K Dataset Analysis ===")

    # Initialize parsers
    logger.info("Loading training data...")
    train_parser = BDD100KParser(train_json)

    logger.info("Loading validation data...")
    val_parser = BDD100KParser(val_json)

    # 1. Class Distribution Analysis
    logger.info("\n--- Class Distribution Analysis ---")
    train_dist = train_parser.get_class_distribution()
    val_dist = val_parser.get_class_distribution()

    distribution_results = {
        "train": train_dist,
        "val": val_dist,
        "total_train_objects": sum(train_dist.values()),
        "total_val_objects": sum(val_dist.values()),
    }

    logger.info(f"Training set - Total objects: {distribution_results['total_train_objects']}")
    for cls, count in sorted(train_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cls}: {count}")

    logger.info(f"\nValidation set - Total objects: {distribution_results['total_val_objects']}")
    for cls, count in sorted(val_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {cls}: {count}")

    # Save distribution results
    with open(output_path / "class_distribution.json", "w") as f:
        json.dump(distribution_results, f, indent=2)

    # 2. Train/Val Split Comparison
    logger.info("\n--- Train/Val Split Comparison ---")
    split_comparison = compare_splits(train_parser, val_parser)

    logger.info("Class distribution differences:")
    for cls, stats in split_comparison["class_distributions"].items():
        logger.info(
            f"  {cls}: Train {stats['train_percentage']:.2f}% vs Val {stats['val_percentage']:.2f}% "
            f"(diff: {stats['difference']:.2f}%)"
        )

    with open(output_path / "split_comparison.json", "w") as f:
        json.dump(split_comparison, f, indent=2)

    # 3. Image Statistics
    logger.info("\n--- Image Statistics ---")
    train_stats = train_parser.get_image_statistics()
    val_stats = val_parser.get_image_statistics()

    image_stats = {"train": train_stats, "val": val_stats}

    logger.info("Training set:")
    logger.info(f"  Total images: {train_stats['total_images']}")
    logger.info(f"  Images with objects: {train_stats['images_with_objects']}")
    logger.info(f"  Avg objects per image: {train_stats['avg_objects_per_image']:.2f}")
    logger.info(f"  Max objects per image: {train_stats['max_objects_per_image']}")

    logger.info("\nValidation set:")
    logger.info(f"  Total images: {val_stats['total_images']}")
    logger.info(f"  Images with objects: {val_stats['images_with_objects']}")
    logger.info(f"  Avg objects per image: {val_stats['avg_objects_per_image']:.2f}")
    logger.info(f"  Max objects per image: {val_stats['max_objects_per_image']}")

    with open(output_path / "image_statistics.json", "w") as f:
        json.dump(image_stats, f, indent=2)

    # 4. Bounding Box Statistics
    logger.info("\n--- Bounding Box Statistics ---")
    train_bbox = train_parser.get_bbox_statistics()
    val_bbox = val_parser.get_bbox_statistics()

    bbox_stats = {"train": train_bbox, "val": val_bbox}

    logger.info("Training set bbox statistics (top 5 classes by area):")
    train_sorted = sorted(
        [(k, v) for k, v in train_bbox.items() if v is not None],
        key=lambda x: x[1]["mean_area"],
        reverse=True,
    )[:5]
    for cls, stats in train_sorted:
        logger.info(
            f"  {cls}: mean_area={stats['mean_area']:.0f}, "
            f"mean_w={stats['mean_width']:.0f}, mean_h={stats['mean_height']:.0f}, "
            f"aspect_ratio={stats['mean_aspect_ratio']:.2f}"
        )

    with open(output_path / "bbox_statistics.json", "w") as f:
        json.dump(bbox_stats, f, indent=2)

    # 5. Occlusion and Truncation Statistics
    logger.info("\n--- Occlusion and Truncation Statistics ---")
    train_occlusion = train_parser.get_occlusion_truncation_stats()
    val_occlusion = val_parser.get_occlusion_truncation_stats()

    occlusion_stats = {"train": train_occlusion, "val": val_occlusion}

    logger.info("Training set occlusion stats (top 5 classes):")
    train_occ_sorted = sorted(
        train_occlusion.items(), key=lambda x: x[1]["occluded"], reverse=True
    )[:5]
    for cls, stats in train_occ_sorted:
        if stats["total"] > 0:
            occ_pct = (stats["occluded"] / stats["total"]) * 100
            trunc_pct = (stats["truncated"] / stats["total"]) * 100
            logger.info(
                f"  {cls}: {stats['occluded']}/{stats['total']} occluded ({occ_pct:.1f}%), "
                f"{stats['truncated']} truncated ({trunc_pct:.1f}%)"
            )

    with open(output_path / "occlusion_statistics.json", "w") as f:
        json.dump(occlusion_stats, f, indent=2)

    # 6. Scene Attributes
    logger.info("\n--- Scene Attributes Analysis ---")
    train_attrs = train_parser.get_scene_attributes()
    val_attrs = val_parser.get_scene_attributes()

    scene_attrs = {"train": train_attrs, "val": val_attrs}

    logger.info("Training set scene distribution:")
    logger.info(f"  Weather: {train_attrs['weather']}")
    logger.info(f"  Time of day: {train_attrs['timeofday']}")
    logger.info(f"  Scene: {train_attrs['scene']}")

    with open(output_path / "scene_attributes.json", "w") as f:
        json.dump(scene_attrs, f, indent=2)

    # 7. Anomaly Detection
    logger.info("\n--- Anomaly Detection ---")
    train_anomalies = train_parser.find_anomalies()
    val_anomalies = val_parser.find_anomalies()

    anomalies = {"train": train_anomalies, "val": val_anomalies}

    logger.info("Training set anomalies:")
    for anomaly_type, images in train_anomalies.items():
        logger.info(f"  {anomaly_type}: {len(images)} images")
        if images and len(images) <= 5:
            logger.info(f"    Examples: {images[:5]}")

    logger.info("\nValidation set anomalies:")
    for anomaly_type, images in val_anomalies.items():
        logger.info(f"  {anomaly_type}: {len(images)} images")

    with open(output_path / "anomalies.json", "w") as f:
        json.dump(anomalies, f, indent=2)

    logger.info(f"\n=== Analysis Complete! Results saved to {output_dir} ===")


def main():
    """Main entry point for the analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze BDD100K dataset for object detection"
    )
    parser.add_argument(
        "--train-json",
        type=str,
        default="/data/labels/100k/train",
        help="Path to training annotations JSON or directory",
    )
    parser.add_argument(
        "--val-json",
        type=str,
        default="/data/labels/100k/val",
        help="Path to validation annotations JSON or directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/app/results",
        help="Directory to save analysis results",
    )

    args = parser.parse_args()

    try:
        analyze_dataset(args.train_json, args.val_json, args.output_dir)
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
