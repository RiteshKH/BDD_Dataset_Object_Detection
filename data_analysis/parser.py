"""BDD100K Dataset Parser.

Handles parsing of BDD100K annotations for object detection task.
Supports both single consolidated JSON and per-image JSON file formats.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# BDD100K detection classes (10 classes for object detection task)
DETECTION_CLASSES = [
    "car",
    "truck",
    "bus",
    "person",
    "bike",
    "motorcycle",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
]


class BDD100KParser:
    """Parser for BDD100K JSON annotations.

    Handles extraction of bounding boxes, class labels, and metadata
    for the 10 object detection classes. Supports both consolidated
    JSON format and per-image JSON file directories.

    Args:
        json_path: Path to annotations JSON file or directory of JSON files
    """

    def __init__(self, json_path: str):
        """Initialize parser and load annotations.

        Args:
            json_path: Path to annotations JSON or directory
        """
        self.json_path = Path(json_path)
        self.annotations = []
        self._load_annotations()

    def _load_annotations(self) -> None:
        """Load annotations from JSON file or directory of JSON files.

        Raises:
            FileNotFoundError: If annotation path does not exist
        """
        if not self.json_path.exists():
            logger.error(f"Annotation file not found: {self.json_path}")
            raise FileNotFoundError(f"Annotation path not found: {self.json_path}")

        # Check if path is directory (per-image JSON format)
        if self.json_path.is_dir():
            logger.info(f"Loading annotations from directory: {self.json_path}")
            json_files = list(self.json_path.glob("*.json"))
            logger.info(f"Found {len(json_files)} JSON files")

            for json_file in json_files:
                try:
                    with open(json_file, "r") as f:
                        annotation = json.load(f)
                        self.annotations.append(annotation)
                except Exception as e:
                    logger.warning(f"Failed to load {json_file}: {e}")

            logger.info(f"Successfully loaded {len(self.annotations)} annotations")

        # Otherwise treat as single consolidated JSON file
        else:
            logger.info(f"Loading annotations from file: {self.json_path}")
            with open(self.json_path, "r") as f:
                data = json.load(f)
                # Handle both list format and dict with 'frames' key
                if isinstance(data, list):
                    self.annotations = data
                elif isinstance(data, dict) and "frames" in data:
                    self.annotations = data["frames"]
                else:
                    self.annotations = [data]

            logger.info(f"Successfully loaded {len(self.annotations)} annotations")

    def get_class_distribution(self) -> Dict[str, int]:
        """Calculate object count per detection class.

        Returns:
            Dictionary mapping class names to object counts
        """
        class_counts = defaultdict(int)

        for annotation in self.annotations:
            labels = annotation.get("labels", [])
            for label in labels:
                category = label.get("category", "")
                # Only count detection classes
                if category in DETECTION_CLASSES:
                    class_counts[category] += 1

        return dict(class_counts)

    def get_image_statistics(self) -> Dict:
        """Calculate statistics about images in the dataset.

        Returns:
            Dictionary with image-level statistics
        """
        total_images = len(self.annotations)
        images_with_objects = 0
        objects_per_image = []

        for annotation in self.annotations:
            labels = annotation.get("labels", [])
            # Filter to detection classes only
            detection_labels = [
                l for l in labels if l.get("category", "") in DETECTION_CLASSES
            ]

            if detection_labels:
                images_with_objects += 1
                objects_per_image.append(len(detection_labels))
            else:
                objects_per_image.append(0)

        return {
            "total_images": total_images,
            "images_with_objects": images_with_objects,
            "images_without_objects": total_images - images_with_objects,
            "avg_objects_per_image": (
                np.mean(objects_per_image) if objects_per_image else 0
            ),
            "max_objects_per_image": max(objects_per_image) if objects_per_image else 0,
            "min_objects_per_image": min(objects_per_image) if objects_per_image else 0,
        }

    def get_bbox_statistics(self) -> Dict[str, Optional[Dict]]:
        """Calculate bounding box statistics per class.

        Returns:
            Dictionary mapping class names to bbox statistics
        """
        bbox_data = defaultdict(lambda: {"widths": [], "heights": [], "areas": []})

        for annotation in self.annotations:
            labels = annotation.get("labels", [])
            for label in labels:
                category = label.get("category", "")
                if category not in DETECTION_CLASSES:
                    continue

                box2d = label.get("box2d", {})
                if not box2d:
                    continue

                x1, y1 = box2d.get("x1", 0), box2d.get("y1", 0)
                x2, y2 = box2d.get("x2", 0), box2d.get("y2", 0)

                width = x2 - x1
                height = y2 - y1
                area = width * height

                if width > 0 and height > 0:
                    bbox_data[category]["widths"].append(width)
                    bbox_data[category]["heights"].append(height)
                    bbox_data[category]["areas"].append(area)

        # Calculate statistics
        stats = {}
        for category in DETECTION_CLASSES:
            if category in bbox_data and bbox_data[category]["areas"]:
                data = bbox_data[category]
                stats[category] = {
                    "mean_width": float(np.mean(data["widths"])),
                    "mean_height": float(np.mean(data["heights"])),
                    "mean_area": float(np.mean(data["areas"])),
                    "mean_aspect_ratio": float(
                        np.mean(np.array(data["widths"]) / np.array(data["heights"]))
                    ),
                    "count": len(data["areas"]),
                }
            else:
                stats[category] = None

        return stats

    def get_occlusion_truncation_stats(self) -> Dict[str, Dict]:
        """Calculate occlusion and truncation statistics per class.

        Returns:
            Dictionary mapping class names to occlusion/truncation stats
        """
        stats = defaultdict(lambda: {"total": 0, "occluded": 0, "truncated": 0})

        for annotation in self.annotations:
            labels = annotation.get("labels", [])
            for label in labels:
                category = label.get("category", "")
                if category not in DETECTION_CLASSES:
                    continue

                stats[category]["total"] += 1

                attributes = label.get("attributes", {})
                if attributes.get("occluded", False):
                    stats[category]["occluded"] += 1
                if attributes.get("truncated", False):
                    stats[category]["truncated"] += 1

        return dict(stats)

    def get_scene_attributes(self) -> Dict:
        """Analyze scene attributes (weather, time of day, scene type).

        Returns:
            Dictionary with scene attribute distributions
        """
        weather_counts = defaultdict(int)
        timeofday_counts = defaultdict(int)
        scene_counts = defaultdict(int)

        for annotation in self.annotations:
            attributes = annotation.get("attributes", {})

            weather = attributes.get("weather", "unknown")
            timeofday = attributes.get("timeofday", "unknown")
            scene = attributes.get("scene", "unknown")

            weather_counts[weather] += 1
            timeofday_counts[timeofday] += 1
            scene_counts[scene] += 1

        return {
            "weather": dict(weather_counts),
            "timeofday": dict(timeofday_counts),
            "scene": dict(scene_counts),
        }

    def find_anomalies(self) -> Dict[str, List[str]]:
        """Identify anomalous samples for failure analysis.

        Returns:
            Dictionary mapping anomaly types to lists of image names
        """
        anomalies = {
            "extremely_small_objects": [],
            "extremely_large_objects": [],
            "many_objects": [],
            "highly_occluded": [],
            "night_scenes": [],
            "adverse_weather": [],
        }

        for annotation in self.annotations:
            name = annotation.get("name", "unknown")
            labels = annotation.get("labels", [])
            attributes = annotation.get("attributes", {})

            # Filter detection labels
            detection_labels = [
                l for l in labels if l.get("category", "") in DETECTION_CLASSES
            ]

            # Check for many objects
            if len(detection_labels) > 30:
                anomalies["many_objects"].append(name)

            # Check occlusion
            occluded_count = sum(
                1 for l in detection_labels if l.get("attributes", {}).get("occluded")
            )
            if occluded_count > 10:
                anomalies["highly_occluded"].append(name)

            # Check scene attributes
            if attributes.get("timeofday") == "night":
                anomalies["night_scenes"].append(name)

            if attributes.get("weather") in ["rainy", "snowy", "foggy"]:
                anomalies["adverse_weather"].append(name)

            # Check bbox sizes
            for label in detection_labels:
                box2d = label.get("box2d", {})
                if box2d:
                    width = box2d.get("x2", 0) - box2d.get("x1", 0)
                    height = box2d.get("y2", 0) - box2d.get("y1", 0)
                    area = width * height

                    # Small objects (< 1% of image area, assuming 1280x720)
                    if area < (1280 * 720 * 0.01):
                        if name not in anomalies["extremely_small_objects"]:
                            anomalies["extremely_small_objects"].append(name)

                    # Large objects (> 50% of image area)
                    if area > (1280 * 720 * 0.5):
                        if name not in anomalies["extremely_large_objects"]:
                            anomalies["extremely_large_objects"].append(name)

        return anomalies


def compare_splits(
    train_parser: BDD100KParser, val_parser: BDD100KParser
) -> Dict:
    """Compare train and validation splits for distribution differences.

    Args:
        train_parser: Parser for training set
        val_parser: Parser for validation set

    Returns:
        Dictionary with comparison statistics
    """
    train_dist = train_parser.get_class_distribution()
    val_dist = val_parser.get_class_distribution()

    train_total = sum(train_dist.values())
    val_total = sum(val_dist.values())

    comparison = {"class_distributions": {}}

    for cls in DETECTION_CLASSES:
        train_count = train_dist.get(cls, 0)
        val_count = val_dist.get(cls, 0)

        train_pct = (train_count / train_total * 100) if train_total > 0 else 0
        val_pct = (val_count / val_total * 100) if val_total > 0 else 0

        comparison["class_distributions"][cls] = {
            "train_count": train_count,
            "val_count": val_count,
            "train_percentage": train_pct,
            "val_percentage": val_pct,
            "difference": abs(train_pct - val_pct),
        }

    return comparison
