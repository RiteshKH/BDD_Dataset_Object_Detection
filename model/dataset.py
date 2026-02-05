"""BDD100K Dataset implementation for PyTorch object detection.

Loads images and per-image JSON annotations for the 10 BDD100K detection
classes. Handles data loading, transforms, and collation for detection models.

Follows PEP8 standards and project coding instructions.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# BDD100K detection classes (10 classes)
BDD100K_CLASSES = [
    "car",
    "truck",
    "bus",
    "person",
    "bike",
    "motor",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
]

# Class name to index mapping (0 is reserved for background)
CLASS_TO_IDX = {cls_name: idx + 1 for idx, cls_name in enumerate(BDD100K_CLASSES)}


class BDD100KDataset(Dataset):
    """PyTorch Dataset for BDD100K object detection.

    Loads images and per-image JSON annotations, applies transforms,
    and returns data in format compatible with torchvision detection models.

    The BDD100K format has structure:
    {
        "name": "image_name",
        "frames": [{"objects": [...]}],
        "attributes": {"weather": ..., "scene": ..., "timeofday": ...}
    }

    Args:
        image_dir: Directory containing images
        annotation_dir: Directory containing per-image JSON annotation files
        transform: Optional transform to apply to images
        train: Whether this is training data (affects augmentation)
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        transform: Optional[Callable] = None,
        train: bool = True,
    ):
        """Initialize BDD100K dataset.

        Args:
            image_dir: Path to directory with images
            annotation_dir: Path to directory with per-image JSON files
            transform: Optional transforms to apply
            train: Whether this is training data
        """
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.transform = transform
        self.train = train

        # Validate directories exist
        if not self.image_dir.exists():
            raise ValueError(f"Image directory not found: {self.image_dir}")
        if not self.annotation_dir.exists():
            raise ValueError(
                f"Annotation directory not found: {self.annotation_dir}"
            )

        # Find all image files
        self.image_files = self._find_image_files()

        if len(self.image_files) == 0:
            logger.error(f"No images found in {self.image_dir}")
            logger.error("Checked extensions: .jpg, .png")
            raise ValueError(
                f"No valid images found in {self.image_dir}. "
                f"Please check the directory path and image format."
            )

        logger.info(
            f"Initialized BDD100K dataset with {len(self.image_files)} images"
        )
        logger.info(f"Image directory: {self.image_dir}")
        logger.info(f"Annotation directory: {self.annotation_dir}")

    def _find_image_files(self) -> List[Path]:
        """Find all valid image files that have corresponding annotations.

        Returns:
            List of image file paths with annotations
        """
        valid_extensions = {".jpg", ".jpeg", ".png"}
        image_files = []

        # Get all image files
        all_images = [
            f
            for f in self.image_dir.iterdir()
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

        # Check which ones have corresponding annotations
        for img_path in all_images:
            # BDD100K annotations have same name as image but .json extension
            ann_path = self.annotation_dir / f"{img_path.stem}.json"

            if ann_path.exists():
                image_files.append(img_path)
            else:
                logger.debug(
                    f"Skipping {img_path.name}: no annotation file found"
                )

        if len(image_files) == 0:
            logger.warning(
                f"No image-annotation pairs found. "
                f"Checked {len(all_images)} images in {self.image_dir}"
            )

        return sorted(image_files)

    def __len__(self) -> int:
        """Return number of samples in dataset.

        Returns:
            Number of images in dataset
        """
        return len(self.image_files)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get image and target at given index.

        Args:
            idx: Index of sample to retrieve

        Returns:
            Tuple of (image_tensor, target_dict) where target_dict contains:
                - boxes: FloatTensor[N, 4] in [x1, y1, x2, y2] format
                - labels: Int64Tensor[N] with class indices (1-based)
                - image_id: Int64Tensor[1] with image ID
                - area: FloatTensor[N] with box areas
                - iscrowd: UInt8Tensor[N] (all zeros for BDD100K)
        """
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_path = self.annotation_dir / f"{img_path.stem}.json"
        with open(ann_path, "r") as f:
            ann_data = json.load(f)

        # Parse annotations
        boxes, labels = self._parse_annotations(ann_data)

        # Handle empty annotations (images with no detection objects)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.uint8)
        else:
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (
                boxes[:, 2] - boxes[:, 0]
            )
            iscrowd = torch.zeros((len(boxes),), dtype=torch.uint8)

        image_id = torch.tensor([idx])

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            # Default transform: convert to tensor
            image = T.ToTensor()(image)

        return image, target

    def _parse_annotations(
        self, ann_data: Dict[str, Any]
    ) -> Tuple[List[List[float]], List[int]]:
        """Parse BDD100K annotation JSON to extract bounding boxes and labels.

        BDD100K format structure:
        {
            "name": "image_name",
            "frames": [
                {
                    "timestamp": 10000,
                    "objects": [
                        {
                            "category": "car",
                            "id": 0,
                            "attributes": {...},
                            "box2d": {"x1": ..., "y1": ..., "x2": ..., "y2": ...}
                        },
                        ...
                    ]
                }
            ],
            "attributes": {...}
        }

        Args:
            ann_data: Annotation dictionary loaded from JSON

        Returns:
            Tuple of (boxes, labels) where:
                - boxes: List of [x1, y1, x2, y2] coordinates
                - labels: List of class indices (1-based)
        """
        boxes = []
        labels = []

        # BDD100K format: navigate to frames[0].objects
        if "frames" not in ann_data or len(ann_data["frames"]) == 0:
            logger.warning(f"No 'frames' key in annotation: {ann_data.get('name')}")
            return boxes, labels

        # Get objects from first frame (BDD100K uses single frame per image)
        frame = ann_data["frames"][0]
        if "objects" not in frame:
            logger.warning(
                f"No 'objects' key in frame for image: {ann_data.get('name')}"
            )
            return boxes, labels

        for obj in frame["objects"]:
            # Only process objects with box2d (bounding boxes)
            # Skip segmentation annotations (poly2d) and other types
            if "box2d" not in obj:
                continue

            category = obj["category"]

            # Only include the 10 detection classes
            if category not in CLASS_TO_IDX:
                continue

            box2d = obj["box2d"]

            # Extract coordinates
            x1 = float(box2d["x1"])
            y1 = float(box2d["y1"])
            x2 = float(box2d["x2"])
            y2 = float(box2d["y2"])

            # Validate box coordinates
            if x2 <= x1 or y2 <= y1:
                logger.debug(
                    f"Invalid box coordinates for {category}: {box2d}"
                )
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_IDX[category])

        return boxes, labels


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict]]
) -> Tuple[List, List]:
    """Collate function for DataLoader.

    Detection models expect list of images and list of targets rather than
    batched tensors.

    Args:
        batch: List of (image, target) tuples

    Returns:
        Tuple of (images_list, targets_list)
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def get_transform(train: bool) -> Callable:
    """Get data transforms for training or validation.

    Args:
        train: Whether to apply training augmentations

    Returns:
        Transform function
    """
    transforms = []
    transforms.append(T.ToTensor())

    # Add data augmentation for training
    if train:
        # Color jitter for varying lighting conditions
        transforms.append(
            T.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        )

    return T.Compose(transforms)


def create_dataloader(
    image_dir: str,
    annotation_dir: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    train: bool = True,
) -> DataLoader:
    """Create DataLoader for BDD100K dataset.

    Args:
        image_dir: Directory containing images
        annotation_dir: Directory containing per-image JSON files
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        train: Whether this is training data (affects augmentation)

    Returns:
        PyTorch DataLoader configured for object detection
    """
    # Create dataset
    transform = get_transform(train=train)
    dataset = BDD100KDataset(
        image_dir=image_dir,
        annotation_dir=annotation_dir,
        transform=transform,
        train=train,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    logger.info(
        f"Created dataloader: {len(dataset)} samples, "
        f"batch_size={batch_size}, "
        f"num_workers={num_workers}"
    )

    return dataloader


if __name__ == "__main__":
    """Test dataset implementation."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dataset.py <image_dir> <annotation_dir>")
        sys.exit(1)

    img_dir = sys.argv[1]
    ann_dir = sys.argv[2]

    # Test dataset
    dataset = BDD100KDataset(
        image_dir=img_dir, annotation_dir=ann_dir, train=True
    )

    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        # Test first sample
        img, target = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Number of objects: {len(target['boxes'])}")
        print(f"Classes: {target['labels']}")
        if len(target["boxes"]) > 0:
            print(f"Sample boxes:\n{target['boxes'][:3]}")

        # Test sample with no detection objects (if exists)
        for i in range(min(10, len(dataset))):
            _, tgt = dataset[i]
            if len(tgt["boxes"]) == 0:
                print(f"\nImage {i} has no detection objects (valid case)")
                break
