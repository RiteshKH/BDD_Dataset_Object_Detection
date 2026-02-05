"""Model architecture selection and loading for BDD100K object detection.

This module provides functions to load and configure detection models,
with focus on Faster R-CNN with ResNet-50 FPN backbone.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_faster_rcnn_model(
    num_classes: int,
    pretrained: bool = True,
    pretrained_backbone: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
) -> FasterRCNN:
    """Load Faster R-CNN model with ResNet-50 FPN backbone.

    This is the recommended model for BDD100K object detection due to:
    - Strong performance on multi-scale objects
    - FPN handles small objects (traffic lights/signs) well
    - Good balance between speed and accuracy
    - Well-established architecture with extensive research

    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to load pretrained weights (COCO)
        pretrained_backbone: Whether to use pretrained ResNet backbone
        min_size: Minimum image size for resizing
        max_size: Maximum image size for resizing

    Returns:
        Faster R-CNN model ready for training/inference
    """
    logger.info(
        f"Loading Faster R-CNN with ResNet-50 FPN: "
        f"num_classes={num_classes}, pretrained={pretrained}"
    )

    # Load pretrained model
    if pretrained:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=None,
            weights_backbone=(
                "DEFAULT" if pretrained_backbone else None
            ),
            min_size=min_size,
            max_size=max_size,
        )

    # Replace the box predictor head for custom number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    logger.info(f"Model loaded with {num_classes} classes")
    return model


def get_retinanet_model(
    num_classes: int,
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
) -> nn.Module:
    """Load RetinaNet model with ResNet-50 FPN backbone.

    Alternative to Faster R-CNN, useful for:
    - One-stage detection (faster inference)
    - Focal loss handles class imbalance well
    - Good for dense object scenarios

    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to load pretrained weights
        min_size: Minimum image size for resizing
        max_size: Maximum image size for resizing

    Returns:
        RetinaNet model ready for training/inference
    """
    logger.info(
        f"Loading RetinaNet with ResNet-50 FPN: "
        f"num_classes={num_classes}, pretrained={pretrained}"
    )

    if pretrained:
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights="DEFAULT",
            min_size=min_size,
            max_size=max_size,
        )
    else:
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None,
            min_size=min_size,
            max_size=max_size,
        )

    # Replace classification head
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes

    # Reinitialize the classification head layers
    in_channels = model.head.classification_head.conv[0].in_channels
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
    )

    logger.info(f"RetinaNet loaded with {num_classes} classes")
    return model


def load_checkpoint(
    model: nn.Module, checkpoint_path: str, device: torch.device
) -> nn.Module:
    """Load model weights from checkpoint.

    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Model with loaded weights
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    logger.info("Checkpoint loaded successfully")
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str,
) -> None:
    """Save model checkpoint.

    Args:
        model: Model instance
        optimizer: Optimizer instance
        epoch: Current epoch number
        loss: Current loss value
        checkpoint_path: Path to save checkpoint
    """
    logger.info(f"Saving checkpoint to {checkpoint_path}")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: epoch {epoch}, loss {loss:.4f}")


def get_model(
    model_name: str = "faster_rcnn",
    num_classes: int = 11,  # 10 classes + background
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to get detection model by name.

    Args:
        model_name: Name of the model ('faster_rcnn' or 'retinanet')
        num_classes: Number of classes including background
        pretrained: Whether to load pretrained weights
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Detection model

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name.lower() == "faster_rcnn":
        return get_faster_rcnn_model(num_classes, pretrained, **kwargs)
    elif model_name.lower() == "retinanet":
        return get_retinanet_model(num_classes, pretrained, **kwargs)
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Supported models: 'faster_rcnn', 'retinanet'"
        )


def count_parameters(model: nn.Module) -> tuple:
    """Count total and trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(
        f"Model parameters: {total_params:,} total, "
        f"{trainable_params:,} trainable"
    )

    return total_params, trainable_params


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Faster R-CNN
    model = get_faster_rcnn_model(num_classes=11, pretrained=True)
    model = model.to(device)

    count_parameters(model)

    # Test forward pass
    model.eval()
    dummy_input = [torch.rand(3, 800, 800).to(device)]

    with torch.no_grad():
        output = model(dummy_input)

    logger.info(f"Test forward pass successful: {len(output)} predictions")
