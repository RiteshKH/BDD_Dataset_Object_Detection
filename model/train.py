"""Training pipeline for BDD100K object detection.

Implements complete training loop with:
- Model training for one or more epochs
- GPU/Multi-GPU support with automatic device detection
- Early stopping (within epoch and across epochs)
- Learning rate scheduling
- Checkpoint saving
- Validation evaluation
- Loss tracking and logging
- Progress visualization with tqdm

Follows PEP8 standards and project coding instructions.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataloader
from model_architecture import get_model, save_checkpoint, count_parameters

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping handler to stop training when loss converges.

    Monitors loss and stops when improvement is below threshold for
    a certain number of consecutive checks.

    Args:
        patience: Number of checks with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        check_frequency: How often to check (in batches)
        verbose: Whether to log stopping decisions
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        check_frequency: int = 50,
        verbose: bool = True,
    ):
        """Initialize early stopping handler."""
        self.patience = patience
        self.min_delta = min_delta
        self.check_frequency = check_frequency
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_loss: float) -> bool:
        """Check if training should stop.

        Args:
            current_loss: Current average loss value

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = current_loss
            return False

        if current_loss < self.best_score - self.min_delta:
            if self.verbose:
                logger.info(
                    f"Loss improved: {self.best_score:.6f} → {current_loss:.6f}"
                )
            self.best_score = current_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"No improvement ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered! Best loss: {self.best_score:.6f}"
                    )
                return True

        return False

    def reset(self):
        """Reset early stopping state for new epoch."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def get_device() -> torch.device:
    """Get available device for training (GPU if available, else CPU).

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
        logger.info(
            f"GPU Memory: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        logger.info("GPU not available, using CPU")

    return device


def setup_model_for_gpu(
    model: torch.nn.Module,
    device: torch.device,
    use_data_parallel: bool = True,
) -> torch.nn.Module:
    """Setup model for GPU training with optional multi-GPU support.

    Args:
        model: PyTorch model to setup
        device: Target device (cuda or cpu)
        use_data_parallel: Whether to use DataParallel for multi-GPU

    Returns:
        Model configured for GPU training
    """
    model = model.to(device)

    if use_data_parallel and torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    return model


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 50,
    early_stopping: Optional[EarlyStopping] = None,
) -> Dict[str, float]:
    """Train model for one epoch on subset of data.

    Args:
        model: Detection model (already on correct device)
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on (cuda or cpu)
        epoch: Current epoch number
        print_freq: Frequency of logging (in iterations)
        early_stopping: Optional early stopping handler

    Returns:
        Dictionary with average losses and early_stopped flag
    """
    model.train()

    total_loss = 0.0
    loss_dict_reduced = {}
    num_batches = 0
    early_stopped = False

    logger.info(f"Starting epoch {epoch}")
    start_time = time.time()

    progress_bar = tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        desc=f"Epoch {epoch}",
        unit="batch",
        ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for i, (images, targets) in progress_bar:
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if not torch.isfinite(losses):
                logger.warning(f"Loss is {losses.item()}, skipping batch")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += losses.item()
            num_batches += 1

            for k, v in loss_dict.items():
                if k not in loss_dict_reduced:
                    loss_dict_reduced[k] = 0.0
                loss_dict_reduced[k] += v.item()

            avg_loss = total_loss / num_batches
            progress_bar.set_postfix(
                {"loss": f"{losses.item():.4f}", "avg_loss": f"{avg_loss:.4f}"}
            )

            if early_stopping is not None and (i + 1) % early_stopping.check_frequency == 0:
                if early_stopping(avg_loss):
                    logger.info(f"Early stopping at batch {i+1}/{len(dataloader)}")
                    early_stopped = True
                    break

            if (i + 1) % print_freq == 0:
                logger.info(
                    f"Epoch [{epoch}] Batch [{i+1}/{len(dataloader)}] "
                    f"Loss: {losses.item():.4f} (avg: {avg_loss:.4f})"
                )

            if device.type == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory at batch {i+1}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    progress_bar.close()

    if num_batches > 0:
        avg_total_loss = total_loss / num_batches
        for k in loss_dict_reduced:
            loss_dict_reduced[k] /= num_batches
    else:
        avg_total_loss = 0.0

    epoch_time = time.time() - start_time
    status = "stopped early" if early_stopped else "completed"
    logger.info(
        f"Epoch {epoch} {status} in {epoch_time:.2f}s - Average loss: {avg_total_loss:.4f}"
    )
    logger.info(f"Loss components: {loss_dict_reduced}")

    return {"total_loss": avg_total_loss, "early_stopped": early_stopped, **loss_dict_reduced}


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on validation set.

    Args:
        model: Detection model (already on correct device)
        dataloader: Validation data loader
        device: Device to evaluate on (cuda or cpu)

    Returns:
        Dictionary with evaluation metrics
    """
    model.train()

    total_loss = 0.0
    num_batches = 0

    logger.info("Starting evaluation")
    start_time = time.time()

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        unit="batch",
        ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for images, targets in progress_bar:
        try:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isfinite(losses):
                total_loss += losses.item()
                num_batches += 1

            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            progress_bar.set_postfix({"val_loss": f"{avg_loss:.4f}"})

            if device.type == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("GPU out of memory during validation")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                raise e

    progress_bar.close()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    eval_time = time.time() - start_time
    logger.info(f"Evaluation completed in {eval_time:.2f}s - Average loss: {avg_loss:.4f}")

    return {"val_loss": avg_loss}


def train(
    model_name: str,
    train_img_dir: str,
    train_ann_dir: str,
    val_img_dir: Optional[str],
    val_ann_dir: Optional[str],
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    output_dir: str,
    device: torch.device,
    subset_size: Optional[int] = None,
    use_data_parallel: bool = True,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.001,
) -> None:
    """Complete training pipeline with GPU support and optional early stopping.

    Args:
        model_name: Name of model to use
        train_img_dir: Training images directory
        train_ann_dir: Training annotations directory
        val_img_dir: Validation images directory (optional)
        val_ann_dir: Validation annotations directory (optional)
        num_epochs: Number of epochs to train
        batch_size: Batch size
        learning_rate: Initial learning rate
        num_workers: Number of data loading workers
        output_dir: Directory to save checkpoints
        device: Device to train on (cuda or cpu)
        subset_size: Optional number of samples for quick testing
        use_data_parallel: Whether to use DataParallel for multi-GPU
        early_stopping_patience: Patience for early stopping (None to disable)
        early_stopping_min_delta: Minimum improvement threshold
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating training dataloader...")
    train_loader = create_dataloader(
        image_dir=train_img_dir,
        annotation_dir=train_ann_dir,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        train=True,
    )

    if subset_size is not None:
        logger.info(f"Using subset of {subset_size} samples for training")
        train_loader.dataset.image_files = train_loader.dataset.image_files[:subset_size]

    val_loader = None
    if val_img_dir and val_ann_dir:
        logger.info("Creating validation dataloader...")
        val_loader = create_dataloader(
            image_dir=val_img_dir,
            annotation_dir=val_ann_dir,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            train=False,
        )

    logger.info(f"Loading model: {model_name}")
    model = get_model(model_name=model_name, num_classes=11, pretrained=True)
    model = setup_model_for_gpu(model, device, use_data_parallel)
    count_parameters(model)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    within_epoch_stopper = None
    epoch_stopper = None

    if early_stopping_patience is not None:
        within_epoch_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            check_frequency=50,
            verbose=True,
        )
        epoch_stopper = EarlyStopping(
            patience=early_stopping_patience * 2,
            min_delta=early_stopping_min_delta,
            check_frequency=1,
            verbose=True,
        )
        logger.info(
            f"Early stopping enabled: patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta}"
        )

    logger.info(f"Starting training for {num_epochs} epochs on {device}")
    best_val_loss = float("inf")

    epoch_progress = tqdm(
        range(1, num_epochs + 1), desc="Training Progress", unit="epoch", ncols=100, position=0
    )

    for epoch in epoch_progress:
        if within_epoch_stopper is not None:
            within_epoch_stopper.reset()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, epoch, print_freq=50, early_stopping=within_epoch_stopper
        )

        if train_metrics.get("early_stopped", False):
            logger.info("Training stopped early within epoch")
            if val_loader is not None:
                val_metrics = evaluate(model, val_loader, device)
                logger.info(f"Final validation loss: {val_metrics['val_loss']:.4f}")
            break

        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)

            if epoch_stopper is not None and epoch_stopper(val_metrics["val_loss"]):
                logger.info("Training stopped early across epochs")
                break

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                checkpoint_path = output_path / "best_model.pth"
                model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                save_checkpoint(model_to_save, optimizer, epoch, val_metrics["val_loss"], checkpoint_path)
                logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")

        epoch_metrics = {"train_loss": f"{train_metrics['total_loss']:.4f}"}
        if val_metrics:
            epoch_metrics["val_loss"] = f"{val_metrics['val_loss']:.4f}"
        epoch_progress.set_postfix(epoch_metrics)

        checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pth"
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        save_checkpoint(model_to_save, optimizer, epoch, train_metrics["total_loss"], checkpoint_path)

        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Learning rate: {current_lr}")

    epoch_progress.close()
    logger.info("Training completed!")
    logger.info(f"Checkpoints saved to {output_dir}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(description="Train BDD100K object detection model")

    parser.add_argument("--train-img-dir", type=str, required=True, help="Training images directory")
    parser.add_argument("--train-ann-dir", type=str, required=True, help="Training annotations directory")
    parser.add_argument("--val-img-dir", type=str, help="Validation images directory")
    parser.add_argument("--val-ann-dir", type=str, help="Validation annotations directory")
    parser.add_argument("--model", type=str, default="faster_rcnn", choices=["faster_rcnn", "retinanet"])
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--subset-size", type=int, default=None, help="Use subset of data")
    parser.add_argument("--no-data-parallel", action="store_true", help="Disable DataParallel")
    parser.add_argument("--early-stopping-patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.001, help="Min improvement")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Output directory")

    args = parser.parse_args()

    device = get_device()

    train(
        model_name=args.model,
        train_img_dir=args.train_img_dir,
        train_ann_dir=args.train_ann_dir,
        val_img_dir=args.val_img_dir,
        val_ann_dir=args.val_ann_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        device=device,
        subset_size=args.subset_size,
        use_data_parallel=args.no_data_parallel,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


if __name__ == "__main__":
    main()