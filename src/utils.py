"""
Utility functions: reproducibility, device management, logging
"""

import random
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, Any


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior (may slow down training slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seed set to {seed} (reproducible mode)")


def get_device() -> torch.device:
    """
    Get the default device (GPU if available, otherwise CPU).
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        return device
    else:
        device = torch.device("cpu")
        print("Warning: GPU not available. Training will be slow.")
        return device


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics (accuracy, F1, AUC, etc.)
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✓ Metrics saved to {output_path}")


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        metrics_path: Path to metrics JSON file
    
    Returns:
        Dictionary of metrics
    """
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def create_directories(base_path: str) -> Dict[str, Path]:
    """
    Create and return standard project directories.
    
    Args:
        base_path: Base project path
    
    Returns:
        Dictionary of directory paths
    """
    base = Path(base_path)
    dirs = {
        'data_raw': base / 'data' / 'raw',
        'results': base / 'results',
        'weights': base / 'results' / 'model_weights',
        'confusion': base / 'results' / 'confusion_matrices',
        'grad_cam': base / 'results' / 'grad_cam',
        'report': base / 'report',
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def compute_class_weights(labels: np.ndarray, num_classes: int = 3) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Weight = total_samples / (num_classes * samples_in_class)
    This gives equal total weight to each class.
    
    Args:
        labels: Array of class labels (0, 1, 2, ...)
        num_classes: Number of classes
    
    Returns:
        torch.Tensor of shape (num_classes,) with class weights
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    
    total_samples = len(labels)
    weights = np.zeros(num_classes)
    
    for class_idx in range(num_classes):
        class_count = np.sum(labels == class_idx)
        if class_count > 0:
            # Inverse frequency weighting
            weight = total_samples / (num_classes * class_count)
            weights[class_idx] = weight
        else:
            # Class not present in data
            weights[class_idx] = 1.0
    
    # Normalize so that mean weight = 1
    weights = weights / weights.mean()
    
    print(f"✓ Class weights computed:")
    for idx, w in enumerate(weights):
        print(f"  Class {idx}: {w:.4f}")
    
    return torch.tensor(weights, dtype=torch.float32)


class Config:
    """
    Configuration class for hyperparameters and settings.
    
    Usage:
        config = Config()
        config.batch_size = 32
        config.num_epochs = 80
    """
    
    # Dataset
    NUM_CLASSES = 3
    CLASS_NAMES = ['NILM', 'LSIL', 'HSIL']
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    
    # Input preprocessing
    INPUT_SIZE_CNN = 384
    INPUT_SIZE_TRANSFORMER = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Training (Phase 1: Warmup)
    LR_PHASE1 = 1e-3
    EPOCHS_PHASE1 = 5
    BATCH_SIZE_PHASE1 = 64
    
    # Training (Phase 2: Fine-tuning)
    LR_PHASE2 = 1e-5
    EPOCHS_PHASE2 = 75
    BATCH_SIZE_PHASE2 = 32
    
    # Optimization
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 10
    EVAL_FREQ = 1  # evaluate every N epochs
    
    # Reproducibility
    SEED = 42
    
    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    def __repr__(self):
        """Pretty print configuration (FIX: now shows class attributes correctly)."""
        lines = [f"\n{'='*60}\nConfiguration\n{'='*60}"]
        
        # Iterate over class attributes (not just instance dict)
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                value = getattr(self, key)
                lines.append(f"{key:.<40} {value}")
        
        lines.append(f"{'='*60}\n")
        return '\n'.join(lines)


if __name__ == "__main__":
    # Test utilities
    set_seed(42)
    device = get_device()
    config = Config()
    print(config)
    
    # Test directory creation
    dirs = create_directories(".")
    print(f"\nCreated directories: {dirs}")
