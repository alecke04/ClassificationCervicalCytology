"""
Data loading, preprocessing, and augmentation pipeline
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import re
import json


class CytologyDataset(Dataset):
    """
    Custom PyTorch Dataset for cervical cytology images.
    
    Args:
        image_paths: List of image file paths
        labels: List of class labels (0=NILM, 1=LSIL, 2=HSIL)
        transform: Albumentations transform pipeline
        input_size: Target input size (384 for CNN, 224 for Swin)
    """
    
    def __init__(
        self,
        image_paths: list,
        labels: list,
        transform: Optional[A.Compose] = None,
        input_size: int = 384,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.input_size = input_size
        
        assert len(image_paths) == len(labels), "Image paths and labels must have same length"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (image tensor, label)
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(str(image_path))
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            # Fallback: convert to tensor without augmentation
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)
        
        return image, label


def detect_patient_groups(image_paths: list) -> Optional[dict]:
    """
    Attempt to detect patient/slide IDs from filenames to identify leakage risk.
    
    Common patterns:
    - patient_001_slide_01.jpg
    - P001_S01.jpg
    - 001_01.jpg
    - patient001_slide01_frame1.jpg
    
    Args:
        image_paths: List of image file paths
    
    Returns:
        Dict mapping patient_id -> list of image paths, or None if no pattern detected
    """
    patterns = [
        (r'patient_?(\d+)', 'patient_id'),  # patient_001 or patient001
        (r'p(\d+)', 'patient_id'),           # p001
        (r'^(\d{3,})', 'patient_id'),        # 001_...
        (r'[Pp]atient[_-]?([A-Z0-9]+)', 'patient_id'),  # Patient_ABC
    ]
    
    patient_groups = {}
    
    for path in image_paths:
        filename = Path(path).stem  # Get filename without extension
        found_id = False
        
        for pattern, id_type in patterns:
            match = re.search(pattern, filename)
            if match:
                patient_id = match.group(1)
                if patient_id not in patient_groups:
                    patient_groups[patient_id] = []
                patient_groups[patient_id].append(path)
                found_id = True
                break
        
        if not found_id:
            # No patient ID found in filename
            return None
    
    # Only return if we found consistent grouping (>1 image per patient for some patients)
    if any(len(v) > 1 for v in patient_groups.values()):
        return patient_groups
    
    return None


def get_augmentation_transform(input_size: int, augment: bool = True) -> A.Compose:
    """
    Get augmentation pipeline (albumentations).
    
    Args:
        input_size: Target image size (384 or 224)
        augment: If True, apply augmentation; else only normalization
    
    Returns:
        Albumentations Compose pipeline
    """
    if augment:
        transform_list = [
            A.Resize(input_size, input_size),
            # Domain-aware augmentation for medical imaging
            A.HorizontalFlip(p=0.5),
            # Note: VerticalFlip removed - questionable biological validity for cytology
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.05,
                p=0.5
            ),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    else:
        # No augmentation for val/test
        transform_list = [
            A.Resize(input_size, input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    
    return A.Compose(transform_list)


def load_data_split(
    data_dir: str,
    input_size: int = 384,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and split dataset into train/val/test DataLoaders.
    
    Args:
        data_dir: Path to data directory containing NILM/, LSIL/, HSIL/ folders
        input_size: Target input size (384 for CNN, 224 for Swin)
        test_size: Fraction of data for test set (default 0.15)
        val_size: Fraction of data for validation set (default 0.15)
        random_state: Random seed for reproducibility
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    data_dir = Path(data_dir)
    class_names = ['NILM', 'LSIL', 'HSIL']
    
    # Load image paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        
        if not class_dir.exists():
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        
        # Get all image files (jpg, png)
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"No images found in {class_dir}")
        
        print(f"Found {len(image_files)} images in {class_name}")
        
        image_paths.extend(image_files)
        labels.extend([class_idx] * len(image_files))
    
    # Stratified train/val/test split
    image_paths = np.array(image_paths)
    labels = np.array(labels)
    
    # Check for patient-level data leakage risk
    print("\n" + "="*60)
    print("DATA LEAKAGE DETECTION")
    print("="*60)
    patient_groups = detect_patient_groups(image_paths.tolist())
    if patient_groups:
        unique_patients = len(patient_groups)
        images_in_groups = sum(len(v) for v in patient_groups.values() if len(v) > 1)
        print(f"⚠️  POTENTIAL DATA LEAKAGE DETECTED:")
        print(f"   - Found {unique_patients} unique patients in dataset")
        print(f"   - {images_in_groups} images from patients with multiple images")
        print(f"   - Multiple images from same patient may appear in train/test!")
        print(f"   - Consider using stratified split by patient group instead")
        print(f"   - Current implementation splits at IMAGE level (not patient level)")
    else:
        print(f"✓ No patient-level grouping pattern detected")
        print(f"  (Images appear to be from different patients)")
    print("="*60 + "\n")
    
    # First split: train (70%) vs temp (30%)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=(test_size + val_size),
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: temp into val (50%) and test (50%)
    val_size_relative = val_size / (test_size + val_size)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=1.0 - val_size_relative,
        stratify=temp_labels,
        random_state=random_state
    )
    
    # Convert back to lists
    train_paths = [str(p) for p in train_paths]
    val_paths = [str(p) for p in val_paths]
    test_paths = [str(p) for p in test_paths]
    train_labels = train_labels.tolist()
    val_labels = val_labels.tolist()
    test_labels = test_labels.tolist()
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_paths)} images")
    print(f"  Val:   {len(val_paths)} images")
    print(f"  Test:  {len(test_paths)} images")
    
    # Create datasets with transforms
    train_transform = get_augmentation_transform(input_size, augment=True)
    val_transform = get_augmentation_transform(input_size, augment=False)
    test_transform = get_augmentation_transform(input_size, augment=False)
    
    train_dataset = CytologyDataset(train_paths, train_labels, train_transform, input_size)
    val_dataset = CytologyDataset(val_paths, val_labels, val_transform, input_size)
    test_dataset = CytologyDataset(test_paths, test_labels, test_transform, input_size)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train loader: {len(train_loader)} batches (batch size {batch_size})")
    print(f"  Val loader:   {len(val_loader)} batches")
    print(f"  Test loader:  {len(test_loader)} batches")
    
    # Return split metadata along with loaders
    split_metadata = {
        'train_paths': train_paths,
        'val_paths': val_paths,
        'test_paths': test_paths,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'test_labels': test_labels,
    }
    
    return train_loader, val_loader, test_loader, split_metadata


def save_split_metadata(split_metadata: Dict, output_dir: str) -> None:
    """
    Save train/val/test split file paths and labels to JSON.
    
    This allows exact reproduction of the same split during evaluation,
    preventing issues if split logic changes.
    
    Args:
        split_metadata: Dictionary with 'train_paths', 'val_paths', 'test_paths', etc.
        output_dir: Directory to save metadata JSON
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = output_dir / 'split_metadata.json'
    
    with open(metadata_path, 'w') as f:
        json.dump(split_metadata, f, indent=2)
    
    print(f"✓ Split metadata saved to {metadata_path}")


def load_split_metadata(output_dir: str) -> Dict:
    """
    Load train/val/test split metadata from JSON.
    
    Args:
        output_dir: Directory containing split_metadata.json
    
    Returns:
        Dictionary with split paths and labels
    """
    metadata_path = Path(output_dir) / 'split_metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Split metadata not found: {metadata_path}\n"
            f"Make sure to train first (which saves the split), then evaluate."
        )
    
    with open(metadata_path, 'r') as f:
        split_metadata = json.load(f)
    
    print(f"✓ Split metadata loaded from {metadata_path}")
    return split_metadata


def load_test_from_metadata(
    split_metadata: Dict,
    input_size: int = 384,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create test DataLoader ONLY from saved split metadata (for efficient evaluation).
    
    This is more efficient than load_data_from_metadata() when only test data is needed.
    Skips building train/val datasets and their augmentation transforms.
    
    Args:
        split_metadata: Dictionary from load_split_metadata()
        input_size: Target input size (384 for CNN, 224 for Swin)
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
    
    Returns:
        DataLoader for test set only
    """
    # Extract test metadata only
    test_paths = split_metadata['test_paths']
    test_labels = split_metadata['test_labels']
    
    # Create test transform (no augmentation)
    test_transform = get_augmentation_transform(input_size, augment=False)
    
    # Create test dataset
    test_dataset = CytologyDataset(test_paths, test_labels, test_transform, input_size)
    
    # Create test DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"\nTest loader from metadata:")
    print(f"  Test loader: {len(test_loader)} batches (batch size {batch_size})")
    
    return test_loader


def load_data_from_metadata(
    split_metadata: Dict,
    input_size: int = 384,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders from saved split metadata (for evaluation).
    
    This ensures evaluation uses the EXACT same split as training.
    
    Args:
        split_metadata: Dictionary from load_split_metadata()
        input_size: Target input size (384 for CNN, 224 for Swin)
        batch_size: Batch size for DataLoader
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Extract metadata
    train_paths = split_metadata['train_paths']
    val_paths = split_metadata['val_paths']
    test_paths = split_metadata['test_paths']
    train_labels = split_metadata['train_labels']
    val_labels = split_metadata['val_labels']
    test_labels = split_metadata['test_labels']
    
    # Create transforms
    train_transform = get_augmentation_transform(input_size, augment=True)
    val_transform = get_augmentation_transform(input_size, augment=False)
    test_transform = get_augmentation_transform(input_size, augment=False)
    
    # Create datasets
    train_dataset = CytologyDataset(train_paths, train_labels, train_transform, input_size)
    val_dataset = CytologyDataset(val_paths, val_labels, val_transform, input_size)
    test_dataset = CytologyDataset(test_paths, test_labels, test_transform, input_size)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"\nDataLoaders from metadata:")
    print(f"  Train loader: {len(train_loader)} batches (batch size {batch_size})")
    print(f"  Val loader:   {len(val_loader)} batches")
    print(f"  Test loader:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing data loading pipeline...")
    
    # Example: load from data/raw directory
    # train_loader, val_loader, test_loader = load_data_split(
    #     data_dir="data/raw",
    #     input_size=384,
    #     batch_size=32,
    # )
    
    print("✓ Data loading pipeline ready")
