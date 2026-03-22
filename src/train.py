"""
Training loop with two-phase strategy (warmup + fine-tuning)
Handles validation, early stopping, and model checkpointing

CRITICAL: Uses REAL macro F1-score for early stopping and model selection,
not fake/placeholder F1. This is essential for medical 3-class problems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, accuracy_score

from src.utils import set_seed, get_device, Config, compute_class_weights
from src.models import get_model, unfreeze_all, print_model_info
from src.data import load_data_split, save_split_metadata


class Trainer:
    """
    Training manager for two-phase fine-tuning strategy.
    
    Phase 1: Warmup (freeze backbone, train head only)
    Phase 2: Fine-tuning (unfreeze all, careful tuning)
    """
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        output_dir: str = "results",
        config: Config = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: 'resnet50', 'efficientnet_b0', or 'swin_tiny'
            device: torch.device (cuda or cpu)
            output_dir: Directory to save results
            config: Config object with hyperparameters
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = Path(output_dir)
        self.config = config or Config()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = get_model(
            model_name=model_name,
            num_classes=self.config.NUM_CLASSES,
            pretrained=True,
            freeze_backbone_phase1=True,
        ).to(device)
        
        print_model_info(self.model, model_name)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize optimizer (will be updated in phase 2)
        self.optimizer = None
        self.scheduler = None
        
        # Tracking
        self.train_history = {
            'loss': [],
            'accuracy': [],
            'f1': [],
        }
        self.val_history = {
            'loss': [],
            'accuracy': [],
            'f1': [],
        }
    
    def set_class_weights(self, train_loader: DataLoader) -> None:
        """
        Compute and set class weights based on training data distribution.
        Handles class imbalance by giving higher weight to minority classes.
        
        Args:
            train_loader: Training DataLoader
        """
        # Extract all labels from training loader
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
        
        all_labels = np.array(all_labels)
        
        # Compute weights
        weights = compute_class_weights(all_labels, num_classes=self.config.NUM_CLASSES)
        
        # Update loss function with weights
        self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.device))
        print(f"✓ Weighted loss function initialized for class imbalance")
    
    def train_phase1(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Phase 1: Warmup training (freeze backbone, train head only).
        
        CRITICAL FIX: Only pass trainable params to optimizer (frozen params ignored otherwise).
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        
        Returns:
            Dictionary with phase 1 metrics
        """
        print(f"\n{'='*60}")
        print(f"PHASE 1: WARMUP (Head Training Only)")
        print(f"Learning Rate: {self.config.LR_PHASE1}")
        print(f"Epochs: {self.config.EPOCHS_PHASE1}")
        print(f"{'='*60}\n")
        
        # CRITICAL FIX: Only optimize trainable parameters (frozen ones are ignored anyway)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.LR_PHASE1,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,
            eta_min=1e-6,
        )
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS_PHASE1):
            # Training
            train_loss, train_acc, train_f1 = self._train_epoch(train_loader)
            
            # Validation (REAL F1, not fake)
            val_loss, val_acc, val_f1 = self._validate(val_loader)
            
            # Track history for later analysis
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['f1'].append(train_f1)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            self.val_history['f1'].append(val_f1)
            
            self.scheduler.step()
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config.EPOCHS_PHASE1}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, F1={train_f1:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}")
            
            # Early stopping based on REAL F1 (not fake accuracy)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self._save_checkpoint(epoch, "phase1")
                print(f"  ✓ New best F1: {val_f1:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'phase': 'phase1',
            'best_val_f1': best_val_f1,
        }
    
    def train_phase2(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Phase 2: Fine-tuning training (unfreeze all, careful tuning).
        
        CRITICAL FIX: Reload best Phase 1 checkpoint BEFORE unfreezing.
        This ensures Phase 2 starts from the best warmup state, not just the last epoch.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
        
        Returns:
            Dictionary with phase 2 metrics
        """
        print(f"\n{'='*60}")
        print(f"PHASE 2: FINE-TUNING (Full Model)")
        print(f"Learning Rate: {self.config.LR_PHASE2}")
        print(f"Epochs: {self.config.EPOCHS_PHASE2}")
        print(f"{'='*60}\n")
        
        # CRITICAL FIX: Reload best Phase 1 checkpoint before unfreezing
        best_phase1_path = self.output_dir / "model_weights" / f"{self.model_name}_phase1_best.pth"
        if best_phase1_path.exists():
            self.model.load_state_dict(torch.load(best_phase1_path, map_location=self.device))
            print(f"✓ Reloaded best Phase 1 checkpoint: {best_phase1_path}")
        else:
            print(f"Warning: Phase 1 checkpoint not found. Continuing with current model.")
        
        # Unfreeze all layers
        unfreeze_all(self.model)
        
        # Reset optimizer for phase 2 with much lower learning rate
        # CRITICAL: Use only trainable params (which are now all of them)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.LR_PHASE2,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-7,
        )
        
        best_val_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.config.EPOCHS_PHASE2):
            # Training
            train_loss, train_acc, train_f1 = self._train_epoch(train_loader)
            
            # Validation (REAL F1, not fake)
            val_loss, val_acc, val_f1 = self._validate(val_loader)
            
            # Track history for later analysis
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.train_history['f1'].append(train_f1)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            self.val_history['f1'].append(val_f1)
            
            self.scheduler.step()
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config.EPOCHS_PHASE2}")
            print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.4f}, F1={train_f1:.4f}")
            print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.4f}, F1={val_f1:.4f}")
            
            # Early stopping based on REAL F1 (not fake accuracy)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                self._save_checkpoint(epoch, "phase2")
                print(f"  ✓ New best F1: {val_f1:.4f}")
            else:
                patience_counter += 1
            
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return {
            'phase': 'phase2',
            'best_val_f1': best_val_f1,
        }
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Train for one epoch. Computes REAL macro F1-score (not fake).
        
        Returns:
            Tuple of (loss, accuracy, macro_f1)
        """
        self.model.train()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Collect predictions for F1 computation
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Compute REAL metrics (not fake)
        avg_loss = total_loss / len(train_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, accuracy, f1_macro
    
    def _validate(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        """
        Validate on validation set. Computes REAL macro F1-score (not fake).
        
        Returns:
            Tuple of (loss, accuracy, macro_f1)
        """
        self.model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute REAL metrics (not fake)
        avg_loss = total_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, accuracy, f1_macro
    
    def _save_checkpoint(self, epoch: int, phase: str) -> None:
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "model_weights"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{self.model_name}_{phase}_best.pth"
        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def save_history(self) -> None:
        """Save training and validation history to JSON for later analysis."""
        import json
        
        history_dir = self.output_dir / "training_history"
        history_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        history_file = history_dir / f"{self.model_name}_history.json"
        history = {
            'train': self.train_history,
            'val': self.val_history,
        }
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"✓ Training history saved to {history_file}")


def main(args):
    """Main training function."""
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    config = Config()
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, _, split_metadata = load_data_split(
        data_dir=args.data_dir,
        input_size=384 if args.model != 'swin_tiny' else 224,
        batch_size=args.batch_size,
    )
    
    # Save split metadata for reproducible evaluation
    save_split_metadata(split_metadata, args.output_dir)
    
    # Create trainer
    trainer = Trainer(
        model_name=args.model,
        device=device,
        output_dir=args.output_dir,
        config=config,
    )
    
    # Set class weights based on training data distribution
    print("\nComputing class weights for imbalance handling...")
    trainer.set_class_weights(train_loader)
    
    # Phase 1: Warmup
    phase1_results = trainer.train_phase1(train_loader, val_loader)
    
    # Phase 2: Fine-tuning
    phase2_results = trainer.train_phase2(train_loader, val_loader)
    
    # Save training history
    trainer.save_history()
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train cervical cytology classifier"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "efficientnet_b0", "swin_tiny"],
        default="resnet50",
        help="Model architecture"
    )
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)
