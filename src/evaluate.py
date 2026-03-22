"""
Evaluation: metrics computation, confusion matrices, Grad-CAM visualizations
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Tuple, List, Optional
import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import argparse


class Evaluator:
    """
    Evaluation manager for model assessment and visualization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str] = ["NILM", "LSIL", "HSIL"],
        output_dir: str = "results",
    ):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            device: torch.device
            class_names: List of class names
            output_dir: Directory to save results
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.eval()
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            test_loader: Test DataLoader
        
        Returns:
            Dictionary of metrics
        """
        print("\nEvaluating on test set...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Evaluating"):
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels, all_probs)
        
        # Generate visualizations
        self._plot_confusion_matrix(all_preds, all_labels)
        self._plot_roc_curves(all_labels, all_probs)
        self._generate_grad_cam_samples(test_loader)
        
        # Save metrics
        self._save_metrics(metrics)
        
        return metrics

    def _get_grad_cam_config(self) -> Tuple[Optional[List[nn.Module]], Optional[Callable]]:
        """Get target layers and optional reshape transform for Grad-CAM."""
        # ResNet50
        if hasattr(self.model, 'layer4'):
            return [self.model.layer4[-1]], None

        # EfficientNet-B0
        if hasattr(self.model, 'features') and len(self.model.features) > 0:
            return [self.model.features[-1]], None

        # Swin Tiny (timm): use last block norm and reshape features to BCHW.
        if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
            try:
                return [self.model.layers[-1].blocks[-1].norm1], self._swin_reshape_transform
            except Exception:
                return None, None

        return None, None

    def _swin_reshape_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape Swin features into BCHW for Grad-CAM.

        Handles either token-like [B, N, C] or channel-last maps [B, H, W, C].
        """
        if tensor.ndim == 4:
            # [B, H, W, C] -> [B, C, H, W]
            return tensor.permute(0, 3, 1, 2)

        if tensor.ndim == 3:
            # [B, N, C] -> [B, C, H, W] assuming square token grid.
            batch_size, token_count, channels = tensor.shape
            spatial_size = int(np.sqrt(token_count))
            if spatial_size * spatial_size != token_count:
                raise ValueError(
                    f"Cannot reshape Swin tokens: token_count={token_count} is not a square"
                )
            tensor = tensor.reshape(batch_size, spatial_size, spatial_size, channels)
            return tensor.permute(0, 3, 1, 2)

        raise ValueError(f"Unsupported Swin feature shape for Grad-CAM: {tuple(tensor.shape)}")

    def _denormalize_image(self, tensor_chw: torch.Tensor) -> np.ndarray:
        """Convert normalized CHW tensor to HWC float image in [0, 1]."""
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor_chw.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor_chw.dtype).view(3, 1, 1)
        image = tensor_chw.cpu() * std + mean
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()
        return image

    def _generate_grad_cam_samples(self, test_loader: DataLoader, num_samples: int = 12) -> None:
        """Generate and save Grad-CAM overlays for a subset of test samples."""
        target_layers, reshape_transform = self._get_grad_cam_config()
        if target_layers is None:
            print("- Grad-CAM skipped: unsupported model backbone for current implementation")
            return

        samples = []
        true_labels = []
        pred_labels = []

        # Collect a small subset from test data.
        for images, labels in test_loader:
            images = images.to(self.device)
            with torch.no_grad():
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)

            for i in range(images.size(0)):
                samples.append(images[i].detach().cpu())
                true_labels.append(int(labels[i].item()))
                pred_labels.append(int(preds[i].item()))
                if len(samples) >= num_samples:
                    break
            if len(samples) >= num_samples:
                break

        if not samples:
            print("- Grad-CAM skipped: no test samples available")
            return

        input_tensor = torch.stack(samples).to(self.device)
        input_tensor.requires_grad_(True)
        cam_targets = [ClassifierOutputTarget(cls_idx) for cls_idx in pred_labels]

        grad_cam_dir = self.output_dir / 'grad_cam'
        grad_cam_dir.mkdir(parents=True, exist_ok=True)

        with GradCAM(
            model=self.model,
            target_layers=target_layers,
            reshape_transform=reshape_transform,
        ) as cam:
            grayscale_cams = cam(input_tensor=input_tensor, targets=cam_targets)

        # Save individual overlays
        for i, cam_map in enumerate(grayscale_cams):
            image_rgb = self._denormalize_image(samples[i])
            overlay = show_cam_on_image(image_rgb, cam_map, use_rgb=True)

            true_name = self.class_names[true_labels[i]]
            pred_name = self.class_names[pred_labels[i]]
            out_path = grad_cam_dir / f"sample_{i+1:02d}_true_{true_name}_pred_{pred_name}.png"
            plt.imsave(out_path, overlay)

        print(f"✓ Grad-CAM samples saved to {grad_cam_dir}")
    
    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> Dict:
        """
        Compute comprehensive metrics.
        
        Returns:
            Dictionary with accuracy, F1, AUC, per-class metrics
        """
        num_classes = len(self.class_names)
        
        # Overall metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1, 2], zero_division=0
        )
        
        # ROC-AUC (one-vs-rest for each class)
        auc_scores = []
        for i in range(num_classes):
            try:
                auc = roc_auc_score(
                    (labels == i).astype(int),
                    probabilities[:, i]
                )
                auc_scores.append(auc)
            except:
                auc_scores.append(np.nan)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
        
        metrics = {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'auc_mean': float(np.nanmean(auc_scores)),
            'per_class': {
                self.class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1': float(f1[i]),
                    'support': int(support[i]),
                    'auc': float(auc_scores[i]),
                }
                for i in range(num_classes)
            },
            'confusion_matrix': cm.tolist(),
        }
        
        return metrics
    
    def _plot_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray) -> None:
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # Save
        output_path = self.output_dir / 'confusion_matrices' / 'confusion_matrix.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {output_path}")
    
    def _plot_roc_curves(self, labels: np.ndarray, probabilities: np.ndarray) -> None:
        """Plot and save ROC curves for each class."""
        num_classes = len(self.class_names)
        
        fig, axes = plt.subplots(1, num_classes, figsize=(15, 4))
        
        for i in range(num_classes):
            ax = axes[i]
            
            # One-vs-rest
            y_true = (labels == i).astype(int)
            y_score = probabilities[:, i]
            
            try:
                # Check if class is present in test set (edge case: class entirely absent)
                if y_true.sum() == 0:
                    # No positive samples for this class
                    ax.text(0.5, 0.5, f'{self.class_names[i]}\n(not in test set)',
                            ha='center', va='center', fontsize=12, color='red')
                    ax.set_title(f'{self.class_names[i]} vs Rest')
                    continue
                
                fpr, tpr, _ = roc_curve(y_true, y_score)
                auc = roc_auc_score(y_true, y_score)
                
                ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
                ax.plot([0, 1], [0, 1], 'k--', label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{self.class_names[i]} vs Rest')
                ax.legend()
                ax.grid(alpha=0.3)
            except Exception as e:
                # Fallback for any ROC computation errors
                ax.text(0.5, 0.5, f'{self.class_names[i]}\n(ROC error: {str(e)[:20]})',
                        ha='center', va='center', fontsize=10, color='red')
                ax.set_title(f'{self.class_names[i]} vs Rest')
        
        # Save
        output_path = self.output_dir / 'confidence' / 'roc_curves.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ ROC curves saved to {output_path}")
    
    def _save_metrics(self, metrics: Dict) -> None:
        """Save metrics to JSON."""
        output_path = self.output_dir / 'metrics.json'
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Also print summary
        print(f"\n{'='*60}")
        print("TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"Accuracy:         {metrics['accuracy']:.4f}")
        print(f"Macro F1:         {metrics['f1_macro']:.4f}")
        print(f"Weighted F1:      {metrics['f1_weighted']:.4f}")
        print(f"Mean AUC:         {metrics['auc_mean']:.4f}")
        print(f"\nPer-Class Metrics:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"\n{class_name}:")
            print(f"  Precision:    {class_metrics['precision']:.4f}")
            print(f"  Recall:       {class_metrics['recall']:.4f}")
            print(f"  F1:           {class_metrics['f1']:.4f}")
            print(f"  AUC:          {class_metrics['auc']:.4f}")
            print(f"  Support:      {class_metrics['support']}")
        print(f"{'='*60}\n")
        
        print(f"✓ Metrics saved to {output_path}")


def main(args):
    """
    Main evaluation function (FIXED from placeholder).
    
    Complete workflow:
    1. Load model architecture
    2. Load best checkpoint
    3. Load EXACT test split from training (not regenerated)
    4. Run comprehensive evaluation
    5. Save metrics and visualizations
    """
    from src.utils import get_device, set_seed
    from src.models import get_model
    from src.data import load_split_metadata, load_test_from_metadata
    
    # Setup
    set_seed(42)
    device = get_device()
    
    print(f"\n{'='*60}")
    print("EVALUATION - Test Set Assessment")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"Loading model: {args.model}")
    model = get_model(
        model_name=args.model,
        num_classes=3,
        pretrained=False,
        freeze_backbone_phase1=False,
    )
    
    # Load checkpoint
    checkpoint_path = Path(args.model_dir) / f"{args.model}_phase2_best.pth"
    if not checkpoint_path.exists():
        # Fallback to phase1 if phase2 doesn't exist
        checkpoint_path = Path(args.model_dir) / f"{args.model}_phase1_best.pth"
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            f"Make sure training completed and checkpoints were saved."
        )
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    print(f"✓ Loaded checkpoint: {checkpoint_path}\n")
    
    # Load SAVED test split (reproducibility - not regenerated!)
    print(f"Loading saved split metadata from: {args.output_dir}")
    split_metadata = load_split_metadata(args.output_dir)
    
    # Create test loader from saved split (test-only, efficient)
    test_loader = load_test_from_metadata(
        split_metadata,
        input_size=384 if args.model != 'swin_tiny' else 224,
        batch_size=args.batch_size,
    )
    
    # Evaluate
    evaluator = Evaluator(
        model=model,
        device=device,
        class_names=["NILM", "LSIL", "HSIL"],
        output_dir=args.output_dir,
    )
    
    metrics = evaluator.evaluate(test_loader)
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate cervical cytology classifier")
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "efficientnet_b0", "swin_tiny"],
        default="resnet50",
        help="Model architecture"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="results/model_weights",
        help="Directory containing model checkpoints"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for metrics and visualizations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation"
    )
    
    args = parser.parse_args()
    main(args)
