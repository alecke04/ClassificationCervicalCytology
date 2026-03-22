"""
Model definitions: ResNet50, EfficientNet-B0, Swin Transformer
All models wrapped for easy training/evaluation
"""

import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b0
from torchvision.models import ResNet50_Weights, EfficientNet_B0_Weights
from timm import create_model


def get_resnet50(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Load and configure ResNet50 for medical image classification.
    
    Args:
        num_classes: Number of output classes (default 3)
        pretrained: Load pretrained ImageNet weights
    
    Returns:
        Modified ResNet50 model with 3-class output head
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model


def get_efficientnet_b0(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Load and configure EfficientNet-B0 for medical image classification.
    
    Args:
        num_classes: Number of output classes (default 3)
        pretrained: Load pretrained ImageNet weights
    
    Returns:
        Modified EfficientNet-B0 model with 3-class output head
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)
    
    # Replace final fully connected layer
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, num_classes)
    )
    
    return model


def get_swin_tiny(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """
    Load and configure Swin Transformer Tiny for medical image classification.
    
    Args:
        num_classes: Number of output classes (default 3)
        pretrained: Load pretrained ImageNet weights
    
    Returns:
        Modified Swin Transformer model with 3-class output head
    
    Note:
        Swin Transformer expects 224×224 input (native patch size 4×4)
    """
    model = create_model(
        'swin_tiny_patch4_window7_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    
    return model


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """
    Freeze backbone layers for fine-tuning (Phase 1: head training only).
    
    CRITICAL: Policy is now consistent across all models - freeze EVERYTHING
    except the final classification head. This ensures fair Phase 1 comparison.
    
    Args:
        model: PyTorch model
        model_name: Name of model ('resnet50', 'efficientnet_b0', or 'swin_tiny')
    """
    if model_name == 'resnet50':
        # Freeze entire backbone; only fc (head) is trainable
        for name, param in model.named_parameters():
            if 'fc' not in name:  # Only leave fc trainable
                param.requires_grad = False
        
    elif model_name == 'efficientnet_b0':
        # Freeze entire backbone; only classifier (head) is trainable
        for name, param in model.named_parameters():
            if 'classifier' not in name:  # Only leave classifier trainable
                param.requires_grad = False
        
    elif model_name == 'swin_tiny':
        # Freeze entire backbone; only head is trainable
        for name, param in model.named_parameters():
            if 'head' not in name:  # Only leave head trainable
                param.requires_grad = False
    
    # Count trainable params for verification
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Backbone frozen for {model_name}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreeze all layers for fine-tuning (phase 2: full model tuning).
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    
    print("✓ All layers unlocked for fine-tuning")


def get_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone_phase1: bool = True,
) -> nn.Module:
    """
    Factory function to load any model by name.
    
    CRITICAL: All models use IDENTICAL freezing policy - entire backbone frozen,
    only head trainable. This ensures fair Phase 1 comparison.
    
    Args:
        model_name: 'resnet50', 'efficientnet_b0', or 'swin_tiny'
        num_classes: Number of output classes
        pretrained: Load pretrained weights
        freeze_backbone_phase1: Freeze backbone for phase 1 (warmup)
    
    Returns:
        Configured PyTorch model
    
    Example:
        >>> model = get_model('resnet50')
        >>> model = get_model('efficientnet_b0', freeze_backbone_phase1=True)
    """
    models = {
        'resnet50': get_resnet50,
        'efficientnet_b0': get_efficientnet_b0,
        'swin_tiny': get_swin_tiny,
    }
    
    if model_name not in models:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Choose from {list(models.keys())}"
        )
    
    model = models[model_name](num_classes=num_classes, pretrained=pretrained)
    
    if freeze_backbone_phase1:
        freeze_backbone(model, model_name)
    
    return model


def count_parameters(model: nn.Module) -> tuple:
    """
    Count trainable and total parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def print_model_info(model: nn.Module, model_name: str) -> None:
    """
    Print model information.
    
    Args:
        model: PyTorch model
        model_name: Name of model for display
    """
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {trainable:,}")
    print(f"Frozen parameters:     {total - trainable:,}")
    print(f"Trainable ratio:       {100 * trainable / total:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test model loading
    import torch
    
    print("Testing model loading...")
    
    for model_name in ['resnet50', 'efficientnet_b0', 'swin_tiny']:
        model = get_model(model_name, num_classes=3)
        print_model_info(model, model_name)
        
        # Test forward pass
        input_size = 224 if model_name == 'swin_tiny' else 384
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✓ {model_name} forward pass successful")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print()
    
    print("✓ All models tested successfully")
