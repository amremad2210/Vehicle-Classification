import torch
import torch.nn as nn
from torchvision import models


class VehicleResNet(nn.Module):
    """ResNet-18 based transfer learning model for vehicle classification."""
    
    def __init__(self, num_classes: int = 12, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            num_classes: Number of vehicle classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: If True, freeze all layers except classifier
        """
        super(VehicleResNet, self).__init__()
        
        # Load pretrained ResNet-18
        self.model = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone layers if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace final fully connected layer with enhanced classifier
        # ResNet-18 has 512 features before FC layer
        num_features = self.model.fc.in_features
        
        # Enhanced multi-layer classifier with dropout
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        print(f"Loaded ResNet-18 (pretrained={pretrained}, freeze_backbone={freeze_backbone})")
        print(f"Enhanced classifier: {num_features} -> 256 -> 128 -> {num_classes}")
    
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - all layers trainable")
    
    def get_num_params(self):
        """Return total number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,} | Trainable: {trainable:,}")
        return total


class VehicleResNet34(nn.Module):
    """ResNet-34 based transfer learning model for vehicle classification."""
    
    def __init__(self, num_classes: int = 12, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            num_classes: Number of vehicle classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: If True, freeze all layers except classifier
        """
        super(VehicleResNet34, self).__init__()
        
        # Load pretrained ResNet-34
        self.model = models.resnet34(pretrained=pretrained)
        
        # Freeze backbone layers if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace final fully connected layer with enhanced classifier
        # ResNet-34 has 512 features before FC layer
        num_features = self.model.fc.in_features
        
        # Enhanced multi-layer classifier with dropout
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        print(f"Loaded ResNet-34 (pretrained={pretrained}, freeze_backbone={freeze_backbone})")
        print(f"Enhanced classifier: {num_features} -> 256 -> 128 -> {num_classes}")
    
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - all layers trainable")
    
    def get_num_params(self):
        """Return total number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,} | Trainable: {trainable:,}")
        return total


class VehicleEfficientNet(nn.Module):
    """EfficientNet-B0 based transfer learning model (alternative option)."""
    
    def __init__(self, num_classes: int = 12, pretrained: bool = True, freeze_backbone: bool = False):
        """
        Args:
            num_classes: Number of vehicle classes
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: If True, freeze all layers except classifier
        """
        super(VehicleEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Freeze backbone layers if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Replace final classifier layer
        # EfficientNet-B0 has 1280 features before classifier
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
        print(f"Loaded EfficientNet-B0 (pretrained={pretrained}, freeze_backbone={freeze_backbone})")
        print(f"Replaced classifier: {num_features} -> {num_classes}")
    
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Backbone unfrozen - all layers trainable")
    
    def get_num_params(self):
        """Return total number of parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,} | Trainable: {trainable:,}")
        return total