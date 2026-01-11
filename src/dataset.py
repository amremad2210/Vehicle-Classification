import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict
import json


class VehicleDataset(Dataset):
    """Dataset for vehicle classification from folder structure."""
    
    def __init__(self, root_dir: str, transform=None, class_map_path: str = None):
        """
        Args:
            root_dir: Path to train/val/test folder
            transform: Optional transform to be applied
            class_map_path: Path to class_names.json (optional, auto-detects from folder names)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Build class mapping from folder names
        self.class_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {folder.name: idx for idx, folder in enumerate(self.class_folders)}
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Gather all images
        self.samples = []
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        for class_folder in self.class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in self.img_extensions:
                    self.samples.append((img_path, class_idx))
        
        print(f"Found {len(self.samples)} images in {len(self.class_folders)} classes")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[str, int]:
        """Return count of samples per class."""
        counts = {}
        for _, label in self.samples:
            class_name = self.idx_to_class[label]
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts


def get_transforms(split: str = 'train', img_size: int = 224) -> transforms.Compose:
    """
    Get data transforms for train/val/test.
    
    Args:
        split: 'train', 'val', or 'test'
        img_size: Target image size (default 224)
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])