"""
Utility Functions
Data loading, preprocessing, and helper functions
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


class FoodFreshnessDataset(Dataset):
    """
    Custom Dataset for food freshness classification
    Loads images and labels from folder and CSV
    """
    
    def __init__(self, image_dir, labels_csv, transform=None):
        """
        Args:
            image_dir: Path to folder with images
            labels_csv: Path to CSV with labels (filename, label)
            transform: Torchvision transforms to apply
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Read labels from CSV
        self.labels_df = pd.read_csv(labels_csv)
        self.images = list(self.image_dir.glob('*.jpg')) + \
                      list(self.image_dir.glob('*.jpeg')) + \
                      list(self.image_dir.glob('*.png'))
        
        print(f"Dataset initialized: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        # Get label from CSV
        filename = img_path.name
        label = self.labels_df[self.labels_df['filename'] == filename]['label'].values[0]
        label = torch.tensor(label, dtype=torch.long)
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        return img, label


def get_transforms(phase='train'):
    """
    Get data augmentation transforms
    
    Args:
        phase: 'train' or 'test'
    """
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:  # test/val
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloaders(train_dir, train_labels_csv, 
                       test_dir, test_labels_csv,
                       batch_size=32, num_workers=0):
    """
    Create PyTorch DataLoaders
    
    Args:
        train_dir: Path to train images
        train_labels_csv: Path to train labels CSV
        test_dir: Path to test images
        test_labels_csv: Path to test labels CSV
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader
    """
    # Create datasets
    train_dataset = FoodFreshnessDataset(
        image_dir=train_dir,
        labels_csv=train_labels_csv,
        transform=get_transforms('train')
    )
    
    test_dataset = FoodFreshnessDataset(
        image_dir=test_dir,
        labels_csv=test_labels_csv,
        transform=get_transforms('test')
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def denormalize(tensor):
    """Reverse ImageNet normalization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    
    return tensor * std + mean


def visualize_batch(images, labels, num_images=4):
    """Visualize a batch of images with labels"""
    fig, axes = plt.subplots(1, num_images, figsize=(12, 3))
    
    for i in range(num_images):
        img = denormalize(images[i]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        
        label_text = "FRESH" if labels[i] == 0 else "SPOILED"
        
        axes[i].imshow(img)
        axes[i].set_title(label_text)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def calculate_metrics(outputs, labels):
    """Calculate accuracy, precision, recall, F1"""
    predictions = torch.argmax(outputs, dim=1)
    
    # Accuracy
    accuracy = (predictions == labels).float().mean().item()
    
    # True Positives, False Positives, False Negatives
    tp = ((predictions == 1) & (labels == 1)).sum().item()
    fp = ((predictions == 1) & (labels == 0)).sum().item()
    fn = ((predictions == 0) & (labels == 1)).sum().item()
    tn = ((predictions == 0) & (labels == 0)).sum().item()
    
    # Precision, Recall, F1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


if __name__ == "__main__":
    print("Utility functions loaded successfully!")
