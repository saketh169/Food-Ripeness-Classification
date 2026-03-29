"""
Training Script
Trains the 11-layer CNN on food freshness dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm

from model import FoodFreshnessDetectionCNN
from utils import create_dataloaders, calculate_metrics


class Trainer:
    def __init__(self, device='cpu', learning_rate=0.001, batch_size=32):
        """
        Initialize trainer
        
        Args:
            device: 'cpu' or 'cuda'
            learning_rate: Initial learning rate
            batch_size: Batch size for training
        """
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Create model
        self.model = FoodFreshnessDetectionCNN().to(device)
        print(f"Model created with {self.model.count_parameters():,} parameters")
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'test_f1': []
        }
        
        self.best_test_accuracy = 0
        self.best_model_state = None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(outputs.detach())
            all_labels.append(labels.detach())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = calculate_metrics(all_predictions, all_labels)
        
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss, metrics['accuracy']
    
    def evaluate(self, test_loader):
        """Evaluate on test set"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Evaluating", leave=False)
            
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                all_predictions.append(outputs)
                all_labels.append(labels)
        
        # Calculate metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = calculate_metrics(all_predictions, all_labels)
        
        avg_loss = total_loss / len(test_loader)
        
        return avg_loss, metrics
    
    def train(self, train_loader, test_loader, epochs=20, save_path='model.pth'):
        """
        Full training loop
        
        Args:
            train_loader: Training dataloader
            test_loader: Test dataloader
            epochs: Number of epochs
            save_path: Where to save best model
        """
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {epochs}")
        print("="*70 + "\n")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Evaluate
            test_loss, test_metrics = self.evaluate(test_loader)
            test_acc = test_metrics['accuracy']
            test_f1 = test_metrics['f1']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_accuracy'].append(test_acc)
            self.history['test_f1'].append(test_f1)
            
            # Learning rate scheduling
            self.scheduler.step(test_loss)
            
            # Save best model
            if test_acc > self.best_test_accuracy:
                self.best_test_accuracy = test_acc
                self.best_model_state = self.model.state_dict().copy()
                best_indicator = " ← BEST"
            else:
                best_indicator = ""
            
            # Print progress
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | "
                  f"F1: {test_f1:.4f}{best_indicator}")
        
        # Save best model
        torch.save(self.best_model_state, save_path)
        print(f"\n✓ Best model saved to: {save_path}")
        print(f"✓ Best test accuracy: {self.best_test_accuracy:.4f}")
        
        return self.history
    
    def save_history(self, path='json/training_history.json'):
        """Save training history to json/ folder"""
        # Ensure json folder exists
        json_dir = Path(path).parent
        json_dir.mkdir(parents=True, exist_ok=True)
        
        history_serializable = {k: [float(v) for v in vals] 
                               for k, vals in self.history.items()}
        
        with open(path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        print(f"✓ Training history saved to: {path}")


def main():
    """Main training function"""
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    train_dir = str(PROJECT_ROOT / 'Dataset' / 'train')
    train_labels_csv = str(PROJECT_ROOT / 'train_labels.csv')
    test_dir = str(PROJECT_ROOT / 'Dataset' / 'test')
    test_labels_csv = str(PROJECT_ROOT / 'test_labels.csv')
    
    # Hyperparameters
    epochs = 25
    batch_size = 32
    learning_rate = 0.001
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        train_dir=train_dir,
        train_labels_csv=train_labels_csv,
        test_dir=test_dir,
        test_labels_csv=test_labels_csv,
        batch_size=batch_size
    )
    
    # Create trainer
    trainer = Trainer(
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        save_path='../best_model.pth'
    )
    
    # Save history to json/ folder
    trainer.save_history('../json/training_history.json')
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
