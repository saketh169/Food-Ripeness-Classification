"""
Model Definition - 11-Layer Custom CNN
All layers mathematically defined with proper initialization
"""

import torch
import torch.nn as nn
import torch.nn.init as init


class FoodFreshnessDetectionCNN(nn.Module):
    """
    Custom 11-layer CNN for food freshness classification
    Binary classification: Fresh (0) or Spoiled (1)
    
    Architecture:
    Input (224x224x3)
        ↓
    Conv Block 1: Conv + ReLU + MaxPool → (112x112x8)
        ↓
    Conv Block 2: Conv + ReLU + MaxPool → (56x56x16)
        ↓
    Conv Block 3: Conv + ReLU + MaxPool → (28x28x32)
        ↓
    Flatten → 25,088 features
        ↓
    Dense Layer 1: Dense + ReLU → 128 neurons
        ↓
    Dropout (0.3)
        ↓
    Dense Layer 2: Dense → 2 classes
        ↓
    Output: Softmax probabilities
    """
    
    def __init__(self):
        super(FoodFreshnessDetectionCNN, self).__init__()
        
        # Layer 1-3: Conv Block 1 (3→8 filters)
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 8 filters, 112×112
        
        # Layer 4-6: Conv Block 2 (8→16 filters)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 16 filters, 56×56
        
        # Layer 7-9: Conv Block 3 (16→32 filters)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 32 filters, 28×28
        
        # Flatten: 32 × 28 × 28 = 25,088
        self.flatten = nn.Flatten()
        
        # Layer 10: Dense Layer (25088 → 128)
        self.dense1 = nn.Linear(32 * 28 * 28, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.3)
        
        # Layer 11: Output Layer (128 → 2)
        self.dense2 = nn.Linear(128, 2)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU networks"""
        for layer in [self.conv1, self.conv2, self.conv3]:
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
        
        for layer in [self.dense1, self.dense2]:
            init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            logits: Raw output before softmax (batch_size, 2)
        """
        # Conv Block 1
        x = self.conv1(x)        # (B, 8, 224, 224)
        x = self.relu1(x)        # (B, 8, 224, 224)
        x = self.maxpool1(x)     # (B, 8, 112, 112)
        
        # Conv Block 2
        x = self.conv2(x)        # (B, 16, 112, 112)
        x = self.relu2(x)        # (B, 16, 112, 112)
        x = self.maxpool2(x)     # (B, 16, 56, 56)
        
        # Conv Block 3
        x = self.conv3(x)        # (B, 32, 56, 56)
        x = self.relu3(x)        # (B, 32, 56, 56)
        x = self.maxpool3(x)     # (B, 32, 28, 28)
        
        # Flatten
        x = self.flatten(x)      # (B, 25088)
        
        # Dense layers
        x = self.dense1(x)       # (B, 128)
        x = self.relu4(x)        # (B, 128)
        x = self.dropout(x)      # (B, 128) - Dropout for regularization
        
        # Output
        logits = self.dense2(x)  # (B, 2)
        
        return logits
    
    def get_layer_outputs(self, x):
        """
        Get intermediate layer outputs for visualization
        
        Returns dict with activation maps
        """
        layer_outputs = {}
        
        x = self.conv1(x)
        layer_outputs['conv1'] = x.detach()
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        layer_outputs['conv2'] = x.detach()
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.conv3(x)
        layer_outputs['conv3'] = x.detach()
        x = self.relu3(x)
        x = self.maxpool3(x)
        
        return layer_outputs
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(device='cpu'):
    """Create and return model"""
    model = FoodFreshnessDetectionCNN().to(device)
    return model


if __name__ == "__main__":
    # Test model
    model = create_model('cpu')
    print(f"Model created successfully!")
    print(f"Total parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
