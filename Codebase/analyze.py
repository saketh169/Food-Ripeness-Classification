"""
Analysis & Explainability Module
Saliency maps, feature importance, color histogram, layer activations
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import json

from model import FoodFreshnessDetectionCNN
from utils import get_transforms, denormalize


class FoodAnalyzer:
    def __init__(self, model_path, device='cpu'):
        """
        Initialize analyzer
        
        Args:
            model_path: Path to saved model
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = FoodFreshnessDetectionCNN().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        self.transform = get_transforms('test')
        self.label_map = {0: 'FRESH', 1: 'SPOILED'}
    
    def load_image(self, image_path):
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img, img_tensor
    
    def predict(self, image_tensor):
        """Get prediction"""
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities
    
    def compute_saliency_map(self, image_tensor):
        """
        Compute saliency map using gradient
        Saliency = |∂Loss/∂Input|
        """
        image_tensor.requires_grad = True
        
        # Forward pass
        logits = self.model(image_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        loss = logits[0, predicted_class]
        loss.backward()
        
        # Get gradient magnitude
        saliency = torch.abs(image_tensor.grad).max(dim=1)[0].squeeze()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency.detach().cpu().numpy()
    
    def compute_color_histogram(self, image_path):
        """
        Compute color histogram
        Analyzes color distribution
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Convert to HSV for better color analysis
        from PIL import Image as PILImage
        img_hsv = img.convert('HSV')
        img_hsv_array = np.array(img_hsv)
        
        # Extract colors
        h = img_hsv_array[:, :, 0]
        
        # Define color ranges (simplified)
        green_mask = (h > 40) & (h < 80)
        yellow_mask = (h > 20) & (h <= 40)
        red_mask = ((h >= 0) & (h < 20)) | (h > 340)
        brown_mask = (h > 10) & (h < 40) & (img_hsv_array[:, :, 1] > 100)
        
        total_pixels = h.size
        colors = {
            'green': np.sum(green_mask) / total_pixels * 100,
            'yellow': np.sum(yellow_mask) / total_pixels * 100,
            'red': np.sum(red_mask) / total_pixels * 100,
            'brown': np.sum(brown_mask) / total_pixels * 100
        }
        
        return colors
    
    def get_layer_activations(self, image_tensor):
        """Get activation maps from intermediate layers"""
        self.model.eval()
        
        with torch.no_grad():
            activations = self.model.get_layer_outputs(image_tensor)
        
        layer_stats = {}
        for layer_name, activation in activations.items():
            layer_stats[layer_name] = {
                'mean_activation': float(activation.mean().item()),
                'max_activation': float(activation.max().item()),
                'min_activation': float(activation.min().item()),
                'shape': str(activation.shape)
            }
        
        return layer_stats
    
    def compute_feature_importance(self, image_tensor):
        """
        Compute feature importance using gradient magnitude
        """
        image_tensor.requires_grad = True
        
        # Forward pass
        logits = self.model(image_tensor)
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Backward pass
        loss = logits[0, predicted_class]
        
        # Get gradient for each layer
        loss.backward(create_graph=True)
        
        # Compute gradient magnitudes for first few layers
        importance_scores = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None and 'conv' in name:
                grad_magnitude = torch.abs(param.grad).mean().item()
                importance_scores[name] = grad_magnitude
        
        # Normalize and sort
        total_importance = sum(importance_scores.values())
        importance_normalized = {k: v/total_importance for k, v in importance_scores.items()}
        importance_ranked = sorted(importance_normalized.items(), key=lambda x: x[1], reverse=True)
        
        return importance_ranked
    
    def analyze_image(self, image_path):
        """
        Full analysis of an image
        """
        print("="*70)
        print("ANALYZING IMAGE")
        print("="*70)
        
        # Load image
        img, img_tensor = self.load_image(image_path)
        
        # Prediction
        predicted_class, confidence, probabilities = self.predict(img_tensor)
        prediction_label = self.label_map[predicted_class]
        
        print(f"\n1. PREDICTION")
        print(f"   Class: {prediction_label}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   Probabilities: Fresh={probabilities[0, 0].item():.4f}, "
              f"Spoiled={probabilities[0, 1].item():.4f}")
        
        # Ripeness stage
        confidence_margin = abs(probabilities[0, 0].item() - probabilities[0, 1].item())
        if confidence >= 0.85:
            stage = "READY" if predicted_class == 0 else "SPOILED"
        elif confidence >= 0.6:
            stage = "APPROACHING" if predicted_class == 0 else "PAST-PEAK"
        else:
            stage = "UNCERTAIN"
        
        print(f"   Ripeness Stage: {stage}")
        print(f"   Confidence Margin: {confidence_margin:.4f}")
        
        # Saliency map
        print(f"\n2. SALIENCY MAP")
        saliency = self.compute_saliency_map(img_tensor)
        print(f"   Gradient magnitude range: [{saliency.min():.4f}, {saliency.max():.4f}]")
        print(f"   Mean gradient: {saliency.mean():.4f}")
        
        # Color histogram
        print(f"\n3. COLOR HISTOGRAM")
        colors = self.compute_color_histogram(image_path)
        for color, percentage in colors.items():
            print(f"   {color.capitalize()}: {percentage:.2f}%")
        
        # Layer activations
        print(f"\n4. LAYER ACTIVATIONS")
        layer_stats = self.get_layer_activations(img_tensor)
        for layer_name, stats in layer_stats.items():
            print(f"   {layer_name}:")
            print(f"     Mean: {stats['mean_activation']:.4f}")
            print(f"     Max: {stats['max_activation']:.4f}")
            print(f"     Shape: {stats['shape']}")
        
        # Feature importance
        print(f"\n5. FEATURE IMPORTANCE (Top 3)")
        importance = self.compute_feature_importance(img_tensor)
        for i, (layer_name, importance_score) in enumerate(importance[:3]):
            print(f"   {i+1}. {layer_name}: {importance_score:.4f}")
        
        # Create analysis dict
        analysis = {
            'timestamp': str(np.datetime64('now')),
            'image': image_path,
            'prediction': {
                'class': prediction_label,
                'confidence': float(confidence),
                'stage': stage
            },
            'saliency': {
                'mean': float(saliency.mean()),
                'max': float(saliency.max()),
                'min': float(saliency.min())
            },
            'colors': colors,
            'layer_activations': layer_stats
        }
        
        print("\n" + "="*70)
        
        return analysis, saliency, colors, layer_stats


if __name__ == "__main__":
    print("Analysis module loaded successfully!")
