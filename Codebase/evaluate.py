"""
Comprehensive Test Set Evaluation
Loads best_model.pth and evaluates on full test set
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from model import FoodFreshnessDetectionCNN
from utils import FoodFreshnessDataset, get_transforms
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, roc_auc_score)
import json

# Project root (parent of Codebase)
PROJECT_ROOT = Path(__file__).parent.parent
JSON_DIR = PROJECT_ROOT / 'json'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Load model
print("\n" + "="*60)
print("LOADING MODEL")
print("="*60)
model = FoodFreshnessDetectionCNN().to(device)
model_path = PROJECT_ROOT / 'best_model.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"✓ Model loaded: {model_path}")
print(f"✓ Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load test dataset
print("\n" + "="*60)
print("LOADING TEST SET")
print("="*60)
test_labels_csv = PROJECT_ROOT / 'test_labels.csv'
test_dir = PROJECT_ROOT / 'Dataset' / 'test'

test_labels = pd.read_csv(test_labels_csv)
print(f"✓ Test images: {len(test_labels)}")
print(f"✓ Fresh: {(test_labels['label'] == 0).sum()}")
print(f"✓ Spoiled: {(test_labels['label'] == 1).sum()}")

# Create dataset
test_dataset = FoodFreshnessDataset(
    str(test_dir),
    str(test_labels_csv),
    transform=get_transforms(phase='test')
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# Evaluate
print("\n" + "="*60)
print("EVALUATING MODEL ON TEST SET")
print("="*60)

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (Spoiled)

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

# Metrics
print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"\nOverall Performance:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f} (Is prediction correct when it says Spoiled?)")
print(f"  Recall:    {recall:.4f} (Did we catch all Spoiled fruits?)")
print(f"  F1 Score:  {f1:.4f} (Harmonic mean of Precision & Recall)")
print(f"  AUC-ROC:   {auc:.4f} (Classification ability)")

# Confusion Matrix
print(f"\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(f"                 Predicted")
print(f"                 Fresh  Spoiled")
print(f"Actual Fresh    [{cm[0,0]:4d}]  [{cm[0,1]:4d}]")
print(f"       Spoiled  [{cm[1,0]:4d}]  [{cm[1,1]:4d}]")

# Per-class metrics
print(f"\nPer-Class Performance:")
print(f"Fresh  (Class 0): Precision={precision_score(all_labels, all_preds, pos_label=0):.4f}, Recall={recall_score(all_labels, all_preds, pos_label=0):.4f}")
print(f"Spoiled (Class 1): Precision={precision_score(all_labels, all_preds, pos_label=1):.4f}, Recall={recall_score(all_labels, all_preds, pos_label=1):.4f}")

# Detailed classification report
print(f"\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=['Fresh', 'Spoiled']))

# Save results to json/ folder
JSON_DIR.mkdir(parents=True, exist_ok=True)
results = {
    'accuracy': float(accuracy),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1),
    'auc_roc': float(auc),
    'confusion_matrix': cm.tolist(),
    'test_samples': len(all_labels),
    'fresh_count': int((all_labels == 0).sum()),
    'spoiled_count': int((all_labels == 1).sum())
}

results_path = JSON_DIR / 'test_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {results_path}")
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
