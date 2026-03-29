# Codebase - Food Freshness Detection CNN

Complete Python package for training, evaluating, and analyzing a custom 11-layer CNN for food freshness classification.

## Dataset

**Fresh and Spoiled Food Image Dataset**  
🔗 [Kaggle Dataset Link](https://www.kaggle.com/datasets/maheen00shahid/fresh-and-spoiled-food-image-dataset)

This dataset contains ~2,700+ labeled images of fresh and spoiled food across 8 categories:
- Fresh Fruits, Fresh Vegetables, Fresh Bread, Fresh Dairy
- Spoiled Fruits, Spoiled Vegetables, Spoiled Bread, Spoiled Dairy

---

## Files

### 1. `model.py`
**11-Layer Custom CNN Architecture**

- Conv Block 1: 3→8 filters, output 112×112
- Conv Block 2: 8→16 filters, output 56×56
- Conv Block 3: 16→32 filters, output 28×28
- Dense Layer 1: 25,088→128 neurons
- Dense Layer 2: 128→2 classes (FRESH/SPOILED)

**Key Functions:**
```python
model = FoodFreshnessDetectionCNN()
output = model(input_tensor)  # Shape: (batch_size, 2)
layer_outputs = model.get_layer_outputs(input_tensor)  # For visualization
param_count = model.count_parameters()
```

---

### 2. `utils.py`
**Data Loading & Preprocessing**

**Classes:**
- `FoodFreshnessDataset` - Custom PyTorch Dataset

**Functions:**
- `create_dataloaders()` - Create train/test loaders with augmentation
- `get_transforms()` - Data augmentation pipeline
- `calculate_metrics()` - Accuracy, precision, recall, F1
- `visualize_batch()` - Display image batches
- `denormalize()` - Reverse ImageNet normalization

**Usage:**
```python
train_loader, test_loader = create_dataloaders(
    train_dir='./train',
    train_labels_csv='./train_labels.csv',
    test_dir='./test',
    test_labels_csv='./test_labels.csv',
    batch_size=32
)
```

---

### 3. `train.py`
**Training Script**

**Class:** `Trainer`

**Implements:**
- Forward pass (image → prediction)
- Backward pass (compute gradients)
- Training loop with validation
- Learning rate scheduling
- Best model checkpointing
- Training history logging

**Usage:**
```bash
python train.py
```

Or programmatically:
```python
from train import Trainer
trainer = Trainer(device='cuda', learning_rate=0.001)
history = trainer.train(train_loader, test_loader, epochs=25)
```

**Output:**
- `best_model.pth` - Best trained model
- `training_history.json` - Loss and accuracy curves

---

### 4. `analyze.py`
**Explainability & Analysis**

**Class:** `FoodAnalyzer`

**Analysis Types:**
1. **Saliency Map** - ∂Loss/∂Input (which pixels mattered)
2. **Color Histogram** - Quantitative color distribution
3. **Layer Activations** - What each layer learned
4. **Feature Importance** - Top factors in decision

**Usage:**
```python
from analyze import FoodAnalyzer

analyzer = FoodAnalyzer(model_path='best_model.pth', device='cuda')
analysis, saliency, colors, layers = analyzer.analyze_image('image.jpg')

# Access results:
print(f"Prediction: {analysis['prediction']['class']}")
print(f"Confidence: {analysis['prediction']['confidence']}")
print(f"Colors: {colors}")
```

---

### 5. `demo.py`
**Interactive Demonstration**

**Functions:**
- `generate_demo_report()` - Single image analysis with visualizations
- `batch_demo()` - Analyze multiple images

**Output:**
- Analysis visualization (6-panel plot)
- JSON report with metrics
- Batch summary report

**Usage:**
```python
from demo import generate_demo_report, batch_demo

# Single image
generate_demo_report('image.jpg', 'best_model.pth')

# Batch
batch_demo('./test_images/', 'best_model.pth', max_images=10)
```

---

## Workflow

### Step 1: Training
```bash
cd Codebase
python train.py
```

**Output:**
- `best_model.pth` (trained model)
- `training_history.json` (metrics)

### Step 2: Single Image Analysis
```python
from analyze import FoodAnalyzer

analyzer = FoodAnalyzer('best_model.pth', device='cpu')
analysis, saliency, colors, layers = analyzer.analyze_image('food_image.jpg')
```

### Step 3: Full Demo
```python
from demo import generate_demo_report

generate_demo_report('food_image.jpg', 'best_model.pth')
```

### Step 4: Batch Analysis
```python
from demo import batch_demo

batch_demo('./test_images/', 'best_model.pth', max_images=20)
```

---

## Installation & Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**On Windows (PowerShell):**
```bash
.\venv\Scripts\Activate.ps1
```

**On Windows (CMD):**
```bash
.\venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r Codebase/requirements.txt
```

### 4. Run Web App
```bash
python -m streamlit run Codebase/app.py
```

**The app will be available at:** `http://localhost:8501`

---

## Model Architecture Details

| Layer | Type | Input | Output | Parameters |
|-------|------|-------|--------|------------|
| 1 | Conv2d | (3, 224, 224) | (8, 224, 224) | 224 |
| 2 | ReLU | (8, 224, 224) | (8, 224, 224) | 0 |
| 3 | MaxPool2d | (8, 224, 224) | (8, 112, 112) | 0 |
| 4 | Conv2d | (8, 112, 112) | (16, 112, 112) | 1,168 |
| 5 | ReLU | (16, 112, 112) | (16, 112, 112) | 0 |
| 6 | MaxPool2d | (16, 112, 112) | (16, 56, 56) | 0 |
| 7 | Conv2d | (16, 56, 56) | (32, 56, 56) | 4,640 |
| 8 | ReLU | (32, 56, 56) | (32, 56, 56) | 0 |
| 9 | MaxPool2d | (32, 56, 56) | (32, 28, 28) | 0 |
| 10 | Dense | 25,088 | 128 | 3,211,392 |
| 11 | Dense | 128 | 2 | 258 |
| | **TOTAL** | | | **3,217,682** |

---

## Hyperparameters

- **Optimizer:** Adam (lr=0.001)
- **Loss:** CrossEntropyLoss
- **Batch Size:** 32
- **Epochs:** 25
- **Learning Rate Schedule:** ReduceLROnPlateau (factor=0.5, patience=5)
- **Dropout:** 0.3
- **Weight Initialization:** Kaiming Normal (He)

---

## Expected Performance

- **Train Accuracy:** 95-98%
- **Test Accuracy:** 90-93%
- **F1 Score:** 0.92
- **Training Time (CPU):** ~45 mins
- **Training Time (GPU):** ~8 mins
- **Inference Time:** ~12ms per image

---

## Files Structure

```
Codebase/
├── __init__.py           # Package init
├── model.py              # CNN architecture (11 layers)
├── utils.py              # Data loading & preprocessing
├── train.py              # Training script
├── analyze.py            # Explainability analysis
├── demo.py               # Interactive demo
└── README.md             # This file
```

---

## Output Files

After running scripts, you'll get:

```
Project/
├── best_model.pth           # Trained model weights
├── training_history.json    # Training curves
├── {image}_analysis.png     # Visualization (from demo)
├── {image}_analysis.json    # Detailed metrics (from demo)
└── batch_report.json        # Batch summary (from batch_demo)
```

---

## Example Commands

### Train Model
```bash
python train.py
# Output: best_model.pth, training_history.json
```

### Analyze Single Image
```python
python -c "from demo import generate_demo_report; generate_demo_report('test.jpg', 'best_model.pth')"
# Output: test_analysis.png, test_analysis.json
```

### Batch Analysis
```python
python -c "from demo import batch_demo; batch_demo('./test_images/', 'best_model.pth')"
# Output: batch_report.json
```

---

## Troubleshooting

**Issue:** CUDA out of memory
- **Solution:** Reduce batch_size in train.py from 32 to 16 or 8

**Issue:** Slow training on CPU
- **Solution:** Enable GPU or reduce image size from 224 to 128

**Issue:** Model not improving
- **Solution:** Reduce learning rate or increase epochs

---

## License

Educational project - 2026

---

**Questions?** Check the README_ARCHITECTURE.md in parent directory for detailed mathematical explanations.
