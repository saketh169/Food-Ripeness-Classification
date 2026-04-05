"""
Food Freshness Detection - Complete CLI Tool
Interactive menu with: predict (single/batch), layers, info, evaluate
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from model import FoodFreshnessDetectionCNN
from utils import get_transforms


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'best_model.pth'
TEST_DIR = PROJECT_ROOT / 'Dataset' / 'test'
SAMPLE_DIR = Path(__file__).parent / 'Samples'
OUTPUT_DIR = PROJECT_ROOT / 'cli_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# COLORS & OUTPUT
# ============================================================================
class C:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    print(f"\n{C.BOLD}{C.BLUE}{text}{C.END}\n")


def print_section(text):
    print(f"\n{text}:\n")


def print_success(text):
    print(f"{C.GREEN}✓ {text}{C.END}")


def print_error(text):
    print(f"{C.RED}✗ {text}{C.END}")


def print_info(text):
    print(f"  {text}")


# ============================================================================
# INTERACTIVE MENU
# ============================================================================
def show_menu():
    """Display interactive menu"""
    print_header("FOOD FRESHNESS DETECTION - INTERACTIVE MENU")
    print("Available options:\n")
    print("  1. Quick Prediction (Single Image)")
    print("  2. Quick Prediction (Multiple Images)")
    print("  3. Layer-by-Layer Visualization")
    print("  4. Model Information")
    print("  5. Print Layer Weights")
    print("  6. Evaluate Model on Test Set")
    print("  7. Exit\n")


def run_menu(model, device):
    """Run interactive menu loop"""
    while True:
        show_menu()
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            # Single image prediction
            while True:
                print("\nAvailable samples in Samples folder:")
                for f in sorted(SAMPLE_DIR.glob('*')):
                    print(f"  {f.name}")
                sample = input("\nEnter sample image name (default: fresh_apple_1.jpg): ").strip() or "fresh_apple_1.jpg"
                
                class Args:
                    pass
                args = Args()
                args.sample = sample
                args.batch = False
                args.save = False
                cmd_predict(args, model, device)
                
                continue_choice = input("\nContinue option 1? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
            
        elif choice == '2':
            # Multiple images prediction
            while True:
                class Args:
                    pass
                args = Args()
                args.sample = None
                args.batch = True
                args.save_opt = input("Save results to CSV? (y/n): ").strip().lower() == 'y'
                args.save = args.save_opt
                cmd_predict(args, model, device)
                
                continue_choice = input("\nContinue option 2? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
            
        elif choice == '3':
            # Layer visualization
            while True:
                print("\nAvailable samples in Samples folder:")
                for f in sorted(SAMPLE_DIR.glob('*')):
                    print(f"  {f.name}")
                sample = input("\nEnter sample image name (default: fresh_apple_1.jpg): ").strip() or "fresh_apple_1.jpg"
                
                class Args:
                    pass
                args = Args()
                args.sample = sample
                cmd_layers(args, model, device)
                
                continue_choice = input("\nContinue option 3? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
            
        elif choice == '4':
            # Model info
            while True:
                class Args:
                    pass
                args = Args()
                cmd_info(args, model, device)
                
                continue_choice = input("\nContinue option 4? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
            
        elif choice == '5':
            # Print layer weights
            while True:
                print_header("PRINT LAYER WEIGHTS")
                print("Available layers: conv1, conv2, conv3, dense1, dense2")
                print("Or type 'all' to view all layers\n")
                layer_name = input("Enter layer name: ").strip().lower()
                
                if layer_name == 'all':
                    for layer in ['conv1', 'conv2', 'conv3', 'dense1', 'dense2']:
                        print_layer_weights(layer, model)
                        input("Press Enter to continue to next layer...")
                else:
                    print_layer_weights(layer_name, model)
                
                continue_choice = input("\nContinue option 5? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
            
        elif choice == '6':
            # Evaluate on custom folder within Codebase
            while True:
                print("\nEnter folder path to evaluate (relative to Codebase):")
                print(f"Examples: Dataset/test, Samples, or any subfolder")
                folder_input = input("Folder path (default: Dataset/test): ").strip() or "Dataset/test"
                
                class Args:
                    pass
                args = Args()
                args.folder = folder_input
                cmd_evaluate(args, model, device)
                
                continue_choice = input("\nContinue option 6? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
            
        elif choice == '7':
            # Exit with confirmation
            confirm = input("\nAre you sure you want to exit? (y/n): ").strip().lower()
            if confirm == 'y':
                print("\nThank you for using Food Freshness Detection CLI!")
                sys.exit(0)
            else:
                print("Returning to menu...")
                continue
            
        else:
            print_error("Invalid choice. Please enter 1-7.")


# ============================================================================
# CORE FUNCTIONS
# ============================================================================
def load_model(device='cpu'):
    """Load the trained model"""
    try:
        if not MODEL_PATH.exists():
            print_error(f"Model not found at {MODEL_PATH}")
            sys.exit(1)
        
        model = FoodFreshnessDetectionCNN().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        
        print_success(f"Model loaded successfully")
        print_info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    except Exception as e:
        print_error(f"Failed to load model: {str(e)}")
        sys.exit(1)


def get_layer_stats(tensor):
    """Calculate statistics for a layer"""
    t = tensor.detach().cpu().numpy().flatten()
    return {
        'mean': np.mean(t),
        'std': np.std(t),
        'min': np.min(t),
        'max': np.max(t),
        'sum': np.sum(t),
        'non_zero': np.count_nonzero(t),
    }


def print_layer_info(layer_name, shape, params_count, stats, weights=None):
    """Print layer information"""
    print(f"\n{layer_name}")
    print(f"  Shape: {shape}")
    print(f"  Parameters: {params_count:,}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  Min: {stats['min']:.6f}")
    print(f"  Max: {stats['max']:.6f}")
    print(f"  Non-Zero: {stats['non_zero']}")
    
    # Print weight stats if provided
    if weights is not None:
        w_flat = weights.detach().cpu().numpy().flatten()
        print(f"  Weight Mean: {np.mean(w_flat):.8f}")
        print(f"  Weight Std: {np.std(w_flat):.8f}")
        print(f"  Sample values: {w_flat[:3]}")


# ============================================================================
# PREDICT COMMAND
# ============================================================================
def cmd_predict(args, model, device):
    """Predict on sample images from test_images folder"""
    print_header("PREDICTION")
    
    if args.batch:
        # Batch mode - all samples
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(SAMPLE_DIR.glob(f'*{ext}'))
            image_files.extend(SAMPLE_DIR.glob(f'*{ext.upper()}'))
        
        if not image_files:
            print_error(f"No images found in {SAMPLE_DIR}")
            return
        
        print_info(f"Found {len(image_files)} sample images")
        
        results = []
        for img_path in tqdm(image_files, desc="Processing"):
            result = predict_single(img_path, model, device, verbose=False)
            if result:
                results.append(result)
        
        # Summary
        print_section("Batch Summary")
        fresh_count = sum(1 for r in results if r['prediction'] == 'FRESH')
        spoiled_count = sum(1 for r in results if r['prediction'] == 'SPOILED')
        
        print_info(f"Total: {len(results)}")
        print_info(f"Fresh: {fresh_count} ({fresh_count/len(results)*100:.1f}%)")
        print_info(f"Spoiled: {spoiled_count} ({spoiled_count/len(results)*100:.1f}%)")
        
        # Save CSV if requested
        if args.save:
            df = pd.DataFrame(results)
            csv_path = OUTPUT_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(csv_path, index=False)
            print_success(f"Results saved to {csv_path}")
    else:
        # Single sample
        sample_file = SAMPLE_DIR / args.sample
        if not sample_file.exists():
            print_error(f"Sample not found: {args.sample}")
            print_info(f"Available samples in {SAMPLE_DIR.name}:")
            for f in sorted(SAMPLE_DIR.glob('*')):
                print_info(f"  {f.name}")
            return
        
        predict_single(str(sample_file), model, device, verbose=True)


def predict_single(image_path, model, device, verbose=True):
    """Predict single image"""
    try:
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = Image.open(img_path).convert('RGB')
        transform = get_transforms(phase='test')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1).item()
        
        fresh_prob = probabilities[0, 0].item()
        spoiled_prob = probabilities[0, 1].item()
        class_names = ['FRESH', 'SPOILED']
        predicted_class = class_names[prediction]
        confidence = max(fresh_prob, spoiled_prob)
        
        result = {
            'image': str(img_path.name),
            'prediction': predicted_class,
            'confidence': confidence,
            'fresh_probability': fresh_prob,
            'spoiled_probability': spoiled_prob,
            'timestamp': datetime.now().isoformat()
        }
        
        if verbose:
            print_info(f"Image: {img_path.name}")
            print_info(f"Prediction: {C.BOLD}{predicted_class}{C.END}")
            print_info(f"Confidence: {confidence*100:.2f}%")
            print_info(f"  - Fresh:   {fresh_prob*100:6.2f}%")
            print_info(f"  - Spoiled: {spoiled_prob*100:6.2f}%")
        
        return result
    
    except Exception as e:
        print_error(f"Prediction failed: {str(e)}")
        return None


# ============================================================================
# LAYERS COMMAND
# ============================================================================
def cmd_layers(args, model, device):
    """Layer-by-layer visualization using sample images"""
    print_header("LAYER-BY-LAYER VISUALIZATION")
    
    sample_file = SAMPLE_DIR / args.sample
    if not sample_file.exists():
        print_error(f"Sample not found: {args.sample}")
        print_info(f"Available samples in {SAMPLE_DIR.name}:")
        for f in sorted(SAMPLE_DIR.glob('*')):
            print_info(f"  {f.name}")
        return
    
    image_path = str(sample_file)
    
    print(f"{C.BOLD}Image:{C.END} {Path(image_path).name}\n")
    
    # Load image
    img = Image.open(image_path).convert('RGB')
    transform = get_transforms(phase='test')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # INPUT
    print(f"\nINPUT IMAGE")
    print(f"  Shape: {tuple(img_tensor.shape)}")
    print(f"  Channels: 3 (RGB)")
    print(f"  Dimensions: 224 x 224")
    print(f"  Device: {device}")
    input_stats = get_layer_stats(img_tensor)
    print(f"  Mean: {input_stats['mean']:.6f}")
    print(f"  Std: {input_stats['std']:.6f}")
    print(f"  Min: {input_stats['min']:.6f}")
    print(f"  Max: {input_stats['max']:.6f}")
    
    # LAYER 1
    x = model.conv1(img_tensor)
    shape = tuple(x.shape)
    params = sum(p.numel() for p in model.conv1.parameters())
    stats = get_layer_stats(x)
    print_layer_info("1. CONVOLUTION (3->8)", shape, params, stats, model.conv1.weight)
    
    # LAYER 2
    x = model.relu1(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("2. ACTIVATION (ReLU)", shape, 0, stats)
    
    # LAYER 3
    x = model.maxpool1(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("3. POOLING (2x2)", shape, 0, stats)
    print("  Size: 224x224 -> 112x112")
    
    # LAYER 4
    x = model.conv2(x)
    shape = tuple(x.shape)
    params = sum(p.numel() for p in model.conv2.parameters())
    stats = get_layer_stats(x)
    print_layer_info("4. CONVOLUTION (8->16)", shape, params, stats, model.conv2.weight)
    
    # LAYER 5
    x = model.relu2(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("5. ACTIVATION (ReLU)", shape, 0, stats)
    
    # LAYER 6
    x = model.maxpool2(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("6. POOLING (2x2)", shape, 0, stats)
    print("  Size: 112x112 -> 56x56")
    
    # LAYER 7
    x = model.conv3(x)
    shape = tuple(x.shape)
    params = sum(p.numel() for p in model.conv3.parameters())
    stats = get_layer_stats(x)
    print_layer_info("7. CONVOLUTION (16->32)", shape, params, stats, model.conv3.weight)
    
    # LAYER 8
    x = model.relu3(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("8. ACTIVATION (ReLU)", shape, 0, stats)
    
    # LAYER 9
    x = model.maxpool3(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("9. POOLING (2x2)", shape, 0, stats)
    print("  Size: 56x56 -> 28x28")
    
    # LAYER 10
    x = model.flatten(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("10. FLATTEN", shape, 0, stats)
    print("  Flattened: 32 x 28 x 28 = 25,088")
    
    # LAYER 11
    x = model.dense1(x)
    shape = tuple(x.shape)
    params = sum(p.numel() for p in model.dense1.parameters())
    stats = get_layer_stats(x)
    print_layer_info("11. DENSE (25088→128)", shape, params, stats)
    
    # LAYER 12
    x = model.relu4(x)
    shape = tuple(x.shape)
    stats = get_layer_stats(x)
    print_layer_info("12. ACTIVATION (ReLU)", shape, 0, stats)
    
    # LAYER 13
    x = model.dropout(x)
    
    # LAYER 14
    logits = model.dense2(x)
    shape = tuple(logits.shape)
    params = sum(p.numel() for p in model.dense2.parameters())
    stats = get_layer_stats(logits)
    print_layer_info("13. OUTPUT (128→2)", shape, params, stats)
    
    # PREDICTION
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(logits, dim=1).item()
    class_names = ['FRESH', 'SPOILED']
    
    print(f"\nFINAL PREDICTION")
    print(f"  Class: {class_names[pred]}")
    print(f"  Confidence: {probs[0, pred].item()*100:.2f}%")
    print(f"  Fresh: {probs[0, 0].item()*100:.2f}%")
    print(f"  Spoiled: {probs[0, 1].item()*100:.2f}%")


# ============================================================================
# INFO COMMAND
# ============================================================================
def print_architecture(model):
    """Print formatted model architecture"""
    print("\nFoodFreshnessDetectionCNN(")
    print("  (conv1): Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))")
    print("  (relu1): ReLU(inplace=True)")
    print("  (maxpool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)")
    print("  (conv2): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))")
    print("  (relu2): ReLU(inplace=True)")
    print("  (maxpool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)")
    print("  (conv3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))")
    print("  (relu3): ReLU(inplace=True)")
    print("  (maxpool3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)")
    print("  (flatten): Flatten(start_dim=1, end_dim=-1)")
    print("  (dense1): Linear(in_features=25088, out_features=128, bias=True)")
    print("  (relu4): ReLU(inplace=True)")
    print("  (dropout): Dropout(p=0.3, inplace=False)")
    print("  (dense2): Linear(in_features=128, out_features=2, bias=True)")
    print(")\n")


def print_layer_weights(layer_name, model):
    """Print all layer weights to console in detailed format"""
    layer_map = {
        'conv1': ('Conv1', model.conv1, (3, 8)),
        'conv2': ('Conv2', model.conv2, (8, 16)),
        'conv3': ('Conv3', model.conv3, (16, 32)),
        'dense1': ('Dense1', model.dense1, (25088, 128)),
        'dense2': ('Dense2', model.dense2, (128, 2)),
    }
    
    if layer_name not in layer_map:
        print_error(f"Layer not found. Available: {', '.join(layer_map.keys())}")
        print_info("Examples: conv1, conv2, conv3, dense1, dense2")
        return
    
    layer_display_name, layer, shape_info = layer_map[layer_name]
    weights = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy() if hasattr(layer, 'bias') and layer.bias is not None else None
    
    print(f"\n{C.BOLD}{C.BLUE}{'='*80}{C.END}")
    print(f"{C.BOLD}{layer_display_name} - Layer Weights{C.END}")
    print(f"{C.BOLD}{C.BLUE}{'='*80}{C.END}")
    
    print(f"\n{C.BOLD}Shape Information:{C.END}")
    print(f"  Weight Shape: {weights.shape}")
    print(f"  Total Parameters: {weights.size:,}")
    
    if len(weights.shape) == 4:
        # Conv layer
        print(f"  Output Channels: {weights.shape[0]}")
        print(f"  Input Channels: {weights.shape[1]}")
        print(f"  Kernel Height: {weights.shape[2]}")
        print(f"  Kernel Width: {weights.shape[3]}")
    else:
        # Dense layer
        print(f"  Output Features: {weights.shape[0]}")
        print(f"  Input Features: {weights.shape[1]}")
    
    print(f"\n{C.BOLD}Weight Statistics:{C.END}")
    print(f"  Mean:  {np.mean(weights):12.8f}")
    print(f"  Std:   {np.std(weights):12.8f}")
    print(f"  Min:   {np.min(weights):12.8f}")
    print(f"  Max:   {np.max(weights):12.8f}")
    
    if bias is not None:
        print(f"\n{C.BOLD}Bias Statistics:{C.END}")
        print(f"  Mean:  {np.mean(bias):12.8f}")
        print(f"  Std:   {np.std(bias):12.8f}")
        print(f"  Min:   {np.min(bias):12.8f}")
        print(f"  Max:   {np.max(bias):12.8f}")
    
    # Display weights
    print(f"\n{C.BOLD}Weight Values:{C.END}")
    
    if len(weights.shape) == 4:
        # Conv layer - show each filter
        for out_ch in range(min(3, weights.shape[0])):  # Show first 3 filters
            print(f"\n  {C.BOLD}Filter {out_ch} (Output Channel {out_ch}):{C.END}")
            for in_ch in range(weights.shape[1]):
                kernel = weights[out_ch, in_ch]  # 3x3
                print(f"    Input Channel {in_ch}:")
                for row_idx, row in enumerate(kernel):
                    row_str = "      " + "  ".join([f"{val:7.4f}" for val in row])
                    print(row_str)
        
        if weights.shape[0] > 3:
            print(f"\n  {C.YELLOW}... ({weights.shape[0] - 3} more filters) ...{C.END}")
    
    else:
        # Dense layer - show first few connections
        print(f"\n  {C.BOLD}Weight Matrix (rows=outputs, cols=inputs):{C.END}")
        display_rows = min(5, weights.shape[0])
        display_cols = min(10, weights.shape[1])
        
        # Column headers
        print("    " + "".join([f"{i:8d}" for i in range(display_cols)]))
        
        # Rows
        for out_idx in range(display_rows):
            row_str = f"  {out_idx}: " + "".join([f"{weights[out_idx, in_idx]:8.4f}" for in_idx in range(display_cols)])
            print(row_str)
        
        if weights.shape[0] > display_rows or weights.shape[1] > display_cols:
            extra_rows = weights.shape[0] - display_rows
            extra_cols = weights.shape[1] - display_cols
            print(f"\n    {C.YELLOW}... ({extra_rows} more rows, {extra_cols} more columns) ...{C.END}")
    
    # Display bias
    if bias is not None:
        print(f"\n{C.BOLD}Bias Values:{C.END}")
        display_bias = min(10, len(bias))
        bias_str = "  " + "  ".join([f"{b:7.4f}" for b in bias[:display_bias]])
        print(bias_str)
        if len(bias) > display_bias:
            print(f"  ... ({len(bias) - display_bias} more bias terms) ...")
    
    print(f"\n{C.BOLD}{C.BLUE}{'='*80}{C.END}\n")


def cmd_info(args, model, device):
    """Model information"""
    print_header("MODEL INFORMATION")
    
    print_info(f"Model: {model.__class__.__name__}")
    print_info(f"Device: {device}")
    print_info(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print_section("Architecture")
    print_architecture(model)
    
    print_section("Layer Details")
    table_data = []
    for name, param in model.named_parameters():
        layer_params = param.numel()
        table_data.append([name, tuple(param.shape), layer_params])
    
    headers = ['Layer', 'Shape', 'Parameters']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))


# ============================================================================
# EVALUATE COMMAND
# ============================================================================
def cmd_evaluate(args, model, device):
    """Evaluate on custom folder within Codebase"""
    print_header("MODEL EVALUATION")
    
    from utils import FoodFreshnessDataset
    
    # Get folder path from args or use default
    folder_path = getattr(args, 'folder', 'Dataset/test')
    
    # Construct full path (relative to Codebase directory)
    test_dir = Path(__file__).parent / folder_path
    
    # Validate path exists
    if not test_dir.exists():
        print_error(f"Folder not found: {test_dir}")
        return
    
    # Security check: ensure path is within project structure
    try:
        test_dir.resolve().relative_to(Path(__file__).parent.resolve())
    except ValueError:
        print_error(f"Path must be within Codebase directory")
        return
    
    print_section("Loading Dataset")
    print_info(f"Folder: {test_dir.name}")
    
    test_labels_csv = PROJECT_ROOT / 'test_labels.csv'
    
    if not test_labels_csv.exists():
        print_error(f"Labels file not found: {test_labels_csv}")
        return
    
    test_labels = pd.read_csv(test_labels_csv)
    print_info(f"Total images: {len(test_labels)}")
    print_info(f"Fresh: {(test_labels['label'] == 0).sum()}")
    print_info(f"Spoiled: {(test_labels['label'] == 1).sum()}")
    
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
    
    print_section("Evaluating")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print_section("Metrics")
    metrics_table = [
        ['Accuracy', f"{accuracy:.4f} ({accuracy*100:.2f}%)"],
        ['Precision', f"{precision:.4f}"],
        ['Recall', f"{recall:.4f}"],
        ['F1-Score', f"{f1:.4f}"],
    ]
    print(tabulate(metrics_table, tablefmt='grid'))


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Food Freshness Detection CLI', 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--menu', action='store_true', help='Run interactive menu')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # PREDICT
    predict_parser = subparsers.add_parser('predict', help='Predict on sample images')
    predict_parser.add_argument('--sample', default='fresh_apple_1.jpg', help='Sample image name (from Samples folder)')
    predict_parser.add_argument('--batch', action='store_true', help='Batch mode (all samples)')
    predict_parser.add_argument('--save', action='store_true', help='Save to CSV')
    
    # LAYERS
    layers_parser = subparsers.add_parser('layers', help='Layer visualization')
    layers_parser.add_argument('--sample', default='fresh_apple_1.jpg', help='Sample image name (from Samples folder)')
    
    # INFO
    info_parser = subparsers.add_parser('info', help='Model info')
    
    # EVALUATE
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    evaluate_parser.add_argument('--folder', default='Dataset/test', help='Folder path within Codebase to evaluate (default: Dataset/test)')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model = load_model(device)
    
    # If no command specified or --menu flag, run interactive menu
    if not args.command or args.menu:
        run_menu(model, device)
        return
    
    try:
        if args.command == 'predict':
            cmd_predict(args, model, device)
        elif args.command == 'layers':
            cmd_layers(args, model, device)
        elif args.command == 'info':
            cmd_info(args, model, device)
        elif args.command == 'evaluate':
            cmd_evaluate(args, model, device)
        
        print_success("\nCommand completed!")
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
