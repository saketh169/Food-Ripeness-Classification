"""
Demo Script
Interactive demo for food freshness detection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

from model import FoodFreshnessDetectionCNN
from analyze import FoodAnalyzer


def generate_demo_report(image_path, model_path, device='cpu'):
    """
    Generate complete analysis report for demo
    """
    print("\n" + "="*70)
    print("FOOD FRESHNESS DETECTION - DEMO")
    print("="*70)
    
    # Initialize analyzer
    analyzer = FoodAnalyzer(model_path, device)
    
    # Analyze image
    analysis, saliency, colors, layer_stats = analyzer.analyze_image(image_path)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Load original image
    from PIL import Image
    img = Image.open(image_path)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Saliency map
    ax2 = plt.subplot(2, 3, 2)
    im = ax2.imshow(saliency, cmap='hot')
    ax2.set_title('Saliency Map\n(Important Pixels)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    # 3. Saliency overlay on image
    ax3 = plt.subplot(2, 3, 3)
    img_array = np.array(img)
    saliency_colored = plt.cm.hot(saliency)
    overlay = (img_array / 255.0) * 0.5 + saliency_colored[:, :, :3] * 0.5
    ax3.imshow(overlay)
    ax3.set_title('Saliency Overlay', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Color histogram
    ax4 = plt.subplot(2, 3, 4)
    colors_list = list(colors.keys())
    colors_values = list(colors.values())
    bars = ax4.bar(colors_list, colors_values, color=['green', 'yellow', 'red', 'brown'])
    ax4.set_ylabel('Percentage (%)', fontsize=10)
    ax4.set_title('Color Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 100])
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 5. Prediction result
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    prediction_text = f"""
    PREDICTION RESULT
    
    Class: {analysis['prediction']['class']}
    Confidence: {analysis['prediction']['confidence']:.2%}
    Stage: {analysis['prediction']['stage']}
    
    Fresh Probability: {analysis['prediction']['confidence']:.4f}
    Spoiled Probability: {1 - analysis['prediction']['confidence']:.4f}
    """
    ax5.text(0.1, 0.5, prediction_text, fontsize=11, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Layer activations
    ax6 = plt.subplot(2, 3, 6)
    layers = list(layer_stats.keys())
    activations = [layer_stats[l]['mean_activation'] for l in layers]
    bars = ax6.barh(layers, activations, color='steelblue')
    ax6.set_xlabel('Mean Activation', fontsize=10)
    ax6.set_title('Layer Activations', fontsize=12, fontweight='bold')
    for i, (bar, val) in enumerate(zip(bars, activations)):
        ax6.text(val, bar.get_y() + bar.get_height()/2., f'{val:.3f}',
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(image_path).stem + '_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    
    # Save analysis JSON to json/ folder
    json_dir = Path(__file__).parent.parent / 'json'
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / (Path(image_path).stem + '_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"✓ Analysis saved: {json_path}")
    
    # Display
    plt.show()
    
    return analysis


def batch_demo(image_folder, model_path, device='cpu', max_images=5):
    """
    Run demo on multiple images
    """
    print("\n" + "="*70)
    print(f"BATCH DEMO - Analyzing up to {max_images} images")
    print("="*70)
    
    image_folder = Path(image_folder)
    image_files = list(image_folder.glob('*.jpg')) + \
                  list(image_folder.glob('*.jpeg')) + \
                  list(image_folder.glob('*.png'))
    
    image_files = image_files[:max_images]
    
    all_analyses = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")
        
        try:
            analysis, _, _, _ = FoodAnalyzer(model_path, device).analyze_image(str(image_path))
            all_analyses.append(analysis)
            print(f"✓ Completed")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Summary report
    print("\n" + "="*70)
    print("BATCH SUMMARY")
    print("="*70)
    
    fresh_count = sum(1 for a in all_analyses if a['prediction']['class'] == 'FRESH')
    spoiled_count = sum(1 for a in all_analyses if a['prediction']['class'] == 'SPOILED')
    avg_confidence = np.mean([a['prediction']['confidence'] for a in all_analyses])
    
    print(f"\nTotal images analyzed: {len(all_analyses)}")
    print(f"Fresh: {fresh_count} ({100*fresh_count/len(all_analyses):.1f}%)")
    print(f"Spoiled: {spoiled_count} ({100*spoiled_count/len(all_analyses):.1f}%)")
    print(f"Average confidence: {avg_confidence:.4f}")
    
    # Save batch report
    batch_report = {
        'timestamp': str(datetime.now()),
        'total_images': len(all_analyses),
        'fresh_count': fresh_count,
        'spoiled_count': spoiled_count,
        'average_confidence': float(avg_confidence),
        'analyses': all_analyses
    }
    
    json_dir = Path(__file__).parent.parent / 'json'
    json_dir.mkdir(parents=True, exist_ok=True)
    report_path = json_dir / 'batch_report.json'
    with open(report_path, 'w') as f:
        json.dump(batch_report, f, indent=2, default=str)
    
    print(f"\n✓ Batch report saved: {report_path}")
    
    return batch_report


if __name__ == "__main__":
    print("Demo module loaded successfully!")
    print("\nUsage:")
    print("  Single image: generate_demo_report('path/to/image.jpg', 'model.pth')")
    print("  Batch: batch_demo('path/to/images/', 'model.pth')")
