"""
Dataset Split Script
Splits dataset 70/30 into train/ and test/ folders
Labels stored in CSV files, metadata stored in json/ folder
"""

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import csv
import json
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_ROOT = PROJECT_ROOT / 'dataset'
DATASET_DIR = PROJECT_ROOT / 'Dataset'
TRAIN_DIR = DATASET_DIR / 'train'
TEST_DIR = DATASET_DIR / 'test'
JSON_DIR = PROJECT_ROOT / 'json'

# Category to label mapping
CATEGORY_LABELS = {
    'fresh_fruits': 0,
    'fresh_vegetables': 0,
    'fresh_bread': 0,
    'fresh_dairy': 0,
    'spoiled_fruits': 1,
    'spoiled_vegetables': 1,
    'spoiled_bread': 1,
    'spoiled_dairy': 1
}

class DatasetSplitter:
    def __init__(self):
        self.split_info = {
            'timestamp': datetime.now().isoformat(),
            'total_images': 0,
            'train_count': 0,
            'test_count': 0,
            'by_category': {}
        }
        self.train_data = []
        self.test_data = []
    
    def collect_images(self):
        """Collect all images and their labels"""
        print("=" * 70)
        print("COLLECTING IMAGES FROM DATASET")
        print("=" * 70)
        
        # Dictionary to store images by category
        category_images = {}
        
        for category, label in CATEGORY_LABELS.items():
            category_path = DATASET_ROOT / category
            
            if not category_path.exists():
                print(f"⚠️  Category folder not found: {category}")
                continue
            
            images = list(category_path.glob('*.jpg')) + list(category_path.glob('*.jpeg')) + \
                    list(category_path.glob('*.png'))
            
            category_images[category] = {
                'label': label,
                'images': images
            }
            
            print(f"✓ {category}: {len(images)} images (Label: {label})")
            self.split_info['by_category'][category] = {
                'total': len(images),
                'label': label,
                'train': 0,
                'test': 0
            }
            
            self.split_info['total_images'] += len(images)
        
        return category_images
    
    def split_by_category(self, category_images):
        """Split each category 70/30"""
        print("\n" + "=" * 70)
        print("SPLITTING BY CATEGORY (70% train, 30% test)")
        print("=" * 70)
        
        for category, data in category_images.items():
            images = data['images']
            label = data['label']
            
            # Split 70/30 per category
            train_images, test_images = train_test_split(
                images,
                test_size=0.30,
                random_state=42
            )
            
            # Store with labels
            for img_path in train_images:
                self.train_data.append((img_path, label))
            
            for img_path in test_images:
                self.test_data.append((img_path, label))
            
            self.split_info['by_category'][category]['train'] = len(train_images)
            self.split_info['by_category'][category]['test'] = len(test_images)
            
            print(f"{category}: {len(train_images)} train | {len(test_images)} test")
        
        self.split_info['train_count'] = len(self.train_data)
        self.split_info['test_count'] = len(self.test_data)
    
    def create_directories(self):
        """Create train/ and test/ directories"""
        print("\n" + "=" * 70)
        print("CREATING DIRECTORIES")
        print("=" * 70)
        
        TRAIN_DIR.mkdir(parents=True, exist_ok=True)
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        JSON_DIR.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Created: {TRAIN_DIR}")
        print(f"✓ Created: {TEST_DIR}")
        print(f"✓ Created: {JSON_DIR}")
    
    def copy_images(self):
        """Copy images to train/ and test/ folders"""
        print("\n" + "=" * 70)
        print("COPYING IMAGES")
        print("=" * 70)
        
        # Copy train images
        print(f"\nCopying {len(self.train_data)} training images...")
        for i, (img_path, label) in enumerate(self.train_data):
            dest = TRAIN_DIR / img_path.name
            shutil.copy2(img_path, dest)
            
            if (i + 1) % 500 == 0:
                print(f"  ✓ Copied {i + 1} / {len(self.train_data)} images")
        
        print(f"✓ Completed: {len(self.train_data)} train images")
        
        # Copy test images
        print(f"\nCopying {len(self.test_data)} test images...")
        for i, (img_path, label) in enumerate(self.test_data):
            dest = TEST_DIR / img_path.name
            shutil.copy2(img_path, dest)
            
            if (i + 1) % 200 == 0:
                print(f"  ✓ Copied {i + 1} / {len(self.test_data)} images")
        
        print(f"✓ Completed: {len(self.test_data)} test images")
    
    def save_labels(self):
        """Save labels to CSV files"""
        print("\n" + "=" * 70)
        print("SAVING LABELS TO CSV")
        print("=" * 70)
        
        # Save train labels
        train_csv = PROJECT_ROOT / 'train_labels.csv'
        with open(train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for img_path, label in self.train_data:
                writer.writerow([img_path.name, label])
        
        print(f"✓ Saved: {train_csv}")
        print(f"  Format: filename, label")
        print(f"  Lines: {len(self.train_data)}")
        
        # Save test labels
        test_csv = PROJECT_ROOT / 'test_labels.csv'
        with open(test_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'label'])
            for img_path, label in self.test_data:
                writer.writerow([img_path.name, label])
        
        print(f"✓ Saved: {test_csv}")
        print(f"  Format: filename, label")
        print(f"  Lines: {len(self.test_data)}")
    
    def save_split_info(self):
        """Save detailed split information to json/ folder"""
        split_json = JSON_DIR / 'split_info.json'
        
        with open(split_json, 'w') as f:
            json.dump(self.split_info, f, indent=2)
        
        print(f"✓ Saved: {split_json}")
    
    def generate_report(self):
        """Generate summary report"""
        print("\n" + "=" * 70)
        print("SPLIT SUMMARY REPORT")
        print("=" * 70)
        
        print(f"\nTotal images: {self.split_info['total_images']}")
        print(f"Train images: {self.split_info['train_count']} (70%)")
        print(f"Test images: {self.split_info['test_count']} (30%)")
        
        # Count by label
        train_fresh = sum(1 for _, label in self.train_data if label == 0)
        train_spoiled = sum(1 for _, label in self.train_data if label == 1)
        test_fresh = sum(1 for _, label in self.test_data if label == 0)
        test_spoiled = sum(1 for _, label in self.test_data if label == 1)
        
        print(f"\nTrain set balance:")
        print(f"  Fresh (0):  {train_fresh} ({100*train_fresh/len(self.train_data):.1f}%)")
        print(f"  Spoiled (1): {train_spoiled} ({100*train_spoiled/len(self.train_data):.1f}%)")
        
        print(f"\nTest set balance:")
        print(f"  Fresh (0):  {test_fresh} ({100*test_fresh/len(self.test_data):.1f}%)")
        print(f"  Spoiled (1): {test_spoiled} ({100*test_spoiled/len(self.test_data):.1f}%)")
        
        print(f"\n--- LABEL MAPPING ---")
        print(f"0 = FRESH")
        print(f"1 = SPOILED")
        
        print(f"\n--- FILES CREATED ---")
        print(f"Folder: {TRAIN_DIR}")
        print(f"  Contains: {self.split_info['train_count']} images")
        print(f"\nFolder: {TEST_DIR}")
        print(f"  Contains: {self.split_info['test_count']} images")
        print(f"\nFile: {PROJECT_ROOT / 'train_labels.csv'}")
        print(f"  Maps: image filename → label")
        print(f"\nFile: {PROJECT_ROOT / 'test_labels.csv'}")
        print(f"  Maps: image filename → label")
        print(f"\nFile: {JSON_DIR / 'split_info.json'}")
        print(f"  Detailed split statistics")
    
    def run(self):
        """Run full split pipeline"""
        try:
            # Collect
            category_images = self.collect_images()
            
            if not category_images:
                print("❌ No images found!")
                return False
            
            # Split
            self.split_by_category(category_images)
            
            # Create directories
            self.create_directories()
            
            # Copy images
            self.copy_images()
            
            # Save labels
            self.save_labels()
            
            # Save info
            self.save_split_info()
            
            # Report
            self.generate_report()
            
            return True
        
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False


if __name__ == "__main__":
    splitter = DatasetSplitter()
    success = splitter.run()
    
    if success:
        print("\n" + "=" * 70)
        print("✓ DATASET SPLIT COMPLETE")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ DATASET SPLIT FAILED")
        print("=" * 70)
