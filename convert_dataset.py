#!/usr/bin/env python
"""
Dataset Conversion Utility for TB Detection

This script helps convert TB X-ray datasets into the format required by our 
TB Detection system, which expects:
- Normal X-rays in data/train/normal/
- TB X-rays in data/train/tb/
- Test X-rays in data/test/

Usage:
    python convert_dataset.py --source_dir SOURCE_DIR --target_dir data
    python convert_dataset.py --source_dir SOURCE_DIR --target_dir data --split 0.8 0.2
"""

import os
import argparse
import shutil
import random
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert TB X-ray dataset format')
    
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Source directory containing the original dataset')
    parser.add_argument('--target_dir', type=str, default='data',
                        help='Target directory for the converted dataset')
    parser.add_argument('--split', type=float, nargs=2, default=[0.8, 0.2],
                        help='Train/test split ratio (e.g., 0.8 0.2)')
    parser.add_argument('--format', type=str, choices=['folders', 'csv', 'auto'], default='auto',
                        help='Source dataset format: folders (Normal/TB), CSV (with path and label), or auto-detect')
    parser.add_argument('--csv_path_col', type=str, default='path',
                        help='Column name for image path in CSV (if format=csv)')
    parser.add_argument('--csv_label_col', type=str, default='label',
                        help='Column name for label in CSV (if format=csv)')
    parser.add_argument('--tb_keywords', type=str, nargs='+', default=['tb', 'tuberculosis'],
                        help='Keywords that indicate TB positive in folder names')
    parser.add_argument('--normal_keywords', type=str, nargs='+', default=['normal', 'healthy'],
                        help='Keywords that indicate Normal in folder names')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def create_target_directories(target_dir):
    """Create the target directory structure."""
    # Create main directories
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train', 'normal'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'train', 'tb'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)
    
    print(f"Created directory structure in {target_dir}")

def detect_format(source_dir, tb_keywords, normal_keywords):
    """Auto-detect the dataset format."""
    # Check if there's a CSV file with potential labels
    csv_files = glob(os.path.join(source_dir, '*.csv'))
    if csv_files:
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Check if the CSV has common label column names
            for col in ['label', 'class', 'target', 'diagnosis', 'category']:
                if col in df.columns:
                    print(f"Detected CSV format with label column: {col}")
                    return 'csv', csv_file
    
    # Check if there are folders that might indicate classes
    subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    for keyword in tb_keywords + normal_keywords:
        for subdir in subdirs:
            if keyword.lower() in subdir.lower():
                print(f"Detected folders format with keyword '{keyword}' in folder name '{subdir}'")
                return 'folders', None
    
    # Default to folders if nothing detected
    print("Could not confidently detect format, defaulting to 'folders'")
    return 'folders', None

def convert_from_folders(source_dir, target_dir, split, tb_keywords, normal_keywords, random_seed):
    """Convert dataset from folder-based structure."""
    # Collect all image files
    image_extensions = ['.png', '.jpg', '.jpeg']
    tb_images = []
    normal_images = []
    
    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        folder_name = os.path.basename(root).lower()
        
        # Determine class based on folder name
        is_tb_folder = any(keyword in folder_name for keyword in tb_keywords)
        is_normal_folder = any(keyword in folder_name for keyword in normal_keywords)
        
        if not (is_tb_folder or is_normal_folder):
            continue
        
        # Collect images
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(root, file)
                
                if is_tb_folder:
                    tb_images.append(img_path)
                elif is_normal_folder:
                    normal_images.append(img_path)
    
    # Split the datasets
    random.seed(random_seed)
    
    train_ratio, test_ratio = split
    train_test_boundary = train_ratio / (train_ratio + test_ratio)
    
    random.shuffle(tb_images)
    random.shuffle(normal_images)
    
    tb_train_count = int(len(tb_images) * train_test_boundary)
    normal_train_count = int(len(normal_images) * train_test_boundary)
    
    tb_train_images = tb_images[:tb_train_count]
    tb_test_images = tb_images[tb_train_count:]
    
    normal_train_images = normal_images[:normal_train_count]
    normal_test_images = normal_images[normal_train_count:]
    
    # Copy the files
    for src_path in tb_train_images:
        dest_path = os.path.join(target_dir, 'train', 'tb', os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
    
    for src_path in normal_train_images:
        dest_path = os.path.join(target_dir, 'train', 'normal', os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
    
    # For test images, rename them to include a TB identifier (for submission format)
    for i, src_path in enumerate(tb_test_images + normal_test_images):
        # Create a file name like TB_test_0001.png
        dest_filename = f"TB_test_{i+1:04d}{os.path.splitext(src_path)[1]}"
        dest_path = os.path.join(target_dir, 'test', dest_filename)
        shutil.copy2(src_path, dest_path)
    
    # Create a hidden ground truth file for testing
    test_labels = [1] * len(tb_test_images) + [0] * len(normal_test_images)
    test_ids = [f"TB_test_{i+1:04d}" for i in range(len(test_labels))]
    
    # Save the ground truth as a CSV
    gt_df = pd.DataFrame({
        'ID': test_ids,
        'Target': test_labels
    })
    gt_df.to_csv(os.path.join(target_dir, 'test_ground_truth.csv'), index=False)
    
    # Print summary
    print(f"Converted dataset: {len(tb_train_images)} TB train images, {len(normal_train_images)} normal train images")
    print(f"Test set: {len(tb_test_images) + len(normal_test_images)} images")
    print(f"Ground truth saved to {os.path.join(target_dir, 'test_ground_truth.csv')}")

def convert_from_csv(source_dir, csv_path, target_dir, split, path_col, label_col, random_seed):
    """Convert dataset from CSV-based structure."""
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure required columns exist
    if path_col not in df.columns:
        raise ValueError(f"Path column '{path_col}' not found in CSV")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")
    
    # Make paths absolute if they're relative
    if not os.path.isabs(df[path_col].iloc[0]):
        csv_dir = os.path.dirname(csv_path)
        df[path_col] = df[path_col].apply(lambda x: os.path.join(csv_dir, x))
    
    # Split into TB and normal
    tb_df = df[df[label_col] == 1]
    normal_df = df[df[label_col] == 0]
    
    # Split into train and test
    train_ratio, test_ratio = split
    
    tb_train, tb_test = train_test_split(
        tb_df, test_size=test_ratio/(train_ratio+test_ratio),
        random_state=random_seed
    )
    
    normal_train, normal_test = train_test_split(
        normal_df, test_size=test_ratio/(train_ratio+test_ratio),
        random_state=random_seed
    )
    
    # Copy the files
    for _, row in tb_train.iterrows():
        src_path = row[path_col]
        dest_path = os.path.join(target_dir, 'train', 'tb', os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
    
    for _, row in normal_train.iterrows():
        src_path = row[path_col]
        dest_path = os.path.join(target_dir, 'train', 'normal', os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
    
    # For test images, rename them to include a TB identifier (for submission format)
    test_df = pd.concat([tb_test, normal_test])
    test_df = test_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    test_ids = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        src_path = row[path_col]
        # Create a file name like TB_test_0001.png
        dest_filename = f"TB_test_{i+1:04d}{os.path.splitext(src_path)[1]}"
        test_ids.append(f"TB_test_{i+1:04d}")
        dest_path = os.path.join(target_dir, 'test', dest_filename)
        shutil.copy2(src_path, dest_path)
    
    # Create a hidden ground truth file for testing
    gt_df = pd.DataFrame({
        'ID': test_ids,
        'Target': test_df[label_col].values
    })
    gt_df.to_csv(os.path.join(target_dir, 'test_ground_truth.csv'), index=False)
    
    # Print summary
    print(f"Converted dataset: {len(tb_train)} TB train images, {len(normal_train)} normal train images")
    print(f"Test set: {len(test_df)} images")
    print(f"Ground truth saved to {os.path.join(target_dir, 'test_ground_truth.csv')}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create target directories
    create_target_directories(args.target_dir)
    
    # Detect format if auto is specified
    format_type = args.format
    csv_path = None
    
    if format_type == 'auto':
        format_type, csv_path = detect_format(args.source_dir, args.tb_keywords, args.normal_keywords)
    
    # Convert the dataset
    if format_type == 'folders':
        convert_from_folders(
            args.source_dir, args.target_dir, args.split,
            args.tb_keywords, args.normal_keywords, args.random_seed
        )
    elif format_type == 'csv':
        if csv_path is None:
            # Find the first CSV file if not detected
            csv_files = glob(os.path.join(args.source_dir, '*.csv'))
            if not csv_files:
                raise ValueError("No CSV file found in the source directory")
            csv_path = csv_files[0]
        
        convert_from_csv(
            args.source_dir, csv_path, args.target_dir, args.split,
            args.csv_path_col, args.csv_label_col, args.random_seed
        )
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
    
    print("Dataset conversion completed!")

if __name__ == "__main__":
    main() 