"""
Filename: stratified_split.py

Description:
    This script loads an existing dataset of images and cleaned JSON annotations,
    groups all items by species name, and performs a *stratified split* into
    training, validation, and test sets using fixed ratios (70% / 20% / 10%).
    
    After splitting, it copies each image and annotation pair into a new
    directory structure under DATA/FINAL/sample_data, preserving the original
    labels and ensuring each species remains proportionally represented.

Notes:
    - Input data must follow this structure:
         DATA/INITIAL/sample_data/<train|test|val>/<img|ann>/
    - Annotations must contain: {"name": "<species>"} after cleaning.
    - Output folders will be created automatically if missing.
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

# Base directory where initial (cleaned) data is stored
INPUT_BASE_DIR = Path(__file__).parent.parent / "DATA" / "INITIAL" / "sample_data"

# Base directory where final stratified split will be written
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "DATA" / "FINAL" / "sample_data"

# Split ratios used for dataset partitioning
SPLIT_RATIOS = {
    "train": 0.70,
    "validate": 0.20,
    "test": 0.10
}

def get_image_files(folder_path):
    """
    Return a sorted list of image filenames from a given directory.

    Only files with standard image extensions are included.
    """
    image_extensions = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'}
    image_files = []

    # Ensure folder exists to avoid errors
    if folder_path.exists():
        for file in folder_path.iterdir():
            # Include file only if its extension matches known image formats
            if file.suffix in image_extensions:
                image_files.append(file.name)

    return sorted(image_files)

def load_all_data():
    """
    Load all images and annotation pairs from INITIAL folders.

    Returns:
        A dictionary mapping species_name -> list of items for that species.
        Each item contains:
            - image filename
            - path to image
            - annotation dictionary
            - path to annotation JSON file
    """
    data_by_species = defaultdict(list)
    
    # Source folders expected to exist
    folders = ["test", "train", "val"]
    
    for folder in folders:
        # Paths to image and annotation folders
        img_folder = INPUT_BASE_DIR / folder / "img"
        ann_folder = INPUT_BASE_DIR / folder / "ann"
        
        # Skip folders that are missing
        if not img_folder.exists() or not ann_folder.exists():
            print(f"Warning: {folder} folder not found, skipping...")
            continue
        
        # Get list of image filenames
        image_files = get_image_files(img_folder)
        
        for img_file in image_files:
            # Derive annotation file name (image.ext + ".json")
            ann_file = ann_folder / (img_file + ".json")
            
            # Skip missing annotations
            if not ann_file.exists():
                print(f"Warning: Annotation file not found for {img_file}, skipping...")
                continue
            
            try:
                # Load cleaned annotation
                with open(ann_file, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                    
                    # Extract species name ("name" field from cleaned JSON)
                    species_name = ann_data.get('name', '')
                    
                    # Only keep data entries where species name exists
                    if species_name:
                        data_by_species[species_name].append({
                            'image': img_file,
                            'image_path': img_folder / img_file,
                            'annotation': ann_data,
                            'annotation_path': ann_file
                        })
            
            except (json.JSONDecodeError, KeyError) as e:
                # Warn but continue processing other files
                print(f"Warning: Could not read annotation {ann_file}: {e}")
                continue
    
    return data_by_species

def stratified_split(data_by_species, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10, random_seed=42):
    """
    Perform stratified dataset splitting, ensuring each species appears in the
    train/validate/test sets according to given ratios.

    Args:
        data_by_species: dict species_name -> list of items
        train_ratio: percentage allocated to training
        val_ratio: percentage allocated to validation
        test_ratio: percentage allocated to testing
        random_seed: ensures reproducible shuffling

    Returns:
        splits: dict containing:
            - 'train' : list of items
            - 'validate' : list of items
            - 'test' : list of items
    """
    random.seed(random_seed)
    
    # Initialize dictionary to store the resulting splits
    splits = {
        'train': [],
        'validate': [],
        'test': []
    }
    
    # Validate that user-provided ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    print(f"\nPerforming stratified split (train={train_ratio:.0%}, validate={val_ratio:.0%}, test={test_ratio:.0%})...")
    print(f"Total species: {len(data_by_species)}")
    
    # Loop over each species for per-species proportional splitting
    for species, items in data_by_species.items():
        # Shuffle items belonging to this species
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)
        
        total_count = len(shuffled_items)
        
        # Compute indices where splits occur
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)
        
        # Perform the actual slicing
        train_items = shuffled_items[:train_end]
        val_items = shuffled_items[train_end:val_end]
        test_items = shuffled_items[val_end:]
        
        # Append the items to the corresponding split lists
        splits['train'].extend(train_items)
        splits['validate'].extend(val_items)
        splits['test'].extend(test_items)
        
        print(f"  {species}: {total_count} total -> train: {len(train_items)}, validate: {len(val_items)}, test: {len(test_items)}")
    
    return splits

def copy_files_to_splits(splits, output_base_dir):
    """
    Copy images and annotation JSON files into their respective split directories.

    Maintains directory structure:
        output/split_name/img/*.jpg
        output/split_name/ann/*.json
    """
    for split_name, items in splits.items():
        # Build output folder paths for images and annotations
        split_img_dir = output_base_dir / split_name / "img"
        split_ann_dir = output_base_dir / split_name / "ann"
        
        # Ensure folders exist
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_ann_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {len(items)} files to {split_name}...")
        
        # Copy each image + annotation pair
        for item in items:
            # Destination for the image file
            dest_img = split_img_dir / item['image']
            shutil.copy2(item['image_path'], dest_img)
            
            # Destination for the annotation JSON file
            ann_file_name = item['image'] + ".json"
            dest_ann = split_ann_dir / ann_file_name
            
            # Write annotation content to new location
            with open(dest_ann, 'w', encoding='utf-8') as f:
                json.dump(item['annotation'], f, indent=4)
        
        print(f"  ✓ {split_name}: {len(items)} images and annotations copied")

def main():
    """Main execution function that orchestrates the entire process."""
    print("=" * 60)
    print("Creating Stratified Data Splits (70-20-10)")
    print("=" * 60)
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    # Verify that input data exists
    if not INPUT_BASE_DIR.exists():
        print(f"Error: Input directory {INPUT_BASE_DIR} does not exist!")
        return
    
    # Step 1: Load and group all data by species
    print("\nLoading all data from initial folders...")
    data_by_species = load_all_data()
    
    total_images = sum(len(items) for items in data_by_species.values())
    print(f"Total images loaded: {total_images}")
    print(f"Number of unique species: {len(data_by_species)}")
    
    # Abort early if dataset is empty
    if total_images == 0:
        print("Error: No data found to split!")
        return
    
    # Step 2: Perform stratified splitting
    splits = stratified_split(
        data_by_species,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['validate'],
        test_ratio=SPLIT_RATIOS['test'],
        random_seed=42
    )
    
    # Step 3: Print summary of split results
    print("\n" + "=" * 60)
    print("Split Summary:")
    print("=" * 60)
    for split_name, items in splits.items():
        ratio = len(items) / total_images
        print(f"  {split_name:10s}: {len(items):4d} images ({ratio:6.2%})")
    
    # Step 4: Copy all files to final output directories
    print("\n" + "=" * 60)
    print("Copying files to output directories...")
    print("=" * 60)
    copy_files_to_splits(splits, OUTPUT_BASE_DIR)
    
    print("\n" + "=" * 60)
    print("Done! Stratified splits created successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()
