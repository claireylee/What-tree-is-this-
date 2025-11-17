import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
import random

INPUT_BASE_DIR = Path(__file__).parent.parent / "DATA" / "INITIAL" / "sample_data"
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "DATA" / "FINAL" / "sample_data"

SPLIT_RATIOS = {
    "train": 0.70,
    "validate": 0.20,
    "test": 0.10
}

def get_image_files(folder_path):
    """Get all image files from a folder."""
    image_extensions = {'.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG'}
    image_files = []
    if folder_path.exists():
        for file in folder_path.iterdir():
            if file.suffix in image_extensions:
                image_files.append(file.name)
    return sorted(image_files)

def load_all_data():
    """Load all images and annotations from the initial data folders."""
    data_by_species = defaultdict(list)
    
    folders = ["test", "train", "val"]
    
    for folder in folders:
        img_folder = INPUT_BASE_DIR / folder / "img"
        ann_folder = INPUT_BASE_DIR / folder / "ann"
        
        if not img_folder.exists() or not ann_folder.exists():
            print(f"Warning: {folder} folder not found, skipping...")
            continue
        
        image_files = get_image_files(img_folder)
        
        for img_file in image_files:
            ann_file = ann_folder / (img_file + ".json")
            
            if not ann_file.exists():
                print(f"Warning: Annotation file not found for {img_file}, skipping...")
                continue
            
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    ann_data = json.load(f)
                    species_name = ann_data.get('name', '')
                    
                    if species_name:
                        data_by_species[species_name].append({
                            'image': img_file,
                            'image_path': img_folder / img_file,
                            'annotation': ann_data,
                            'annotation_path': ann_file
                        })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not read annotation {ann_file}: {e}")
                continue
    
    return data_by_species

def stratified_split(data_by_species, train_ratio=0.70, val_ratio=0.20, test_ratio=0.10, random_seed=42):
    """
    Perform stratified splitting ensuring proportional representation of each species.
    
    Args:
        data_by_species: Dictionary mapping species names to lists of data items
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with 'train', 'validate', 'test' keys containing lists of data items
    """
    random.seed(random_seed)
    
    splits = {
        'train': [],
        'validate': [],
        'test': []
    }
    
    # Verify ratios sum to 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
    
    print(f"\nPerforming stratified split (train={train_ratio:.0%}, validate={val_ratio:.0%}, test={test_ratio:.0%})...")
    print(f"Total species: {len(data_by_species)}")
    
    for species, items in data_by_species.items():
        # Shuffle items for this species
        shuffled_items = items.copy()
        random.shuffle(shuffled_items)
        
        total_count = len(shuffled_items)
        
        # Calculate split indices
        train_end = int(total_count * train_ratio)
        val_end = train_end + int(total_count * val_ratio)
        
        # Split the data
        train_items = shuffled_items[:train_end]
        val_items = shuffled_items[train_end:val_end]
        test_items = shuffled_items[val_end:]
        
        # Add to splits
        splits['train'].extend(train_items)
        splits['validate'].extend(val_items)
        splits['test'].extend(test_items)
        
        print(f"  {species}: {total_count} total -> train: {len(train_items)}, validate: {len(val_items)}, test: {len(test_items)}")
    
    return splits

def copy_files_to_splits(splits, output_base_dir):
    """Copy image and annotation files to the new split directories."""
    for split_name, items in splits.items():
        split_img_dir = output_base_dir / split_name / "img"
        split_ann_dir = output_base_dir / split_name / "ann"
        
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_ann_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {len(items)} files to {split_name}...")
        
        for item in items:
            # Copy image file
            dest_img = split_img_dir / item['image']
            shutil.copy2(item['image_path'], dest_img)
            
            # Copy and save annotation file
            ann_file_name = item['image'] + ".json"
            dest_ann = split_ann_dir / ann_file_name
            
            with open(dest_ann, 'w', encoding='utf-8') as f:
                json.dump(item['annotation'], f, indent=4)
        
        print(f"  ✓ {split_name}: {len(items)} images and annotations copied")

def main():
    """Main function to create stratified splits."""
    print("=" * 60)
    print("Creating Stratified Data Splits (70-20-10)")
    print("=" * 60)
    print(f"Input directory: {INPUT_BASE_DIR}")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    if not INPUT_BASE_DIR.exists():
        print(f"Error: Input directory {INPUT_BASE_DIR} does not exist!")
        return
    
    # Load all data grouped by species
    print("\nLoading all data from initial folders...")
    data_by_species = load_all_data()
    
    total_images = sum(len(items) for items in data_by_species.values())
    print(f"Total images loaded: {total_images}")
    print(f"Number of unique species: {len(data_by_species)}")
    
    if total_images == 0:
        print("Error: No data found to split!")
        return
    
    # Perform stratified splitting
    splits = stratified_split(
        data_by_species,
        train_ratio=SPLIT_RATIOS['train'],
        val_ratio=SPLIT_RATIOS['validate'],
        test_ratio=SPLIT_RATIOS['test'],
        random_seed=42
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Split Summary:")
    print("=" * 60)
    for split_name, items in splits.items():
        ratio = len(items) / total_images
        print(f"  {split_name:10s}: {len(items):4d} images ({ratio:6.2%})")
    
    # Copy files to output directories
    print("\n" + "=" * 60)
    print("Copying files to output directories...")
    print("=" * 60)
    copy_files_to_splits(splits, OUTPUT_BASE_DIR)
    
    print("\n" + "=" * 60)
    print("Done! Stratified splits created successfully.")
    print("=" * 60)

if __name__ == "__main__":
    main()

