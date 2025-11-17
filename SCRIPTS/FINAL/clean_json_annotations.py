import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent / "DATA" / "INITIAL" / "sample_data"

ANNOTATION_FOLDERS = [
    "test/ann/",
    "train/ann/",
    "val/ann/"
]

def clean_annotation_file(json_path):
    """Clean a single annotation JSON file to only contain the name field."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        name = ""
        if 'tags' in data and len(data['tags']) > 0:
            name = data['tags'][0].get('name', '')
        
        cleaned_data = {"name": name}
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4)
        
        return True
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not process {json_path}: {e}")
        return False

def clean_all_annotations():
    """Clean all annotation JSON files in the specified folders."""
    total_files = 0
    successful = 0
    failed = 0
    
    for folder_name in ANNOTATION_FOLDERS:
        ann_folder = BASE_DIR / folder_name.replace('/', os.sep)
        
        if not ann_folder.exists():
            print(f"Warning: Folder {ann_folder} does not exist!")
            continue
        
        json_files = list(ann_folder.glob("*.json"))
        total_files += len(json_files)
        
        print(f"Processing {len(json_files)} files in {folder_name}...")
        
        for json_file in json_files:
            if clean_annotation_file(json_file):
                successful += 1
            else:
                failed += 1
    
    print(f"\nSummary:")
    print(f"  Total files: {total_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

if __name__ == "__main__":
    print("Starting JSON annotation cleaning process...")
    print(f"Base directory: {BASE_DIR}")
    
    if not BASE_DIR.exists():
        print(f"Error: Base directory {BASE_DIR} does not exist!")
    else:
        clean_all_annotations()
        print("\nDone!")

