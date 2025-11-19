"""
Filename: clean_json_annotations.py

Description:
    This script cleans JSON annotation files by extracting only the 'name'
    field from the first tag in each annotation. It rewrites each JSON file
    so that it contains only:
        {
            "name": "<value>"
        }

Notes:
    - Expects dataset structure under DATA/INITIAL/sample_data
    - Requires test/ann/, train/ann/, and val/ann/ folders to exist
    - Ignores missing or malformed JSON files but logs warnings
"""

import os
import json
from pathlib import Path

# Base directory where all annotation folders are located
BASE_DIR = Path(__file__).parent.parent.parent / "DATA" / "INITIAL" / "sample_data"

# Relative annotation folder paths (test, train, val)
ANNOTATION_FOLDERS = [
    "test/ann/",
    "train/ann/",
    "val/ann/"
]

def clean_annotation_file(json_path):
    """
    Clean a single annotation JSON file so it only contains:
        {"name": "<string>"}

    Parameters:
        json_path (Path): Path to the JSON annotation file.

    Returns:
        bool: True if cleaned successfully, False otherwise.
    """
    try:
        # Read original annotation JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Default empty name in case tags field is missing or empty
        name = ""
        
        # Extract name from the first tag if available
        if 'tags' in data and len(data['tags']) > 0:
            name = data['tags'][0].get('name', '')
        
        # Create the cleaned JSON structure
        cleaned_data = {"name": name}
        
        # Overwrite the file with the cleaned version
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=4)
        
        return True
    
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        # Warn but continue processing other files
        print(f"Warning: Could not process {json_path}: {e}")
        return False

def clean_all_annotations():
    """
    Iterate over all annotation directories and clean every JSON file found.

    Prints processing progress and summarizing totals.
    """
    total_files = 0      # Count of all JSON files found
    successful = 0       # Number of successfully cleaned files
    failed = 0           # Number of files that caused errors
    
    # Loop through each annotation folder (test/train/val)
    for folder_name in ANNOTATION_FOLDERS:
        # Convert path to OS-specific separators and append to base directory
        ann_folder = BASE_DIR / folder_name.replace('/', os.sep)
        
        # Check folder existence to avoid exceptions
        if not ann_folder.exists():
            print(f"Warning: Folder {ann_folder} does not exist!")
            continue
        
        # Get all .json annotation files inside the folder
        json_files = list(ann_folder.glob("*.json"))
        total_files += len(json_files)
        
        print(f"Processing {len(json_files)} files in {folder_name}...")
        
        # Clean each file individually
        for json_file in json_files:
            if clean_annotation_file(json_file):
                successful += 1
            else:
                failed += 1
    
    # Print summary results
    print(f"\nSummary:")
    print(f"  Total files: {total_files}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

if __name__ == "__main__":
    # Entry point when running as a script
    print("Starting JSON annotation cleaning process...")
    print(f"Base directory: {BASE_DIR}")
    
    # Ensure the base directory exists before processing
    if not BASE_DIR.exists():
        print(f"Error: Base directory {BASE_DIR} does not exist!")
    else:
        # Run the full cleaning routine
        clean_all_annotations()
        print("\nDone!")
