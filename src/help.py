import os
from pathlib import Path


def count_files_in_subfolders(root_dir: str) -> None:
    """
    Counts the number of files in each subfolder and prints the results.
    
    Args:
        root_dir: Path to the starting directory (default: current directory)
    """
    root = Path(root_dir).resolve()
    
    if not root.is_dir():
        print(f"Error: {root} is not a directory")
        return
    
    print(f"\nCounting files in subfolders of: {root}")
    print("-" * 60)
    
    total_files = 0
    folder_count = 0
    
    # Walk through directory tree
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip the root folder itself if you only want subfolders
        if Path(dirpath) == root:
            continue
            
        file_count = len(filenames)
        if file_count > 0 or True:  # show even empty folders (remove "or True" to hide them)
            relative_path = Path(dirpath).relative_to(root)
            print(f"{relative_path} : {file_count:,} files")
            total_files += file_count
            folder_count += 1
    
    print("-" * 60)
    print(f"Total subfolders scanned : {folder_count}")
    print(f"Total files found        : {total_files:,}")


# ------------------------------------------------
#  Usage examples
# ------------------------------------------------

if __name__ == "__main__":
    # Option 1: Current directory
    # count_files_in_subfolders()

    # Option 2: Specific folder (recommended)
    #folder_to_scan = r"C:\Users\Fan\Pictures"          # ??? change this path
    # folder_to_scan = "/home/fan/photos"              # Linux/Mac example
    folder_to_scan = '/vast/home/fwang/image_ai/data/train/'    # Path to testing data folder
    
    count_files_in_subfolders(folder_to_scan)