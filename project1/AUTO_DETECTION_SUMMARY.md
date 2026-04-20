# Automatic Class Name Detection Implementation

## Overview

The medical imaging classification project now automatically detects class names from the directory structure, eliminating the need for manual configuration.

**Key Feature**: Class names are determined by the subfolder names in `data/train/`, sorted alphabetically for reproducibility.

## How It Works

### 1. **Config File Auto-Detection** (`config/config.py`)

When the config module is imported, it:
- Calls `_get_class_names_from_directory(str(TRAIN_DIR))`
- Reads all subdirectories from the training directory
- Sorts them alphabetically
- Sets `CLASS_NAMES` and `NUM_CLASSES` automatically
- Falls back to default classes if directory is not accessible during import

```python
# Auto-detection logic in config.py:
_AUTO_CLASS_NAMES = _get_class_names_from_directory(str(TRAIN_DIR))
if _AUTO_CLASS_NAMES:
    CLASS_NAMES = _AUTO_CLASS_NAMES
    NUM_CLASSES = len(CLASS_NAMES)
else:
    # Fallback to default if directory not found
    CLASS_NAMES = [f"class_{i}" for i in range(10)]
    NUM_CLASSES = len(CLASS_NAMES)
```

### 2. **Utility Function** (`src/utils.py`)

New function `get_class_names_from_directory()` provides reusable logic:
- Takes a directory path as input
- Returns a sorted list of class names
- Validates that subdirectories exist
- Provides informative error messages

```python
def get_class_names_from_directory(data_dir: str) -> List[str]:
    """
    Auto-detect class names from directory structure.
    
    Args:
        data_dir: Path to directory containing class subdirectories
        
    Returns:
        Sorted list of class names (subdirectory names)
    """
```

### 3. **Enhanced Data Validation** (`src/utils.py`)

Updated `verify_data_structure()` function:
- Now accepts optional `test_dir` and `num_classes` parameters
- Auto-detects classes from training directory
- Displays detected class names with image counts
- Validates test directory has same classes as training

## Example Data Structure

```
data/
├── train/
│   ├── pneumonia/          ← CLASS_NAMES[0] = 'pneumonia'
│   ├── normal/             ← CLASS_NAMES[1] = 'normal'
│   └── covid/              ← CLASS_NAMES[2] = 'covid'
└── test/
    ├── pneumonia/
    ├── normal/
    └── covid/
```

With this structure:
- `CLASS_NAMES = ['covid', 'normal', 'pneumonia']` (alphabetically sorted)
- `NUM_CLASSES = 3`
- No manual configuration needed!

## Implementation Details

### Files Modified

1. **`config/config.py`**
   - Added `_get_class_names_from_directory()` function
   - Replaced hardcoded `CLASS_NAMES` list with auto-detection
   - Updated to calculate `NUM_CLASSES` automatically

2. **`src/utils.py`**
   - Added `get_class_names_from_directory()` public function
   - Updated `verify_data_structure()` with optional parameters
   - Enhanced to display detected classes

3. **`train.py`**
   - Updated to use combined `verify_data_structure(TRAIN_DIR, TEST_DIR)` call

4. **Documentation Files**
   - `README.md` - Configuration section now highlights auto-detection
   - `QUICKREF.py` - Updated with auto-detection examples
   - `examples.py` - Updated with auto-detection emphasis
   - `PROJECT_SUMMARY.md` - Removed manual configuration instructions
   - `setup.py` - Updated data structure checks

### Backward Compatibility

- ✅ Fallback to default class names if directory is not found
- ✅ Optional parameters in verification functions
- ✅ No breaking changes to existing code
- ✅ Auto-detected classes are sorted for consistency

## Usage Examples

### Simple Usage
```python
from config.config import CLASS_NAMES, NUM_CLASSES

print(f"Classes: {CLASS_NAMES}")           # ['covid', 'normal', 'pneumonia']
print(f"Number of classes: {NUM_CLASSES}") # 3
```

### Auto-Detection in Data Loader
```python
from src.utils import verify_data_structure
from config.config import TRAIN_DIR, TEST_DIR

# Verification automatically displays detected classes
verify_data_structure(TRAIN_DIR, TEST_DIR)
```

### Custom Directory
```python
from src.utils import get_class_names_from_directory

# Get class names from any directory
classes = get_class_names_from_directory('path/to/directory')
print(f"Detected classes: {classes}")
```

## Key Benefits

1. **Zero Configuration**: No need to manually edit class names
2. **Scalability**: Works with any number of classes
3. **Consistency**: Alphabetical sorting ensures reproducibility
4. **Flexibility**: Works with any naming convention
5. **Robustness**: Fallback mechanism for safety
6. **Validation**: Automatic check that test directory has same classes

## Testing the Auto-Detection

To verify auto-detection is working:

```bash
# Run setup script to see detected classes
python setup.py

# Run training script
python train.py

# View detected classes in logs
```

## Migration Notes

### From Previous Version
- **Before**: Had to manually edit `config.py` to set `CLASS_NAMES`
- **After**: Just create subdirectories with class names in `data/train/`

### No Changes Required
- All existing code continues to work
- Training, evaluation, and prediction scripts unchanged
- No dependencies added or removed

## Troubleshooting

### Classes Not Detected
- **Check**: `data/train/` directory exists
- **Check**: Subdirectories are named correctly (no special characters)
- **Check**: Classes are used consistently in `data/test/`

### Different Class Order
- **Note**: Classes are sorted alphabetically
- **Example**: `['apple', 'banana', 'cherry']` not `['banana', 'cherry', 'apple']`
- This is intentional for reproducibility

### Manual Override (If Needed)
```python
# In config.py, after auto-detection, you can override:
CLASS_NAMES = ['custom', 'order', 'here']
NUM_CLASSES = len(CLASS_NAMES)
```

## Summary

This implementation provides a seamless, automatic way to configure class names from your data directory structure. Simply organize your data with descriptive folder names, and the project handles the rest!
