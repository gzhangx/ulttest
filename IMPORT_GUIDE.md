# Importing Ultralytics from Local Source Code

When working with the Ultralytics source code directly (instead of the installed package), you need to modify the Python import path to use the local code. Here are several methods to accomplish this:

## Method 1: Modify sys.path in Your Script (Recommended)

This is the approach used in the updated sample files:

```python
import sys
from pathlib import Path

# Add the local ultralytics directory to Python path
current_dir = Path(__file__).parent
ultralytics_source_dir = current_dir.parent / "ultralytics"  # Go up to parent, then to ultralytics
if ultralytics_source_dir.exists():
    sys.path.insert(0, str(ultralytics_source_dir))
    print(f"✅ Added ultralytics path: {ultralytics_source_dir}")
else:
    print(f"❌ Ultralytics not found at: {ultralytics_source_dir}")

# Now import from the local source
from ultralytics import YOLO
```

## Method 2: Set PYTHONPATH Environment Variable

### Option A: Set in terminal before running
```bash
export PYTHONPATH="/mnt/d/work/cur/ultralitics/ultralytics:$PYTHONPATH"
python your_script.py
```

### Option B: Set in script
```python
import os
import sys

# Add to PYTHONPATH
ultralytics_path = "/mnt/d/work/cur/ultralitics/ultralytics"
os.environ['PYTHONPATH'] = f"{ultralytics_path}:{os.environ.get('PYTHONPATH', '')}"
sys.path.insert(0, ultralytics_path)

from ultralytics import YOLO
```

## Method 3: Install in Development Mode

Install the local ultralytics package in editable/development mode:

```bash
cd /mnt/d/work/cur/ultralitics/ultralytics
pip install -e .
```

This creates a link to your source code, so changes are immediately available without reinstalling.

## Method 4: Direct Import with Absolute Path

For quick testing without modifying sys.path globally:

```python
import sys
sys.path.append('/mnt/d/work/cur/ultralitics/ultralytics')

from ultralytics import YOLO
```

## Directory Structure Understanding

Your current structure:
```
/mnt/d/work/cur/ultralitics/
├── ultralytics/                    # Git repository root
│   ├── ultralytics/               # Python package directory
│   │   ├── __init__.py           # Makes it importable
│   │   ├── models/
│   │   ├── utils/
│   │   └── ...
│   ├── examples/
│   ├── tests/
│   └── pyproject.toml
├── ulttest/                       # Your example scripts directory
│   ├── simple_segmentation.py
│   ├── image_segmentation_sample.py
│   ├── ultralytics_import_helper.py
│   └── ...
├── cv/                            # Virtual environment
└── other files...
```

The key is that you need to add `/mnt/d/work/cur/ultralitics/ultralytics` (the outer one) to your Python path so that Python can find the `ultralytics` package inside it.

**For scripts in the `ulttest` subdirectory**, the path calculation becomes:
- Current directory: `/mnt/d/work/cur/ultralitics/ulttest`  
- Go up one level: `/mnt/d/work/cur/ultralitics`
- Then into ultralytics: `/mnt/d/work/cur/ultralitics/ultralytics`

## Verification

To verify the import is working from local source:

```python
import ultralytics
print(f"Ultralytics location: {ultralytics.__file__}")
print(f"Ultralytics version: {ultralytics.__version__}")
```

You should see a path pointing to your local directory, not a site-packages directory.

## Best Practices

1. **Use Method 1 (sys.path modification)** for scripts you're developing alongside the ultralytics code
2. **Use Method 3 (pip install -e)** if you're planning to make changes to ultralytics and want them available system-wide
3. **Always use relative paths** when possible to make your code portable
4. **Document your import method** in your script comments so others understand the setup

## Troubleshooting

### Import Error: "No module named 'ultralytics'"
- Check that the path you're adding actually contains the ultralytics package
- Verify the ultralytics directory has an `__init__.py` file

### Import Error: "Cannot resolve ultralytics.utils"
- This is often a VS Code linting issue and doesn't affect runtime
- The code should still work when executed

### Module Not Found After Adding to sys.path
- Make sure you're adding the parent directory of the ultralytics package, not the package itself
- Use absolute paths to avoid confusion

## Example: Complete Working Script

```python
#!/usr/bin/env python3
"""
Example script using local ultralytics source code.
"""

import sys
from pathlib import Path

def setup_local_ultralytics():
    """Setup import path for local ultralytics."""
    current_dir = Path(__file__).parent
    ultralytics_source_dir = current_dir / "ultralytics"
    
    if ultralytics_source_dir.exists():
        sys.path.insert(0, str(ultralytics_source_dir))
        print(f"✅ Added local ultralytics path: {ultralytics_source_dir}")
        return True
    else:
        print(f"❌ Local ultralytics not found at: {ultralytics_source_dir}")
        print("💡 Falling back to installed package")
        return False

def main():
    # Setup local imports
    setup_local_ultralytics()
    
    # Now import (will use local if available, installed package otherwise)
    from ultralytics import YOLO
    
    # Verify which version we're using
    import ultralytics
    print(f"Using ultralytics from: {ultralytics.__file__}")
    
    # Your segmentation code here
    model = YOLO('yolo11n-seg.pt')
    results = model('https://ultralytics.com/images/bus.jpg')
    print(f"✅ Segmentation completed successfully!")

if __name__ == "__main__":
    main()
```

This approach provides robust import handling that works whether you have the local source or just the installed package.