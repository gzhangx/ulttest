# Ultralytics Image Segmentation Examples

This directory contains Python examples for performing image segmentation using Ultralytics YOLO models.

## Files Overview

### Core Examples
- **`simple_segmentation.py`** - Basic segmentation example, perfect for beginners
- **`image_segmentation_sample.py`** - Comprehensive example with advanced features
- **`clean_segmentation_example.py`** - Clean example using the import helper
- **`test_segmentation.py`** - Quick test script to verify setup

### Utilities
- **`ultralytics_import_helper.py`** - Utility for importing from local ultralytics source
- **`output_helper.py`** - NEW: Organized output directory management
- **`requirements_segmentation.txt`** - Python dependencies

### Documentation
- **`README_SEGMENTATION.md`** - This documentation file
- **`IMPORT_GUIDE.md`** - Detailed guide for importing from local source code
- **`OUTPUT_MANAGEMENT.md`** - NEW: Guide for organized output directories
- **`FIX_SUMMARY.md`** - Summary of import fixes for moved directories

## Quick Start

### 1. Install Dependencies

```bash
# Basic installation
pip install ultralytics

# Or install all optional dependencies
pip install -r requirements_segmentation.txt
```

### 2. Run Basic Example

```bash
python simple_segmentation.py
```

This will:
- Download the YOLO11n segmentation model (if not already downloaded)
- Perform segmentation on a sample image
- Create timestamped output directory and save organized results
- Show summary of created files with sizes

### 2b. Using Local Ultralytics Source Code

If you're working with the ultralytics source code directly (not the installed package), the examples automatically detect and use the local source:

```bash
# Test the import setup
python ultralytics_import_helper.py

# Run clean example with helper
python clean_segmentation_example.py

# Quick test
python test_segmentation.py
```

### 3. Run Advanced Example

```bash
python image_segmentation_sample.py
```

This comprehensive example includes:
- Batch image processing
- Video segmentation
- Custom visualizations
- Feature extraction
- Detailed mask analysis

## Available Models

Ultralytics offers several segmentation models with different sizes and performance characteristics:

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|--------|-----------|----------|
| `yolo11n-seg.pt` | Nano | Fastest | Good | Real-time, mobile |
| `yolo11s-seg.pt` | Small | Fast | Better | Balanced performance |
| `yolo11m-seg.pt` | Medium | Moderate | Very Good | General purpose |
| `yolo11l-seg.pt` | Large | Slower | Excellent | High accuracy needed |
| `yolo11x-seg.pt` | Extra Large | Slowest | Best | Maximum accuracy |

## Basic Usage Examples

### Segment a Single Image

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolo11n-seg.pt')

# Run segmentation
results = model('path/to/your/image.jpg')

# Save annotated result
results[0].save('output.jpg')
```

### Segment Multiple Images

```python
from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

# Process multiple images
image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
results = model(image_list)

# Save all results
for i, result in enumerate(results):
    result.save(f'output_{i}.jpg')
```

### Segment Video

```python
from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')

# Process video
results = model('path/to/video.mp4', save=True)
```

### Access Mask Data

```python
from ultralytics import YOLO

model = YOLO('yolo11n-seg.pt')
results = model('image.jpg')

for result in results:
    if result.masks is not None:
        # Get masks as numpy arrays
        masks = result.masks.data.cpu().numpy()
        
        # Get bounding boxes
        boxes = result.boxes.xyxy.cpu().numpy()
        
        # Get class names
        class_names = [model.names[int(cls)] for cls in result.boxes.cls]
        
        print(f"Found {len(masks)} objects: {class_names}")
```

## Customization Options

### Adjust Detection Thresholds

```python
# Higher confidence threshold (fewer, more confident detections)
results = model('image.jpg', conf=0.7)

# Lower confidence threshold (more detections, potentially less accurate)
results = model('image.jpg', conf=0.3)
```

### Specify Device

```python
# Use GPU (if available)
model = YOLO('yolo11n-seg.pt')
results = model('image.jpg', device='cuda')

# Force CPU
results = model('image.jpg', device='cpu')
```

### Custom Image Size

```python
# Larger image size for better accuracy (slower)
results = model('image.jpg', imgsz=1280)

# Smaller image size for speed (less accurate)
results = model('image.jpg', imgsz=320)
```

## Output Formats

The segmentation results include:

1. **Bounding Boxes**: Rectangular regions around detected objects
2. **Segmentation Masks**: Pixel-level object boundaries
3. **Class Labels**: Identified object categories (person, car, etc.)
4. **Confidence Scores**: Detection confidence (0-1)

## Common Use Cases

### 1. Object Counting
Count specific objects in images (people, vehicles, products, etc.)

### 2. Background Removal
Use segmentation masks to remove or replace backgrounds

### 3. Quality Control
Inspect products or components for defects

### 4. Medical Imaging
Segment anatomical structures or abnormalities

### 5. Autonomous Systems
Object detection and tracking for robotics or vehicles

### 6. Content Creation
Automatic masking for photo editing or AR/VR applications

## Performance Tips

1. **Model Selection**: Choose the right model size for your speed/accuracy requirements
2. **Image Size**: Larger images give better accuracy but are slower to process
3. **GPU Acceleration**: Use CUDA-enabled GPUs for significant speed improvements
4. **Batch Processing**: Process multiple images together for better throughput
5. **Confidence Threshold**: Adjust `conf` parameter to filter results

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure ultralytics is installed: `pip install ultralytics`
2. **CUDA Error**: Install appropriate PyTorch version with CUDA support
3. **Memory Error**: Reduce image size or batch size
4. **No Detections**: Lower confidence threshold or try different model

### GPU Setup

For CUDA support:

```bash
# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

## 🌟 **Key Features**

### 🔧 **Smart Import Management**
- **Automatic Detection**: Scripts automatically detect and use local ultralytics source
- **Fallback Support**: Falls back to installed package if local source not found
- **Reusable Helper**: Import helper can be used across multiple projects
- **Clear Feedback**: Status messages show which ultralytics version is being used

### 📁 **Organized Output Management** (NEW!)
- **Automatic Directories**: Creates timestamped output directories for each run
- **Clean Workspace**: No more scattered files in root directory
- **Structured Organization**: Separate folders for images, videos, visualizations, data
- **File Summaries**: Shows created files with sizes and locations
- **Easy Discovery**: Clear paths to all generated content

### 📊 **Example Output Structure**
```
simple_segmentation_output_20251031_143741/
├── bus_segmentation_result.jpg (350.5KB)
└── advanced_segmentation_result.jpg (350.5KB)

comprehensive_segmentation_demo_20251031_143800/
├── images/           # Segmented images
├── videos/           # Processed videos  
├── visualizations/   # Analysis plots
├── data/            # Extracted features
└── logs/            # Processing logs
```

### 📚 **Further Resources**

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLO Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [Model Training Tutorial](https://docs.ultralytics.com/modes/train/)
- [Export Models Guide](https://docs.ultralytics.com/modes/export/)
- [OUTPUT_MANAGEMENT.md](OUTPUT_MANAGEMENT.md) - Detailed output organization guide

## License

These examples are provided under the same license as the Ultralytics library (AGPL-3.0).