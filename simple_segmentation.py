#!/usr/bin/env python3
"""
Simple Ultralytics Image Segmentation Example
============================================

A minimal example showing how to use Ultralytics YOLO for image segmentation.

Requirements:
- ultralytics
- opencv-python (optional, for advanced image handling)
- matplotlib (optional, for visualization)

Usage:
    python simple_segmentation.py
"""

import sys
import os
from pathlib import Path

# Add the local ultralytics directory to Python path
# This allows importing from the local source code instead of installed package
current_dir = Path(__file__).parent
ultralytics_source_dir = current_dir.parent / "ultralytics"  # Go up to parent, then to ultralytics
print(f"Looking for ultralytics at: {ultralytics_source_dir}")
if ultralytics_source_dir.exists():
    sys.path.insert(0, str(ultralytics_source_dir))
    print(f"✅ Added ultralytics path: {ultralytics_source_dir}")
else:
    print(f"❌ Ultralytics not found at: {ultralytics_source_dir}")
    print("💡 Falling back to installed package")

# Now import from the local ultralytics source
from ultralytics import YOLO
import cv2

# Import helpers for organized file management and GPU selection
from output_helper import create_output_directory, get_output_path, print_output_summary, get_timestamped_filename
from gpu_helper import select_best_device, setup_device_for_model


def basic_segmentation():
    """
    Basic image segmentation example.
    """
    print("🚀 Loading YOLO segmentation model...")
    
    # Create shared output directory
    output_dir = create_output_directory()
    print(f"📁 Output directory: {output_dir}")
    files_created = []
    
    # Load a segmentation model (will download automatically if not present)
    model = YOLO('yolo11n-seg.pt')  # nano version for speed
    # Other options: yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt
    
    print("✅ Model loaded successfully!")
    
    # 🖥️ Setup optimal device (CPU/GPU) automatically
    print("\n🖥️ Configuring compute device...")
    device = select_best_device(prefer_gpu=True, verbose=True)
    
    # Example 1: Segment an image from URL
    print("\n📸 Example 1: Segmenting image from URL")
    results = model('https://ultralytics.com/images/bus.jpg', device=device)
    
    # Process results
    for i, result in enumerate(results):
        # Print detection info
        if result.boxes is not None:
            print(f"Found {len(result.boxes)} objects")
            
            # Print details for each detected object
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = model.names[class_id]
                print(f"  - {class_name}: {confidence:.2f}")
        
        if result.masks is not None:
            print(f"Generated {len(result.masks)} segmentation masks")
        
        # Save the annotated image to shared output directory with timestamp
        timestamped_filename = get_timestamped_filename('bus_segmentation_result.jpg', 'simple')
        output_path = output_dir / timestamped_filename
        result.save(str(output_path))
        files_created.append(output_path.name)
        print(f"💾 Result saved to: {output_path}")
    
    # Print summary of created files
    print_output_summary(output_dir, files_created)
    
    return results


def segment_custom_image(image_path):
    """
    Segment a custom image.
    
    Args:
        image_path (str): Path to your image file
    """
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return None
    
    print(f"\n📸 Segmenting custom image: {image_path}")
    
    # Use shared output directory
    output_dir = create_output_directory()
    print(f"📁 Output directory: {output_dir}")
    files_created = []
    
    # Load model and setup device
    model = YOLO('yolo11n-seg.pt')
    device = select_best_device(prefer_gpu=True, verbose=False)
    
    # Run segmentation
    results = model(image_path, device=device)
    
    # Process and save results
    for result in results:
        if result.boxes is not None:
            print(f"✅ Found {len(result.boxes)} objects")
            
            # Show detected classes
            classes_found = set()
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                classes_found.add(class_name)
            
            print(f"Classes detected: {', '.join(classes_found)}")
        else:
            print("No objects detected")
        
        # Save result with segmentation masks to shared output directory
        timestamped_filename = get_timestamped_filename(f"segmented_{Path(image_path).name}", 'custom')
        output_path = output_dir / timestamped_filename
        result.save(str(output_path))
        files_created.append(output_path.name)
        print(f"💾 Result saved to: {output_path}")
    
    # Print summary of created files
    print_output_summary(output_dir, files_created)
    
    return results


def advanced_segmentation_options():
    """
    Demonstrate advanced segmentation options.
    """
    print("\n🔧 Advanced Segmentation Options")
    
    # Use shared output directory for advanced results
    output_dir = create_output_directory()
    
    # Setup device for advanced options
    device = select_best_device(prefer_gpu=True, verbose=False)
    print(f"📁 Output directory: {output_dir}")
    files_created = []
    
    model = YOLO('yolo11n-seg.pt')
    
    # Segment with custom parameters
    print("\n⚙️ Example with custom parameters...")
    results = model(
        'https://ultralytics.com/images/bus.jpg',
        conf=0.3,      # Confidence threshold
        iou=0.5,       # IoU threshold for NMS
        imgsz=640,     # Image size
        device=device  # Use auto-selected device
    )
    
    # Access detailed information
    for result in results:
        print(f"Image shape: {result.orig_img.shape}")
        print(f"Processing time: {result.speed}")
        
        if result.masks is not None:
            # Access mask data
            masks = result.masks.data  # Mask tensors
            print(f"Mask tensor shape: {masks.shape}")
            
            # Access individual masks
            for i, mask in enumerate(result.masks):
                print(f"Mask {i+1} shape: {mask.data.shape}")
        
        if result.boxes is not None:
            # Access bounding box coordinates
            boxes = result.boxes.xyxy  # Box coordinates in xyxy format
            confidences = result.boxes.conf  # Confidence scores
            classes = result.boxes.cls  # Class indices
            
            print(f"Box coordinates shape: {boxes.shape}")
            print(f"Confidence scores: {confidences}")
        
        # Save advanced results
        timestamped_filename = get_timestamped_filename('advanced_segmentation_result.jpg', 'advanced')
        output_path = output_dir / timestamped_filename
        result.save(str(output_path))
        files_created.append(output_path.name)
        print(f"💾 Advanced result saved to: {output_path}")
    
    # Print summary of created files
    print_output_summary(output_dir, files_created)
    return results


def main():
    """
    Main function with examples.
    """
    print("🎯 Ultralytics Image Segmentation Examples")
    print("=" * 50)
    
    # Example 1: Basic segmentation
    basic_segmentation()
    
    # Example 2: Advanced options
    advanced_segmentation_options()
    
    # Example 3: Custom image (uncomment and provide your image path)
    print(f"\n📝 To segment your own image, use:")
    print(f"segment_custom_image('path/to/your/image.jpg')")
    
    # Uncomment the line below and replace with your image path:
    # segment_custom_image('your_image.jpg')
    
    print("\n🎉 Examples completed!")
    print("\n📚 Additional tips:")
    print("• Use larger models (yolo11m-seg.pt, yolo11l-seg.pt) for better accuracy")
    print("• Adjust conf parameter to filter detections by confidence")
    print("• Use GPU (device='cuda') for faster processing")
    print("• Check result.show() to display results directly")


if __name__ == "__main__":
    main()