#!/usr/bin/env python3
"""
Ultralytics Image Segmentation Sample
=====================================

This script demonstrates various image segmentation capabilities using Ultralytics YOLO models.
It includes examples for:
1. Basic instance segmentation
2. Batch processing of images
3. Video segmentation
4. Custom model training preparation
5. Results visualization and export

Requirements:
- ultralytics
- opencv-python
- matplotlib
- pillow

Usage:
    python image_segmentation_sample.py
"""

import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add the local ultralytics directory to Python path
# This allows importing from the local source code instead of installed package
current_dir = Path(__file__).parent
ultralytics_source_dir = current_dir / "ultralytics"
sys.path.insert(0, str(ultralytics_source_dir))

# Now import from the local ultralytics source
from ultralytics import YOLO
from ultralytics.utils import ASSETS


class ImageSegmentationDemo:
    """
    Comprehensive image segmentation demonstration using Ultralytics YOLO models.
    """
    
    def __init__(self, model_name="yolo11n-seg.pt"):
        """
        Initialize the segmentation model.
        
        Args:
            model_name (str): Name of the YOLO segmentation model to use.
                            Options: yolo11n-seg.pt, yolo11s-seg.pt, yolo11m-seg.pt, yolo11l-seg.pt, yolo11x-seg.pt
        """
        print(f"Loading YOLO segmentation model: {model_name}")
        self.model = YOLO(model_name)
        print(f"Model loaded successfully!")
        print(f"Model info: {self.model.info()}")
    
    def segment_single_image(self, image_path, save_path=None, show_results=True):
        """
        Perform segmentation on a single image.
        
        Args:
            image_path (str): Path to the input image
            save_path (str, optional): Path to save the annotated result
            show_results (bool): Whether to display the results
            
        Returns:
            results: YOLO results object containing segmentation masks and bounding boxes
        """
        print(f"\n=== Segmenting single image: {image_path} ===")
        
        # Run inference
        results = self.model(image_path)
        
        # Process results
        for i, result in enumerate(results):
            print(f"Found {len(result.boxes)} objects" if result.boxes is not None else "No objects detected")
            
            if result.masks is not None:
                print(f"Generated {len(result.masks)} segmentation masks")
                
                # Get class names and confidence scores
                if result.boxes is not None:
                    for box, mask in zip(result.boxes, result.masks):
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        class_name = self.model.names[class_id]
                        print(f"  - {class_name}: {confidence:.2f}")
            
            # Visualize results
            if show_results:
                annotated_frame = result.plot()
                
                # Display using matplotlib
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                plt.title(f"Image Segmentation Results - {Path(image_path).name}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # Save results if requested
            if save_path:
                result.save(save_path)
                print(f"Results saved to: {save_path}")
        
        return results
    
    def segment_batch_images(self, image_folder, output_folder=None):
        """
        Process multiple images in batch.
        
        Args:
            image_folder (str): Path to folder containing images
            output_folder (str, optional): Path to save processed images
        """
        print(f"\n=== Processing batch images from: {image_folder} ===")
        
        image_folder = Path(image_folder)
        if not image_folder.exists():
            print(f"Error: Folder {image_folder} does not exist")
            return
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = [f for f in image_folder.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print("No image files found in the specified folder")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # Process all images
        results = self.model(image_files)
        
        # Create output directory if specified
        if output_folder:
            output_folder = Path(output_folder)
            output_folder.mkdir(parents=True, exist_ok=True)
        
        # Process results
        for i, (result, image_file) in enumerate(zip(results, image_files)):
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            
            if result.masks is not None:
                print(f"  Found {len(result.masks)} objects with masks")
            
            if output_folder:
                output_path = output_folder / f"segmented_{image_file.name}"
                result.save(output_path)
    
    def segment_video(self, video_path, output_path=None, show_live=False):
        """
        Perform segmentation on video frames.
        
        Args:
            video_path (str): Path to input video
            output_path (str, optional): Path to save output video
            show_live (bool): Whether to show live processing
        """
        print(f"\n=== Processing video: {video_path} ===")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path is specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}", end='\r')
                
                # Run segmentation
                results = self.model(frame)
                
                # Get annotated frame
                annotated_frame = results[0].plot()
                
                # Save frame if output video is specified
                if output_path:
                    out.write(annotated_frame)
                
                # Show live processing
                if show_live:
                    cv2.imshow('YOLO Segmentation', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            # Cleanup
            cap.release()
            if output_path:
                out.release()
            if show_live:
                cv2.destroyAllWindows()
            
            print(f"\nVideo processing complete!")
            if output_path:
                print(f"Output saved to: {output_path}")
    
    def extract_masks_and_features(self, image_path):
        """
        Extract detailed mask information and features.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            dict: Dictionary containing masks, bounding boxes, and other features
        """
        print(f"\n=== Extracting detailed features from: {image_path} ===")
        
        results = self.model(image_path)
        result = results[0]
        
        extracted_data = {
            'masks': [],
            'boxes': [],
            'classes': [],
            'confidences': [],
            'class_names': []
        }
        
        if result.masks is not None and result.boxes is not None:
            for mask, box in zip(result.masks, result.boxes):
                # Extract mask as numpy array
                mask_array = mask.data.cpu().numpy()
                extracted_data['masks'].append(mask_array)
                
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                extracted_data['boxes'].append([x1, y1, x2, y2])
                
                # Extract class information
                class_id = int(box.cls)
                confidence = float(box.conf)
                class_name = self.model.names[class_id]
                
                extracted_data['classes'].append(class_id)
                extracted_data['confidences'].append(confidence)
                extracted_data['class_names'].append(class_name)
        
        # Print summary
        print(f"Extracted {len(extracted_data['masks'])} objects:")
        for i, (class_name, confidence) in enumerate(zip(extracted_data['class_names'], extracted_data['confidences'])):
            print(f"  {i+1}. {class_name}: {confidence:.3f}")
        
        return extracted_data
    
    def custom_visualization(self, image_path, output_path=None):
        """
        Create custom visualization of segmentation results.
        
        Args:
            image_path (str): Path to input image
            output_path (str, optional): Path to save custom visualization
        """
        print(f"\n=== Creating custom visualization for: {image_path} ===")
        
        # Load original image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        results = self.model(image_path)
        result = results[0]
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Segmentation Analysis - {Path(image_path).name}', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Standard YOLO result
        if result.boxes is not None:
            annotated = result.plot()
            axes[0, 1].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title('YOLO Segmentation')
            axes[0, 1].axis('off')
        
        # Individual masks overlay
        if result.masks is not None:
            mask_overlay = image_rgb.copy()
            colors = plt.cm.tab10(np.linspace(0, 1, len(result.masks)))
            
            for i, mask in enumerate(result.masks):
                mask_array = mask.data.cpu().numpy()[0]
                color = (np.array(colors[i][:3]) * 255).astype(np.uint8)
                
                # Create colored mask
                colored_mask = np.zeros_like(image_rgb)
                colored_mask[mask_array > 0.5] = color
                
                # Blend with original image
                mask_overlay = cv2.addWeighted(mask_overlay, 0.8, colored_mask, 0.3, 0)
            
            axes[1, 0].imshow(mask_overlay)
            axes[1, 0].set_title('Colored Masks Overlay')
            axes[1, 0].axis('off')
        
        # Statistics
        if result.boxes is not None and result.masks is not None:
            class_counts = {}
            confidences = []
            
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = self.model.names[class_id]
                confidence = float(box.conf)
                
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                confidences.append(confidence)
            
            # Bar plot of class counts
            if class_counts:
                axes[1, 1].bar(class_counts.keys(), class_counts.values())
                axes[1, 1].set_title('Object Counts by Class')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 1].text(0.5, 0.5, 'No objects detected', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Object Statistics')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Custom visualization saved to: {output_path}")
        
        plt.show()


def main():
    """
    Main function demonstrating various segmentation capabilities.
    """
    print("🚀 Ultralytics Image Segmentation Demo")
    print("=" * 50)
    
    # Initialize the segmentation model
    # You can change this to other models like yolo11s-seg.pt, yolo11m-seg.pt, etc.
    demo = ImageSegmentationDemo("yolo11n-seg.pt")
    
    # Demo 1: Segment a sample image from ASSETS
    print(f"\n📸 Demo 1: Single Image Segmentation")
    sample_image = ASSETS / 'bus.jpg'  # Built-in sample image
    
    if sample_image.exists():
        results = demo.segment_single_image(str(sample_image))
        
        # Extract detailed features
        features = demo.extract_masks_and_features(str(sample_image))
        
        # Create custom visualization
        demo.custom_visualization(str(sample_image), 'custom_segmentation_result.png')
    else:
        print("Sample image not found. Please provide your own image path.")
        # Example with custom image:
        # results = demo.segment_single_image('path/to/your/image.jpg')
    
    # Demo 2: Batch processing (uncomment to use with your own images)
    print(f"\n📁 Demo 2: Batch Processing")
    print("To use batch processing, uncomment the following lines and provide your image folder:")
    print("# demo.segment_batch_images('path/to/image/folder', 'path/to/output/folder')")
    
    # Demo 3: Video processing (uncomment to use with your own video)
    print(f"\n🎥 Demo 3: Video Processing")
    print("To process videos, uncomment the following lines and provide your video path:")
    print("# demo.segment_video('path/to/video.mp4', 'output_video.mp4')")
    
    # Demo 4: Model information
    print(f"\n📊 Model Information:")
    print(f"Model names/classes: {list(demo.model.names.values())[:10]}...")  # Show first 10 classes
    print(f"Total classes: {len(demo.model.names)}")
    
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Replace sample images with your own images")
    print("2. Adjust confidence thresholds using: model.predict(source, conf=0.5)")
    print("3. Train custom models on your own data")
    print("4. Export models to different formats (ONNX, TensorRT, etc.)")


if __name__ == "__main__":
    main()