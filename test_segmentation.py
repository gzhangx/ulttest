#!/usr/bin/env python3
"""
Minimal Ultralytics Segmentation Test
=====================================

Quick test to verify the segmentation setup works.
This script will download a model and test it on a sample image.
"""

import sys
from pathlib import Path

def test_segmentation():
    """Quick segmentation test."""
    try:
        # Add the local ultralytics directory to Python path
        # This allows importing from the local source code instead of installed package
        current_dir = Path(__file__).parent
        ultralytics_source_dir = current_dir / "ultralytics"
        sys.path.insert(0, str(ultralytics_source_dir))
        
        # Import ultralytics from local source
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
        
        # Load model (will auto-download if needed)
        print("📥 Loading YOLO segmentation model...")
        model = YOLO('yolo11n-seg.pt')
        print("✅ Model loaded successfully")
        
        # Test with a URL image (no local file needed)
        print("🔍 Running segmentation on test image...")
        results = model('https://ultralytics.com/images/bus.jpg', verbose=False)
        
        # Check results
        result = results[0]
        if result.boxes is not None:
            num_objects = len(result.boxes)
            print(f"✅ Segmentation successful! Found {num_objects} objects")
            
            # Show detected classes
            classes = []
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                confidence = float(box.conf)
                classes.append(f"{class_name} ({confidence:.2f})")
            
            print(f"📋 Detected: {', '.join(classes)}")
            
            if result.masks is not None:
                print(f"🎭 Generated {len(result.masks)} segmentation masks")
            
            # Save result
            result.save('test_segmentation_output.jpg')
            print("💾 Result saved as 'test_segmentation_output.jpg'")
            
            return True
        else:
            print("❌ No objects detected")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing ultralytics: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Ultralytics Segmentation Test")
    print("=" * 40)
    
    success = test_segmentation()
    
    if success:
        print("\n🎉 Test completed successfully!")
        print("\n🔧 Ready to use segmentation features:")
        print("• Run 'python simple_segmentation.py' for basic examples")
        print("• Run 'python image_segmentation_sample.py' for advanced features")
        print("• Check 'README_SEGMENTATION.md' for documentation")
    else:
        print("\n❌ Test failed. Please check your installation.")
        print("\n💡 Installation steps:")
        print("1. pip install ultralytics")
        print("2. Ensure you have internet connection for model download")

if __name__ == "__main__":
    main()