#!/usr/bin/env python3
"""
Clean Ultralytics Segmentation Example
======================================

This example uses the import helper for clean, reusable code.
"""

# Import the helper first
from ultralytics_import_helper import setup_local_ultralytics, verify_ultralytics_import

# Setup local ultralytics import
setup_local_ultralytics(verbose=True)

# Now we can import ultralytics
from ultralytics import YOLO
from pathlib import Path


def run_segmentation_demo():
    """
    Run a clean segmentation demonstration.
    """
    print("\n🎯 Clean Segmentation Demo")
    print("=" * 30)
    
    # Verify our setup
    if not verify_ultralytics_import(verbose=False):
        print("❌ Ultralytics import verification failed")
        return False
    
    try:
        # Load model
        print("📥 Loading YOLO segmentation model...")
        model = YOLO('yolo11n-seg.pt')
        
        # Run segmentation on sample image
        print("🔍 Running segmentation...")
        results = model('https://ultralytics.com/images/bus.jpg', verbose=False)
        
        # Process results
        result = results[0]
        if result.boxes is not None:
            print(f"✅ Success! Found {len(result.boxes)} objects")
            
            # Show what was detected
            detected_classes = []
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                confidence = float(box.conf)
                detected_classes.append(f"{class_name} ({confidence:.2f})")
            
            print(f"📋 Detected: {', '.join(detected_classes)}")
            
            # Save result
            result.save('clean_demo_result.jpg')
            print("💾 Result saved as 'clean_demo_result.jpg'")
            
            return True
        else:
            print("❌ No objects detected")
            return False
            
    except Exception as e:
        print(f"❌ Error during segmentation: {e}")
        return False


def main():
    """
    Main function.
    """
    print("🚀 Ultralytics Local Import Demo")
    print("Using helper for clean imports")
    
    success = run_segmentation_demo()
    
    if success:
        print("\n🎉 Demo completed successfully!")
        print("\n💡 This example shows how to:")
        print("• Use the import helper for clean code")
        print("• Import from local ultralytics source")
        print("• Run segmentation with minimal setup")
    else:
        print("\n❌ Demo failed")


if __name__ == "__main__":
    main()