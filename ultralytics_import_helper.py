#!/usr/bin/env python3
"""
Ultralytics Import Helper
========================

Utility functions to help import ultralytics from local source code.
"""

import sys
import os
from pathlib import Path


def setup_local_ultralytics(verbose=True):
    """
    Setup Python path to import ultralytics from local source code.
    
    This function automatically detects and adds the local ultralytics
    source directory to the Python path, allowing you to import from
    the development version instead of the installed package.
    
    Args:
        verbose (bool): If True, prints status messages
        
    Returns:
        bool: True if local ultralytics was found and added to path, False otherwise
    """
    current_dir = Path(__file__).parent
    ultralytics_source_dir = current_dir / "ultralytics"
    
    # Check if local ultralytics exists
    ultralytics_init_file = ultralytics_source_dir / "ultralytics" / "__init__.py"
    
    if ultralytics_init_file.exists():
        # Add to Python path if not already there
        ultralytics_path_str = str(ultralytics_source_dir)
        if ultralytics_path_str not in sys.path:
            sys.path.insert(0, ultralytics_path_str)
            
        if verbose:
            print(f"✅ Using local ultralytics source: {ultralytics_source_dir}")
        return True
    else:
        if verbose:
            print(f"ℹ️ Local ultralytics not found, using installed package")
            print(f"   Looked for: {ultralytics_init_file}")
        return False


def get_ultralytics_info():
    """
    Get information about the currently imported ultralytics package.
    
    Returns:
        dict: Information about ultralytics (path, version, etc.)
    """
    try:
        import ultralytics
        info = {
            'location': ultralytics.__file__,
            'version': getattr(ultralytics, '__version__', 'unknown'),
            'is_local': 'site-packages' not in ultralytics.__file__,
            'package_dir': Path(ultralytics.__file__).parent
        }
        return info
    except ImportError:
        return {'error': 'ultralytics not found'}


def verify_ultralytics_import(verbose=True):
    """
    Verify ultralytics import and display information.
    
    Args:
        verbose (bool): If True, prints detailed information
        
    Returns:
        bool: True if ultralytics is available, False otherwise
    """
    try:
        info = get_ultralytics_info()
        
        if 'error' in info:
            if verbose:
                print(f"❌ {info['error']}")
            return False
        
        if verbose:
            print(f"📦 Ultralytics Package Information:")
            print(f"   Version: {info['version']}")
            print(f"   Location: {info['location']}")
            print(f"   Source: {'Local development' if info['is_local'] else 'Installed package'}")
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"❌ Error checking ultralytics: {e}")
        return False


# Convenience function for quick setup
def quick_setup():
    """
    One-line function to setup local ultralytics and verify import.
    
    Returns:
        module: ultralytics module if successfully imported, None otherwise
    """
    setup_local_ultralytics(verbose=True)
    
    try:
        import ultralytics
        verify_ultralytics_import(verbose=True)
        return ultralytics
    except ImportError as e:
        print(f"❌ Failed to import ultralytics: {e}")
        return None


if __name__ == "__main__":
    """
    Demo of the import helper functions.
    """
    print("🔧 Ultralytics Import Helper Demo")
    print("=" * 40)
    
    # Setup local ultralytics
    found_local = setup_local_ultralytics()
    
    # Verify import
    import_success = verify_ultralytics_import()
    
    if import_success:
        print("\n✅ Ready to use ultralytics!")
        
        # Quick test
        try:
            from ultralytics import YOLO
            print("✅ YOLO import successful")
        except ImportError as e:
            print(f"❌ YOLO import failed: {e}")
    else:
        print("\n❌ Ultralytics setup failed")
        print("💡 Try installing ultralytics: pip install ultralytics")