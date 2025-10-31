#!/usr/bin/env python3
"""
Output Directory Helper
======================

Utility functions for managing output directories in segmentation scripts.
"""

import os
from pathlib import Path
from datetime import datetime


def get_timestamped_filename(filename, script_name=""):
    """
    Generate a timestamped filename for shared outputs directory.
    
    Args:
        filename (str): Original filename
        script_name (str): Name of script generating the file (for identification)
        
    Returns:
        str: Timestamped filename
    """
    file_path = Path(filename)
    stem = file_path.stem
    extension = file_path.suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if script_name:
        return f"{script_name}_{stem}_{timestamp}{extension}"
    else:
        return f"{stem}_{timestamp}{extension}"


def create_output_directory(base_name="segmentation_output", add_timestamp=True, use_shared_outputs=True):
    """
    Create an output directory for segmentation results.
    
    Args:
        base_name (str): Base name for the output directory
        add_timestamp (bool): Whether to add timestamp to make directory unique
        use_shared_outputs (bool): Whether to use shared "outputs" directory
        
    Returns:
        Path: Path to the created output directory
    """
    if use_shared_outputs:
        # Use shared outputs directory
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    else:
        # Use individual timestamped directories (legacy behavior)
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dir_name = f"{base_name}_{timestamp}"
        else:
            dir_name = base_name
        
        output_dir = Path(dir_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir


def get_output_path(output_dir, filename, prefix="", suffix="", make_unique=True):
    """
    Get a complete output path for a file within the output directory.
    
    Args:
        output_dir (Path): Output directory path
        filename (str): Original filename
        prefix (str): Prefix to add to filename
        suffix (str): Suffix to add before extension
        make_unique (bool): Whether to make filename unique if it already exists
        
    Returns:
        Path: Complete output path
    """
    file_path = Path(filename)
    stem = file_path.stem
    extension = file_path.suffix
    
    new_filename = f"{prefix}{stem}{suffix}{extension}"
    output_path = output_dir / new_filename
    
    # Make unique if file already exists and make_unique is True
    if make_unique and output_path.exists():
        counter = 1
        while True:
            unique_filename = f"{prefix}{stem}{suffix}_{counter:03d}{extension}"
            unique_path = output_dir / unique_filename
            if not unique_path.exists():
                return unique_path
            counter += 1
    
    return output_path


def setup_output_structure(base_output_dir="outputs", use_shared=True):
    """
    Create a structured output directory with subdirectories.
    
    Args:
        base_output_dir (str): Base output directory name
        use_shared (bool): Whether to use shared outputs directory
        
    Returns:
        dict: Dictionary with paths to different output subdirectories
    """
    if use_shared:
        # Use shared outputs directory with subdirectories
        base_dir = Path("outputs")
    else:
        # Use timestamped directory (legacy behavior)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path(f"{base_output_dir}_{timestamp}")
    
    # Create subdirectories
    subdirs = {
        'base': base_dir,
        'images': base_dir / 'images',
        'videos': base_dir / 'videos',
        'visualizations': base_dir / 'visualizations',
        'data': base_dir / 'data',
        'logs': base_dir / 'logs'
    }
    
    # Create all directories
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def print_output_summary(output_dir, files_created):
    """
    Print a summary of created output files.
    
    Args:
        output_dir (Path): Output directory path
        files_created (list): List of created file names
    """
    print(f"\n📁 Output Summary:")
    print(f"   Directory: {output_dir.absolute()}")
    print(f"   Files created: {len(files_created)}")
    
    for file in files_created:
        file_path = output_dir / file if not str(file).startswith(str(output_dir)) else file
        file_size = "unknown"
        try:
            if Path(file_path).exists():
                size_bytes = Path(file_path).stat().st_size
                if size_bytes > 1024*1024:
                    file_size = f"{size_bytes/(1024*1024):.1f}MB"
                elif size_bytes > 1024:
                    file_size = f"{size_bytes/1024:.1f}KB"
                else:
                    file_size = f"{size_bytes}B"
        except:
            pass
        
        print(f"   - {Path(file_path).name} ({file_size})")


if __name__ == "__main__":
    """
    Demo of output directory utilities.
    """
    print("📁 Output Directory Helper Demo")
    print("=" * 40)
    
    # Demo 1: Shared output directory
    print("\n1. Creating shared output directory:")
    output_dir = create_output_directory()
    print(f"   Created: {output_dir}")
    
    # Demo 2: Structured output with subdirectories
    print("\n2. Creating structured shared output:")
    structure = setup_output_structure()
    print(f"   Base directory: {structure['base']}")
    print(f"   Subdirectories: {list(structure.keys())[1:]}")
    
    # Demo 3: Timestamped filename generation
    print("\n3. Generating timestamped filenames:")
    test_file = "sample_image.jpg"
    timestamped_filename = get_timestamped_filename(test_file, "demo")
    print(f"   Original: {test_file}")
    print(f"   Timestamped: {timestamped_filename}")
    
    # Demo 4: Output path generation with uniqueness
    print("\n4. Generating unique output paths:")
    output_path = get_output_path(output_dir, test_file, prefix="processed_", make_unique=True)
    print(f"   Original: {test_file}")
    print(f"   Output path: {output_path}")
    
    print("\n✅ Demo completed!")