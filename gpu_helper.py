#!/usr/bin/env python3
"""
GPU Selection Helper
===================

Utility functions for automatic GPU detection and selection in segmentation scripts.
"""

import torch
import sys
from pathlib import Path


def detect_available_devices():
    """
    Detect available compute devices (CPU, CUDA GPU, MPS).
    
    Returns:
        dict: Dictionary with device information
    """
    devices = {
        'cpu': True,
        'cuda': False,
        'cuda_count': 0,
        'mps': False,
        'recommended': 'cpu'
    }
    
    # Check CUDA availability
    if torch.cuda.is_available():
        devices['cuda'] = True
        devices['cuda_count'] = torch.cuda.device_count()
        devices['recommended'] = 'cuda'
        
    # Check MPS (Apple Silicon) availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices['mps'] = True
        if devices['recommended'] == 'cpu':  # Only set as recommended if no CUDA
            devices['recommended'] = 'mps'
    
    return devices


def get_device_info(device_name=None):
    """
    Get detailed information about a specific device.
    
    Args:
        device_name (str): Device name ('cpu', 'cuda', 'cuda:0', 'mps')
        
    Returns:
        dict: Device information
    """
    if device_name is None:
        devices = detect_available_devices()
        device_name = devices['recommended']
    
    info = {
        'device': device_name,
        'available': False,
        'name': 'Unknown',
        'memory': 'Unknown'
    }
    
    try:
        if device_name == 'cpu':
            info['available'] = True
            info['name'] = 'CPU'
            
        elif device_name.startswith('cuda'):
            if torch.cuda.is_available():
                device_id = 0
                if ':' in device_name:
                    device_id = int(device_name.split(':')[1])
                
                if device_id < torch.cuda.device_count():
                    info['available'] = True
                    info['name'] = torch.cuda.get_device_name(device_id)
                    
                    # Get memory info
                    memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
                    info['memory'] = f"{memory_gb:.1f} GB"
                    
        elif device_name == 'mps':
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info['available'] = True
                info['name'] = 'Metal Performance Shaders (Apple Silicon)'
                
    except Exception as e:
        info['error'] = str(e)
    
    return info


def select_best_device(prefer_gpu=True, verbose=True):
    """
    Automatically select the best available device.
    
    Args:
        prefer_gpu (bool): Whether to prefer GPU over CPU
        verbose (bool): Whether to print device information
        
    Returns:
        str: Selected device name
    """
    devices = detect_available_devices()
    
    if verbose:
        print("🖥️  Device Detection:")
        print(f"   CPU: Available")
        print(f"   CUDA: {'Available' if devices['cuda'] else 'Not Available'}")
        if devices['cuda']:
            print(f"   CUDA GPUs: {devices['cuda_count']}")
        print(f"   MPS: {'Available' if devices['mps'] else 'Not Available'}")
    
    # Select device based on preference and availability
    if prefer_gpu:
        if devices['cuda']:
            selected = 'cuda'
        elif devices['mps']:
            selected = 'mps'
        else:
            selected = 'cpu'
    else:
        selected = 'cpu'
    
    # Get detailed info about selected device
    info = get_device_info(selected)
    
    if verbose:
        print(f"\n✅ Selected Device: {selected}")
        print(f"   Name: {info['name']}")
        if info['memory'] != 'Unknown':
            print(f"   Memory: {info['memory']}")
    
    return selected


def validate_device(device_name):
    """
    Validate if a device is available and working.
    
    Args:
        device_name (str): Device to validate
        
    Returns:
        bool: True if device is available and working
    """
    try:
        # Try to create a tensor on the device
        test_tensor = torch.tensor([1.0], device=device_name)
        return True
    except Exception as e:
        print(f"❌ Device '{device_name}' validation failed: {e}")
        return False


def setup_device_for_model(model, device=None, verbose=True):
    """
    Setup device for a YOLO model with automatic detection.
    
    Args:
        model: YOLO model instance
        device (str, optional): Specific device to use, or None for auto-detection
        verbose (bool): Whether to print device information
        
    Returns:
        str: Device name that was configured
    """
    if device is None:
        device = select_best_device(verbose=verbose)
    else:
        if verbose:
            info = get_device_info(device)
            if info['available']:
                print(f"✅ Using specified device: {device}")
                print(f"   Name: {info['name']}")
                if info['memory'] != 'Unknown':
                    print(f"   Memory: {info['memory']}")
            else:
                print(f"❌ Specified device '{device}' not available, falling back to auto-detection")
                device = select_best_device(verbose=verbose)
    
    # Validate device works
    if not validate_device(device):
        print(f"⚠️  Device '{device}' validation failed, falling back to CPU")
        device = 'cpu'
    
    # The model device is set during inference with device parameter
    # model.to(device) is typically not needed for YOLO models
    
    return device


def benchmark_devices(test_iterations=5):
    """
    Benchmark available devices for performance comparison.
    
    Args:
        test_iterations (int): Number of test iterations
        
    Returns:
        dict: Benchmark results
    """
    devices = detect_available_devices()
    results = {}
    
    print(f"🏃 Benchmarking devices ({test_iterations} iterations)...")
    
    # Test devices
    test_devices = ['cpu']
    if devices['cuda']:
        test_devices.append('cuda')
    if devices['mps']:
        test_devices.append('mps')
    
    for device in test_devices:
        print(f"\n   Testing {device}...")
        
        try:
            import time
            times = []
            
            for i in range(test_iterations):
                start_time = time.time()
                
                # Simple tensor operations for benchmarking
                x = torch.randn(1000, 1000, device=device)
                y = torch.randn(1000, 1000, device=device)
                z = torch.matmul(x, y)
                
                # Synchronize for accurate timing
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            results[device] = {
                'avg_time': avg_time,
                'times': times,
                'available': True
            }
            
            print(f"      Average time: {avg_time:.4f}s")
            
        except Exception as e:
            results[device] = {
                'error': str(e),
                'available': False
            }
            print(f"      Error: {e}")
    
    # Recommend fastest device
    if results:
        fastest_device = min([d for d in results.keys() if results[d].get('available', False)], 
                           key=lambda d: results[d]['avg_time'])
        print(f"\n🏆 Fastest device: {fastest_device}")
        results['recommended'] = fastest_device
    
    return results


if __name__ == "__main__":
    """
    Demo of GPU selection utilities.
    """
    print("🎯 GPU Selection Helper Demo")
    print("=" * 40)
    
    # Demo 1: Device detection
    print("\n1. Detecting available devices:")
    devices = detect_available_devices()
    for key, value in devices.items():
        print(f"   {key}: {value}")
    
    # Demo 2: Automatic device selection
    print("\n2. Automatic device selection:")
    best_device = select_best_device()
    
    # Demo 3: Device validation
    print(f"\n3. Validating selected device '{best_device}':")
    is_valid = validate_device(best_device)
    print(f"   Valid: {is_valid}")
    
    # Demo 4: Benchmark (optional, can be slow)
    print(f"\n4. Quick benchmark (optional):")
    print("   Uncomment the line below to run device benchmark:")
    print("   # benchmark_results = benchmark_devices(3)")
    
    print(f"\n✅ Demo completed!")
    print(f"\n💡 Usage in segmentation scripts:")
    print(f"   from gpu_helper import select_best_device")
    print(f"   device = select_best_device()")
    print(f"   results = model('image.jpg', device=device)")