#!/usr/bin/env python3
"""
Quick test script to verify Vehicle Counter Web App setup
"""

import sys
import os

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing Python imports...")
    
    required_packages = [
        ('flask', 'Flask'),
        ('flask_socketio', 'Flask-SocketIO'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('werkzeug', 'Werkzeug'),
    ]
    
    all_ok = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_files():
    """Test that required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        'app.py',
        'counter_worker.py',
        'directional_counter.py',
        'requirements.txt',
        'templates/index.html',
        'static/css/style.css',
        'static/js/app.js',
    ]
    
    all_ok = True
    
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✓ {filepath}")
        else:
            print(f"  ✗ {filepath} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_directories():
    """Test that required directories exist"""
    print("\nChecking directories...")
    
    required_dirs = [
        'uploads',
        'outputs',
        'templates',
        'static/css',
        'static/js',
    ]
    
    all_ok = True
    
    for dirpath in required_dirs:
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            print(f"  ✓ {dirpath}/")
        else:
            print(f"  ✗ {dirpath}/ - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_models():
    """Test that model files exist"""
    print("\nChecking model files...")
    
    model_pairs = [
        ('yolov8_f_512.py', 'yolov8_f_512.pth', 'Femto 512'),
        ('yolov8_m.py', 'yolov8_m.pth', 'Medium'),
        ('yolov8_l_mobilenet_v2_512x288_indices_246.py', 
         'yolov8_l_mobilenet_v2_512x288_indices_246.pth', 'Large'),
    ]
    
    found_models = 0
    
    for config, checkpoint, name in model_pairs:
        if os.path.exists(config) and os.path.exists(checkpoint):
            print(f"  ✓ {name} model")
            found_models += 1
        else:
            print(f"  ⚠ {name} model - NOT FOUND (optional)")
    
    if found_models == 0:
        print("  ⚠ No models found - you'll need at least one model to run")
        return False
    
    return True


def test_opencv():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test encoding
        ret, buffer = cv2.imencode('.jpg', test_img)
        
        if ret:
            print("  ✓ OpenCV JPEG encoding works")
            return True
        else:
            print("  ✗ OpenCV JPEG encoding failed")
            return False
            
    except Exception as e:
        print(f"  ✗ OpenCV test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Vehicle Counter Web App - Installation Test")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Files", test_files()))
    results.append(("Directories", test_directories()))
    results.append(("Models", test_models()))
    results.append(("OpenCV", test_opencv()))
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {test_name}: {status}")
        if not result:
            all_passed = False
    
    print()
    
    if all_passed:
        print("✓ All tests passed! You're ready to run the application.")
        print()
        print("To start the server:")
        print("  python3 app.py")
        print()
        print("Then open: http://localhost:5000")
        return 0
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print()
        print("Run setup.sh to create missing directories:")
        print("  ./setup.sh")
        print()
        print("Install missing packages:")
        print("  pip3 install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
