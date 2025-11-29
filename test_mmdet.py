#!/usr/bin/env python3
import sys

print("Testing MMDetection installation...\n")

# Test 1: Import mmcv ops
try:
    from mmcv.ops import roi_align
    print("✓ mmcv.ops (C++ extensions) - OK")
except Exception as e:
    print(f"✗ mmcv.ops failed: {e}")
    sys.exit(1)

# Test 2: Import mmdet
try:
    from mmdet.apis import init_detector, inference_detector
    print("✓ mmdet.apis - OK")
except Exception as e:
    print(f"✗ mmdet.apis failed: {e}")
    sys.exit(1)

# Test 3: Check versions
try:
    import mmcv
    import mmdet
    import torch
    print(f"\n✓ PyTorch: {torch.__version__}")
    print(f"✓ MMCV: {mmcv.__version__}")
    print(f"✓ MMDet: {mmdet.__version__}")
except Exception as e:
    print(f"✗ Version check failed: {e}")

print("\n✅ All tests passed! Ready to run the web app.")