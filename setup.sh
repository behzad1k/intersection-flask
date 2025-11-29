#!/bin/bash

# Vehicle Counter Web App Setup Script

echo "=================================="
echo "Vehicle Counter - Setup"
echo "=================================="
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads
mkdir -p outputs
mkdir -p static/css
mkdir -p static/js
mkdir -p templates

echo "✓ Directories created"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

echo ""

# Check for model files
echo "Checking for model files..."
models_found=0

if [ -f "yolov8_f_512.py" ] && [ -f "yolov8_f_512.pth" ]; then
    echo "✓ Found femto_512 model"
    ((models_found++))
fi

if [ -f "yolov8_m.py" ] && [ -f "yolov8_m.pth" ]; then
    echo "✓ Found medium model"
    ((models_found++))
fi

if [ -f "yolov8_l_mobilenet_v2_512x288_indices_246.py" ] && [ -f "yolov8_l_mobilenet_v2_512x288_indices_246.pth" ]; then
    echo "✓ Found large model"
    ((models_found++))
fi

if [ $models_found -eq 0 ]; then
    echo "⚠ Warning: No model files found!"
    echo "  Please ensure you have model .py and .pth files in this directory"
fi

echo ""

# Check for directional_counter.py
if [ -f "directional_counter.py" ]; then
    echo "✓ Found directional_counter.py"
else
    echo "✗ Missing directional_counter.py"
    echo "  Please ensure the original counter script is in this directory"
    exit 1
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To start the application:"
echo "  python3 app.py"
echo ""
echo "Then open your browser to:"
echo "  http://localhost:5000"
echo ""
