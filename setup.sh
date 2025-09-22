#!/bin/bash

# MozzaTong Setup Script
# Automated setup for MozzaTong development and usage

set -e  # Exit on any error

echo "🍕 MozzaTong Setup Script"
echo "========================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "Checking Python version..."
if ! command_exists python3; then
    echo "❌ Error: Python 3 is required but not installed."
    echo "Please install Python 3.8 or later and try again."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Found Python $PYTHON_VERSION"

# Check if version is at least 3.8
if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)'; then
    echo "✅ Python version is compatible"
else
    echo "❌ Error: Python 3.8 or later is required"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected."
    echo "It's recommended to use a virtual environment."
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled. Please create a virtual environment first:"
        echo "  python3 -m venv mozzatong-env"
        echo "  source mozzatong-env/bin/activate"
        echo "  bash setup.sh"
        exit 1
    fi
fi

# Install PyTorch based on system
echo "Installing PyTorch..."
if command_exists nvidia-smi; then
    echo "✅ NVIDIA GPU detected, installing PyTorch with CUDA support"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "ℹ️  No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install MozzaTong in development mode
echo "Installing MozzaTong..."
pip install -e .

# Create necessary directories
echo "Creating directories..."
mkdir -p models datasets logs examples

# Test installation
echo "Testing installation..."
if python -c "import mozzatong; print('✅ MozzaTong imported successfully')"; then
    echo "✅ Installation successful!"
else
    echo "❌ Installation failed - import test failed"
    exit 1
fi

# Test CLI
if mozzatong --version >/dev/null 2>&1; then
    echo "✅ CLI tool working correctly"
else
    echo "❌ CLI tool not working correctly"
    exit 1
fi

echo ""
echo "🎉 MozzaTong setup completed successfully!"
echo ""
echo "Quick start:"
echo "  1. Generate a dataset:    mozzatong generate-dataset -n 100"
echo "  2. Run example:          python examples/usage_example.py"
echo "  3. Get help:             mozzatong --help"
echo ""
echo "For training, you'll need a dataset:"
echo "  mozzatong generate-dataset -n 1000 -o ./datasets/my_dataset"
echo "  mozzatong train -d ./datasets/my_dataset/train_dataset.jsonl"
echo ""
echo "Happy coding with MozzaTong! 🍕✨"