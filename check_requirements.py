#!/usr/bin/env python3
"""
Script to check and install missing requirements before running evaluate.py
"""

import sys
import subprocess
import importlib
import pkg_resources

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.device_count()} devices")
            return True
        else:
            print("⚠ CUDA not available - GPU required for this script")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False

def check_nltk_data():
    """Check and download required NLTK data"""
    try:
        import nltk
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"⚠ NLTK data download failed: {e}")
        return False

def main():
    print("Checking requirements for AlphaEdit evaluate.py...")
    
    # Critical packages
    critical_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
        ("nltk", "nltk"),
        ("matplotlib", "matplotlib"),
        ("einops", "einops"),
        ("hydra-core", "hydra"),
    ]
    
    missing_packages = []
    for package, import_name in critical_packages:
        if not check_package(import_name):
            missing_packages.append(package)
            print(f"✗ Missing: {package}")
        else:
            print(f"✓ Found: {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    # Check CUDA
    if not check_cuda():
        print("\n⚠ Warning: CUDA not available. Script may fail.")
    
    # Check NLTK data
    check_nltk_data()
    
    print("\n✓ All requirements checked!")
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
