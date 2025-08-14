#!/usr/bin/env python3
"""
Fix NLTK initialization issues by setting up proper data directory and downloading required data.
"""
import os
import subprocess
import sys

# Get NLTK data directory from environment
nltk_data_dir = os.environ.get('NLTK_DATA', os.path.expanduser("/home/stud/golab/nltk_data"))
os.makedirs(nltk_data_dir, exist_ok=True)

print(f"NLTK data directory: {nltk_data_dir}")

# Download required NLTK data using subprocess to avoid import issues
try:
    subprocess.run([
        sys.executable, "-c", 
        f"import nltk; nltk.download('punkt', download_dir='{nltk_data_dir}'); nltk.download('stopwords', download_dir='{nltk_data_dir}')"
    ], check=True)
    print("NLTK data downloaded successfully to:", nltk_data_dir)
except subprocess.CalledProcessError as e:
    print("Error downloading NLTK data:", e)

print("NLTK setup complete.")
