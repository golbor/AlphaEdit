#!/usr/bin/env python3
"""
Fix NLTK initialization issues by setting up proper data directory and downloading required data.
"""
import os
import nltk

# Set NLTK data directory
nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download required NLTK data
try:
    nltk.download('punkt', download_dir=nltk_data_dir)
    nltk.download('stopwords', download_dir=nltk_data_dir)
    print("NLTK data downloaded successfully to:", nltk_data_dir)
except Exception as e:
    print("Error downloading NLTK data:", e)

print("NLTK setup complete. Add 'export NLTK_DATA=~/nltk_data' to your shell profile.")
