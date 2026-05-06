"""
Setup for pothole detection app.
Attempts to build Deformable DETR CUDA extension, but fails gracefully if unavailable.
"""

import os
import sys
from setuptools import setup, find_packages

# Try to build the extension, but don't fail if it doesn't work
ext_modules = []

try:
    import torch
    if torch.cuda.is_available():
        print("CUDA is available, attempting to build extension...")
        from deformable_detr.models.ops.setup import get_extensions
        ext_modules = get_extensions()
    else:
        print("CUDA not available, skipping extension build. Using pure PyTorch fallback.")
except Exception as e:
    print(f"⚠ Warning: Could not build extension: {e}")
    print("Continuing with pure PyTorch fallback...")
    ext_modules = []

setup(
    name='pothole_app',
    version='0.1',
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=[
        'torch>=2.1.0',
        'torchvision>=0.16.0',
        'opencv-python-headless>=4.8.0',
        'easyocr>=1.7.0',
        'streamlit>=1.32.0',
        'folium>=0.14.0',
        'streamlit-folium>=0.15.0',
    ],
)
