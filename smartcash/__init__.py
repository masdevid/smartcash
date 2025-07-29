"""
File: smartcash/__init__.py
Deskripsi: File inisialisasi untuk package SmartCash dengan impor exceptions untuk akses global
"""

# Fix OpenMP duplicate library issue for the entire package
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__version__ = '2.0.0'

# Import subpackages to make them available at the package level
from . import ui

__all__ = ['ui', '__version__']