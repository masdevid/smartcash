"""
File: tests/conftest.py
Deskripsi: Konfigurasi dan fixture untuk testing
"""
import sys
from pathlib import Path

# Tambahkan direktori root ke path
root_dir = str(Path(__file__).parent.absolute())
if root_dir not in sys.path:
    sys.path.append(root_dir)
