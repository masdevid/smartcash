"""
File: smartcash/ui/setup/cell_1_2_env_config.py
Deskripsi: Cell konfigurasi environment untuk proyek SmartCash
"""

import sys
import os

# Pastikan direktori project ada di path
if '.' not in sys.path:
    sys.path.append('.')

# Import komponen UI environment config
from smartcash.ui.setup.env_config import setup_environment_config

# Jalankan konfigurasi environment
setup_environment_config()