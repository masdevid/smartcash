"""
File: smartcash/ui/setup/cell_1_2_env_config.py
Deskripsi: Cell konfigurasi environment untuk proyek SmartCash yang terintegrasi dengan tema UI
"""

import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Import komponen UI environment config dengan tema terintegrasi
    from smartcash.ui.setup.env_config import setup_environment_config
    setup_environment_config()
except ImportError as e: err_alert(e)