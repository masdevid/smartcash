"""
File: smartcash/ui/cells/cell_1_2_env_config.py
Deskripsi: Cell konfigurasi environment terintegrasi dengan sinkronisasi konfigurasi Drive
"""
import sys
from IPython.display import display
if '.' not in sys.path: sys.path.append('.')

try:
    # Import dan gunakan setup environment dari modul
    from smartcash.ui.setup.env_config import setup_environment_config
    
    # Setup dan tampilkan UI environment config (termasuk inisialisasi Drive)
    ui_components = setup_environment_config()
    display(ui_components['ui'])
    
except ImportError as e: create_alert(f"Error: {str(e)}", 'error')