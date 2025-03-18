"""
File: smartcash/ui/setup/cell_1_2_env_config.py
Deskripsi: Cell konfigurasi environment untuk proyek SmartCash yang terintegrasi dengan tema UI
"""

from IPython.display import display, HTML
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Import komponen UI environment config dengan tema terintegrasi
    from smartcash.ui.setup.env_config import setup_environment_config
    setup_environment_config()
except ImportError as e:
    display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Inisialisasi</h3><p>{str(e)}</p></div>"))