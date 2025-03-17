"""
File: smartcash/ui/setup/cell_1_2_env_config.py
Deskripsi: Cell konfigurasi environment untuk proyek SmartCash yang terintegrasi dengan tema UI
"""

import sys
import os

# Pastikan direktori project ada di path
if '.' not in sys.path:
    sys.path.append('.')

# Import komponen UI environment config dengan tema terintegrasi
def setup_environment_config():
    """Jalankan konfigurasi environment dengan komponen UI terintegrasi"""
    try:
        from smartcash.ui.setup.env_config import setup_environment_config
        setup_environment_config()
    except ImportError as e:
        from IPython.display import display, HTML
        display(HTML(f"""
        <div style="padding:10px; background-color:#f8d7da; color:#721c24; border-radius:4px; margin:10px 0">
            <h3 style="margin:5px 0">❌ Error saat import modul</h3>
            <p>Modul environment config tidak ditemukan: {str(e)}</p>
            <p>Pastikan repository SmartCash telah di-clone dengan benar.</p>
        </div>
        """))

# Jalankan konfigurasi environment
setup_environment_config()