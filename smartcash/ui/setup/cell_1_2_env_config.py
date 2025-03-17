"""
File: smartcash/ui/setup/cell_1_1_env_config.py
Deskripsi: Cell konfigurasi environment untuk proyek SmartCash
"""

import sys
import os
import ipywidgets as widgets
from IPython.display import display, HTML
from pathlib import Path

# Pastikan direktori project ada di path
if '.' not in sys.path:
    sys.path.append('.')

def main():
    """
    Konfigurasi environment utama untuk SmartCash.
    
    Returns:
        Dictionary komponen UI untuk konfigurasi environment
    """
    try:
        # Path konfigurasi default
        config_path = "configs/colab_config.yaml"
        
        # Coba gunakan cell_template untuk menjalankan cell
        from smartcash.ui.components.cell_template import run_cell
        ui_components = run_cell("env_config", config_path)
        return ui_components
    except ImportError:
        # Fallback jika template tidak tersedia
        print("Template cell tidak ditemukan, menggunakan implementasi langsung...")
        return setup_manual_env_config()

def setup_manual_env_config():
    """Implementasi manual untuk setup environment config jika template tidak tersedia."""
    try:
        # Import komponen
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
        
        # Coba membaca konfigurasi
        config = {}
        try:
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            try:
                if os.path.exists("configs/colab_config.yaml"):
                    config = config_manager.load_config("configs/colab_config.yaml")
            except:
                pass
        except ImportError:
            pass
        
        # Coba mendapatkan environment manager
        env = None
        try:
            from smartcash.common.environment import get_environment_manager
            env = get_environment_manager()
        except ImportError:
            pass
            
        # Buat UI components
        ui_components = create_env_config_ui(env, config)
        
        # Setup handlers
        ui_components = setup_env_config_handlers(ui_components, env, config)
        
        # Tampilkan UI
        display(ui_components['ui'])
        
        return ui_components
    except Exception as e:
        # Fallback paling dasar - tampilkan pesan error
        error_html = f"""
        <div style="padding: 15px; background-color: #f8d7da; color: #721c24; border-radius: 5px; margin: 10px 0;">
            <h3>❌ Error Konfigurasi Environment</h3>
            <p>{str(e)}</p>
            <p>Pastikan repository SmartCash sudah diclone dengan benar dan semua dependensi terpasang.</p>
        </div>
        """
        display(HTML(error_html))
        return {"error": str(e)}

# Jalankan konfigurasi
main()