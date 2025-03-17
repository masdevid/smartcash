"""
File: smartcash/ui/01_setup/env_config_cell.py
Deskripsi: Cell notebook untuk konfigurasi environment SmartCash
"""

# !pip install -q ipywidgets tqdm pyyaml matplotlib seaborn
# Tambahkan import dan inisialisasi
import sys
import os
from pathlib import Path
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

def run_env_config_cell():
    """
    Cell runner untuk konfigurasi environment
    """
    # Pastikan smartcash dalam path
    if '.' not in sys.path:
        sys.path.append('.')
        
    # Cek jika direktori SmartCash ada
    if not Path('smartcash').exists():
        display(HTML("""
        <div style="padding:15px;background-color:#f8d7da;border-left:4px solid #721c24;color:#721c24;margin:10px 0;border-radius:4px">
            <h3 style="margin-top:0">❌ Folder SmartCash tidak ditemukan!</h3>
            <p>Repository belum di-clone dengan benar. Silakan jalankan cell clone repository terlebih dahulu.</p>
            <ol>
                <li>Jalankan cell repository clone (Cell 1.1)</li>
                <li>Restart runtime (Runtime > Restart runtime)</li>
                <li>Jalankan kembali notebook dari awal</li>
            </ol>
        </div>
        """))
        return
        
    try:
        # Import dari smartcash
        from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handler import setup_env_config_handlers
        
        # Setup environment dan konfigurasi
        env, config = setup_notebook_environment('env_config', 'configs/colab_config.yaml')
        
        # Buat komponen UI
        ui_components = create_env_config_ui(env, config)
        
        # Setup handlers
        ui_components = setup_env_config_handlers(ui_components, env, config)
        
        # Tampilkan UI
        display_ui(ui_components)
        
        # Register cleanup untuk menghapus resources saat exit (opsional)
        try:
            from smartcash.components.observer.cleanup_observer import register_cleanup_function
            if 'cleanup' in ui_components and callable(ui_components['cleanup']):
                register_cleanup_function(ui_components['cleanup'])
        except ImportError:
            pass
            
        # Return komponen UI untuk penggunaan lebih lanjut
        return ui_components
        
    except Exception as e:
        from smartcash.ui.setup.env_config_fallback import handle_fallback_environment
        return handle_fallback_environment(e)

# Jalankan cell
ui_components = run_env_config_cell()