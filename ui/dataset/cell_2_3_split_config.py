"""
File: smartcash/ui/dataset/cell_2_3_split_config.py
Deskripsi: Cell untuk konfigurasi pembagian dataset SmartCash dengan struktur modular yang dioptimalkan
"""

# Import dasar
from IPython.display import display, HTML
import sys
if '.' not in sys.path: sys.path.append('.')

# Cell template untuk konfigurasi split
try:
    # Setup environment dan komponen UI dengan dukungan fallback
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    
    # Setup environment dan load config
    env, config = setup_notebook_environment("split_config", "configs/dataset_config.yaml")
    
    # Buat direktori 'configs' jika belum ada
    import os
    os.makedirs('configs', exist_ok=True)
    
    # Setup komponen UI dari split_config_component
    try:
        from smartcash.ui.dataset.split_config_component import create_split_config_ui
        ui_components = create_split_config_ui(env, config)
    except ImportError:
        ui_components = setup_ui_component(env, config, "split_config")
    
    # Setup handler untuk komponen UI
    from smartcash.ui.dataset.split_config_handler import setup_split_config_handlers
    ui_components = setup_split_config_handlers(ui_components, env, config)
    
    # Tampilkan UI
    display_ui(ui_components)
    
except Exception as e:
    # Tampilkan error dengan HTML sederhana
    from IPython.display import HTML
    display(HTML(f"""
    <div style='padding:15px; background:#f8d7da; color:#721c24; border-radius:5px; margin:10px 0;'>
        <h3>❌ Error Inisialisasi Split Config</h3>
        <p>{str(e)}</p>
        <p><strong>Tip:</strong> Pastikan semua file komponen 'split_config' sudah ada di struktur proyek.</p>
    </div>
    """))