"""
File: cell_2_1_dataset_download.py
Deskripsi: Cell untuk download dataset SmartCash dengan kode minimal
"""

# Import dasar
from IPython.display import display
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Setup environment dan komponen UI
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    
    # Setup environment dan load config
    env, config = setup_notebook_environment("dataset_download", "configs/colab_config.yaml")
    
    # Setup komponen UI
    ui_components = setup_ui_component(env, config, "dataset_download")
    
    # Setup dataset handler
    from smartcash.ui.dataset.dataset_download_handler import setup_dataset_download_handlers
    ui_components = setup_dataset_download_handlers(ui_components, env, config)
    
    # Tampilkan UI
    display_ui(ui_components)
    
except ImportError as e:
    from IPython.display import HTML
    display(HTML(f"""
    <div style="padding:10px; background:#f8d7da; color:#721c24; border-radius:5px; margin:10px 0">
        <h3 style="margin-top:0">❌ Error Inisialisasi</h3>
        <p>{str(e)}</p>
        <p>Pastikan repository SmartCash telah di-clone dengan benar.</p>
    </div>
    """))