"""
File: smartcash/ui/training_config/cell_3_1_backbone_selection.py
Deskripsi: Cell untuk pemilihan model, backbone dan konfigurasi layer SmartCash yang kompatibel dengan ModelManager
"""

# Import dasar dengan error handling yang konsisten
from IPython.display import display, HTML
import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Tambahkan CSS styles untuk UI yang konsisten
    from smartcash.ui.utils.ui_helpers import inject_css_styles
    inject_css_styles()
    
    # Setup environment dan load config
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    env, config = setup_notebook_environment("backbone_selection", "configs/model_config.yaml")
    
    # Setup UI components
    ui_components = setup_ui_component(env, config, "backbone_selection")
    
    # Setup handler secara manual dengan error handling yang lebih baik
    try:
        from smartcash.ui.training_config.backbone_selection_handler import setup_backbone_selection_handlers
        ui_components = setup_backbone_selection_handlers(ui_components, env, config)
        
        # Cek apakah handler berhasil setup
        if 'layer_summary' in ui_components:
            with ui_components['layer_summary']:
                display(HTML("<p>✅ Handler berhasil diinisialisasi</p>"))
    except Exception as e:
        if 'status' in ui_components:
            with ui_components['status']:
                display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Setup Handler</h3><p>{str(e)}</p></div>"))
        else:
            display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Setup Handler</h3><p>{str(e)}</p></div>"))
        
    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e:
    display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Import Module</h3><p>{str(e)}</p></div>"))
except Exception as e:
    display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Inisialisasi</h3><p>{str(e)}</p></div>"))