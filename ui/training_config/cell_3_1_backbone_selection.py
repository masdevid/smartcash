"""
File: smartcash/ui/training_config/cell_3_1_backbone_selection.py
Deskripsi: Cell untuk pemilihan model, backbone dan konfigurasi layer SmartCash dengan yang kompatibel dengan ModelManager
"""

# Import dari utility cell
from IPython.display import display, HTML

# Setup environment dan load config dengan error handling sederhana
try:
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    env, config = setup_notebook_environment("backbone_selection", "configs/model_config.yaml")
    ui_components = setup_ui_component(env, config, "backbone_selection")
    
    # Setup handler secara manual
    from smartcash.ui.training_config.backbone_selection_handler import setup_backbone_selection_handlers
    ui_components = setup_backbone_selection_handlers(ui_components, env, config)
        
    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e:
    display(HTML(f"<div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'><h3>❌ Error Inisialisasi</h3><p>{str(e)}</p></div>"))