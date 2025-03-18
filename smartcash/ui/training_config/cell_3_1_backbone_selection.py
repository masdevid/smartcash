"""
File: smartcash/ui/training_config/cell_3_1_backbone_selection.py
Deskripsi: Cell untuk pemilihan backbone dan konfigurasi layer model SmartCash
"""

# Import komponen UI dari smartcash
from IPython.display import display

# Import dari utility cell
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui, cleanup_resources

# Setup environment dan load config
env, config = setup_notebook_environment(
    cell_name="backbone_selection",
    config_path="configs/model_config.yaml"
)

# Setup komponen UI dan handler
ui_components = setup_ui_component(env, config, "backbone_selection")

# Setup handler secara manual
try:
    from smartcash.ui.training_config.backbone_selection_handler import setup_backbone_selection_handlers
    ui_components = setup_backbone_selection_handlers(ui_components, env, config)
except ImportError as e:
    print(f"⚠️ Tidak dapat setup handler backbone_selection: {e}")
    
# Tampilkan UI
display_ui(ui_components)

# Untuk cleanup resources ketika selesai:
# cleanup_resources(ui_components)