"""
File: smartcash/ui/dataset/cell_2_2_preprocessing.py
Deskripsi: Cell untuk preprocessing dataset SmartCash
"""

# Import komponen UI dari smartcash
from IPython.display import display

# Import dari utility cell
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui, cleanup_resources

# Setup environment dan load config
env, config = setup_notebook_environment(
    cell_name="preprocessing",
    config_path="configs/preprocessing_config.yaml"
)

# Setup komponen UI dan handler
ui_components = setup_ui_component(env, config, "preprocessing")

# Setup handler secara manual jika tidak otomatis
try:
    from smartcash.ui.dataset.preprocessing_handler import setup_preprocessing_handlers
    ui_components = setup_preprocessing_handlers(ui_components, env, config)
except ImportError as e:
    print(f"⚠️ Tidak dapat setup handler preprocessing: {e}")
    
# Tampilkan UI
display_ui(ui_components)

# Untuk cleanup resources ketika selesai:
# cleanup_resources(ui_components)