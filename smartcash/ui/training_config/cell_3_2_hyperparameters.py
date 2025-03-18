"""
File: smartcash/ui/training_config/cell_3_2_hyperparameters.py
Deskripsi: Cell untuk konfigurasi hyperparameter training model SmartCash
"""

# Import komponen UI dari smartcash
from IPython.display import display

# Import dari utility cell
from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui, cleanup_resources

# Setup environment dan load config
env, config = setup_notebook_environment(
    cell_name="hyperparameters",
    config_path="configs/training_config.yaml"
)

# Setup komponen UI dan handler
ui_components = setup_ui_component(env, config, "hyperparameters")

# Setup handler secara manual
try:
    from smartcash.ui.training_config.hyperparameters_handler import setup_hyperparameters_handlers
    ui_components = setup_hyperparameters_handlers(ui_components, env, config)
except ImportError as e:
    print(f"⚠️ Tidak dapat setup handler hyperparameters: {e}")
    
# Tampilkan UI
display_ui(ui_components)

# Untuk cleanup resources ketika selesai:
# cleanup_resources(ui_components)