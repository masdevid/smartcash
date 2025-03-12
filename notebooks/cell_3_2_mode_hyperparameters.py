# Cell 3.2 - Model Hyperparameters
# Konfigurasi hyperparameter model untuk training SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="hyperparameters",
    config_path="configs/training_config.yaml",
    create_dirs=["configs", "runs/train/checkpoints"]
)

# Setup UI component
ui_components = setup_ui_component(env, config, "hyperparameters")

# Tampilkan UI
display_ui(ui_components)