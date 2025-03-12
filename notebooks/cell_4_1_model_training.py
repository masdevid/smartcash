# Cell 3.4 - Model Training Execution
# Eksekusi training model dengan konfigurasi yang sudah disiapkan

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="model_training",
    config_path="configs/training_config.yaml", 
    create_dirs=["runs/train/weights", "logs/training"]
)

# Setup UI component
ui_components = setup_ui_component(env, config, "model_training")

# Display UI
display_ui(ui_components)