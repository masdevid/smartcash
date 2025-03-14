# Cell 3.3 - Training Strategy
# Konfigurasi strategi dan teknik optimasi untuk training model SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="training_strategy",
    config_path="configs/training_config.yaml",
    create_dirs=["configs", "runs/train/logs"]
)

# Setup UI component
ui_components = setup_ui_component(env, config, "training_strategy")

# Tampilkan UI
display_ui(ui_components)