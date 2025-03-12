# Cell 3.2: Hyperparameters Template for Hyperparameters & Dataset Download
from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment with appropriate config
env, config = setup_notebook_environment(
    cell_name="hyperparameters",
    config_path="configs/base_config.yaml",
    create_dirs=["data"]
)

# First demonstrate the fixed hyperparameters component
print("🧮 Loading Hyperparameters UI...")
ui_components = setup_ui_component(env, config, "hyperparameters")
display_ui(ui_components)
