# Cell 3.1 - Backbone Selection
# Pemilihan backbone dan konfigurasi layer untuk model SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="backbone_selection",
    config_path="configs/model_config.yaml",
    create_dirs=["configs", "runs/train/weights"]
)

# Setup UI component
ui_components = setup_ui_component(env, config, "backbone_selection")

# Tampilkan UI
display_ui(ui_components)