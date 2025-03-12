# Cell 2.2 - Dataset Preprocessing
# Preprocessing dataset untuk training model SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="preprocessing",
    config_path="configs/base_config.yaml"
)

# Setup UI component
ui_components = setup_ui_component(env, config, "preprocessing")

# Tampilkan UI
display_ui(ui_components)