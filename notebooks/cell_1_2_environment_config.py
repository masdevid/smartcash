# Cell 1.2 - Environment Configuration
# Setup lingkungan kerja untuk project SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="env_config",
    config_path="configs/base_config.yaml"
)

# Setup UI component
ui_components = setup_ui_component(env, config, "env_config")

# Tampilkan UI
display_ui(ui_components)