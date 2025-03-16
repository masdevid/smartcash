"""
Cell 1.2 - Environment Configuration
File: smartcash/ui/cells/setup/cell_1_2_environment_config.py
Deskripsi: Setup lingkungan kerja untuk project SmartCash
"""

from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="env_config",
    config_path="configs/colab_config.yaml"
)

# Setup UI component
ui_components = setup_ui_component(env, config, "env_config")

# Tampilkan UI
display_ui(ui_components)