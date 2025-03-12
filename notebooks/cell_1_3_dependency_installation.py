# Cell 1.3 - Dependency Installation
# Instalasi package yang diperlukan untuk SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="dependency_installer",
    config_path="configs/preprocessing_config.yaml"
)
print(config)

# Setup UI component
ui_components = setup_ui_component(env, config, "dependency_installer")
# Tampilkan UI
display_ui(ui_components)