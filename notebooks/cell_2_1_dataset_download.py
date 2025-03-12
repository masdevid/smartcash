# Cell 2.1 - Dataset Download
# Download dataset untuk training model SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="dataset_download",
    config_path="configs/dataset_config.yaml",
    create_dirs=["data"]
)

if not config or 'data' not in config:
    raise ValueError("Config tidak valid! Pastikan configs/dataset_config.yaml berisi konfigurasi dataset.")

# Setup UI component
ui_components = setup_ui_component(env, config, "dataset_download")

# Tampilkan UI
display_ui(ui_components)