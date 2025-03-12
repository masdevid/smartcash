# Cell 2.1 - Dataset Download
# Persiapan dataset untuk training model SmartCash

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="dataset_download",
    config_path="configs/base_config.yaml",
    create_dirs=["configs", "data"]
)

# Setup UI component
ui_components = setup_ui_component(env, config, "dataset_download")

# Tampilkan UI
display_ui(ui_components)