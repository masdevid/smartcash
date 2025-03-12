# Cell 2.3 - Dataset Split Configuration
# Konfigurasi pembagian dataset untuk training, validation, dan testing

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="dataset_split",
    config_path="configs/dataset_config.yaml",
    create_dirs=["data/splits_backup"]
)

# Setup UI component
ui_components = setup_ui_component(env, config, "split_config")

# Tampilkan UI
display_ui(ui_components)