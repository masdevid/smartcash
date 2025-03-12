# Cell 2.4 - Data Augmentation
# Augmentasi dataset untuk meningkatkan variasi dan jumlah data training

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="data_augmentation",
    config_path="configs/augmentation_config.yaml",
    create_dirs=["data/augmented", "data/backup/augmentation"]
)

# Setup UI component
ui_components = setup_ui_component(env, config, "augmentation")

# Display UI
display_ui(ui_components)