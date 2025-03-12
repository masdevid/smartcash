# Cell 2.4 - Dataset Augmentation
# Augmentasi dataset untuk meningkatkan variasi dan jumlah data training

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment dengan config khusus augmentasi
env, config = setup_notebook_environment(
    cell_name="augmentation",
    config_path="configs/augmentation_config.yaml",
    create_dirs=["data/augmented"]
)

# Setup UI component dengan config
ui_components = setup_ui_component(env, config, "augmentation")

# Tampilkan UI
display_ui(ui_components)