"""
File: smartcash/ui/cells/cell_2_4_augmentation.py
Deskripsi: Cell untuk augmentasi dataset SmartCash dengan kode minimal
"""

import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Import utilitas cell dan komponen UI
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui

    # Setup environment dan load config
    env, config = setup_notebook_environment("augmentation", "configs/augmentation_config.yaml")

    # Setup komponen UI
    ui_components = setup_ui_component(env, config, "augmentation")

    # Setup handler
    from smartcash.ui.dataset.augmentation_handler import setup_augmentation_handlers
    ui_components = setup_augmentation_handlers(ui_components, env, config)

    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e:
    print(f"❌ Error: {e}")