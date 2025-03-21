"""
File: smartcash/ui/cells/cell_2_2_preprocessing.py
Deskripsi: Cell minimal untuk preprocessing dataset SmartCash
"""

import sys
if '.' not in sys.path: sys.path.append('.')

try:
    # Import utilitas cell dan komponen UI
    from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
    
    # Setup environment dan load config
    env, config = setup_notebook_environment("preprocessing", "configs/preprocessing_config.yaml")
    
    # Setup komponen UI
    ui_components = setup_ui_component(env, config, "preprocessing")
    
    # Setup handler
    from smartcash.ui.dataset.preprocessing_handler import setup_preprocessing_handlers
    ui_components = setup_preprocessing_handlers(ui_components, env, config)
    
    # Tampilkan UI
    display_ui(ui_components)

except ImportError as e: create_alert(e, 'error')