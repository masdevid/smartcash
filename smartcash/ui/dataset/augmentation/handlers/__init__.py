"""
File: smartcash/ui/dataset/augmentation/handlers/__init__.py
Deskripsi: Handlers module exports dengan backward compatibility dan enhanced integration
"""

from smartcash.ui.dataset.augmentation.handlers.augmentation_handlers import setup_augmentation_handlers
from smartcash.ui.dataset.augmentation.handlers.config_handler import AugmentationConfigHandler
from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
from smartcash.ui.dataset.augmentation.handlers.config_updater import update_augmentation_ui, reset_augmentation_ui
from smartcash.ui.dataset.augmentation.handlers.defaults import get_default_augmentation_config

# Main setup function
def setup_handlers(ui_components, config, env=None):
    """Main setup function dengan backward compatibility"""
    return setup_augmentation_handlers(ui_components, config, env)

# One-liner utilities
create_config_handler = lambda module_name='augmentation', parent_module='dataset': AugmentationConfigHandler(module_name, parent_module)
extract_config = lambda ui_components: extract_augmentation_config(ui_components)
update_ui = lambda ui_components, config: update_augmentation_ui(ui_components, config)
reset_ui = lambda ui_components: reset_augmentation_ui(ui_components)
get_defaults = lambda: get_default_augmentation_config()

__all__ = [
    # Main functions
    'setup_augmentation_handlers',
    'setup_handlers',
    
    # Config management
    'AugmentationConfigHandler',
    'extract_augmentation_config',
    'update_augmentation_ui',
    'reset_augmentation_ui',
    'get_default_augmentation_config',
    
    # One-liner utilities
    'create_config_handler',
    'extract_config',
    'update_ui',
    'reset_ui',
    'get_defaults'
]