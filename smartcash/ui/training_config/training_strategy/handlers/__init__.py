"""
File: smartcash/ui/training_config/training_strategy/handlers/__init__.py
Deskripsi: Export fungsi-fungsi handler untuk training strategy UI
"""

# Default configs
from smartcash.ui.training_config.training_strategy.handlers.default_config import (
    get_default_training_strategy_config,
    get_default_config,
    get_default_base_dir
)

# Config loader and saver
from smartcash.ui.training_config.training_strategy.handlers.config_loader import (
    get_training_strategy_config,
    save_training_strategy_config
)

# UI updater
from smartcash.ui.training_config.training_strategy.handlers.ui_updater import (
    update_ui_from_config
)

# Config extractor
from smartcash.ui.training_config.training_strategy.handlers.config_extractor import (
    update_config_from_ui
)

# Info updater
from smartcash.ui.training_config.training_strategy.handlers.info_updater import (
    update_training_strategy_info
)

# Button handlers
from smartcash.ui.training_config.training_strategy.handlers.button_handlers import (
    on_save_click,
    on_reset_click,
    setup_training_strategy_button_handlers
)

# Form handlers
from smartcash.ui.training_config.training_strategy.handlers.form_handlers import (
    setup_training_strategy_form_handlers
)

# Status handlers
from smartcash.ui.training_config.training_strategy.handlers.status_handlers import (
    add_status_panel,
    update_status_panel,
    clear_status_panel
)

# Sync logger
from smartcash.ui.training_config.training_strategy.handlers.sync_logger import (
    update_sync_status,
    update_sync_status_only,
    log_sync_status
)

# Drive handlers
try:
    from smartcash.ui.training_config.training_strategy.handlers.drive_handlers import (
        sync_to_drive,
        sync_from_drive
    )
except ImportError:
    # Optional - mungkin tidak tersedia di lingkungan non-Colab
    pass

__all__ = [
    # Default configs
    'get_default_training_strategy_config',
    'get_default_config',
    'get_default_base_dir',
    
    # Config loader and saver
    'get_training_strategy_config',
    'save_training_strategy_config',
    
    # UI updater
    'update_ui_from_config',
    
    # Config extractor
    'update_config_from_ui',
    
    # Info updater
    'update_training_strategy_info',
    
    # Button handlers
    'on_save_click',
    'on_reset_click',
    'setup_training_strategy_button_handlers',
    
    # Form handlers
    'setup_training_strategy_form_handlers',
    
    # Status handlers
    'add_status_panel',
    'update_status_panel',
    'clear_status_panel',
    
    # Sync logger
    'update_sync_status',
    'update_sync_status_only',
    'log_sync_status',
]