"""
File: smartcash/ui/dataset/preprocessing/__init__.py
Deskripsi: Ekspor utilitas dan fungsi preprocessing dataset
"""

# Ekspor fungsi inisialisasi
from smartcash.ui.dataset.preprocessing.preprocessing_initializer import (
    initialize_dataset_preprocessing_ui,
    initialize_preprocessing_ui
)

# Ekspor utilitas logging dan UI state
from smartcash.ui.dataset.preprocessing.utils.logger_helper import (
    log_message,
    is_initialized,
    setup_ui_logger
)

# Ekspor utilitas UI state manager
from smartcash.ui.dataset.preprocessing.utils.ui_state_manager import (
    update_status_panel,
    update_ui_state,
    reset_ui_after_preprocessing,
    update_ui_before_preprocessing,
    is_preprocessing_running,
    set_preprocessing_state,
    show_confirmation,
    ensure_confirmation_area,
    reset_after_operation
)

# Ekspor utilitas progress manager
from smartcash.ui.dataset.preprocessing.utils.progress_manager import (
    update_progress,
    reset_progress_bar,
    start_progress,
    complete_progress,
    create_progress_callback
)

# Ekspor utilitas observer
from smartcash.ui.dataset.preprocessing.utils.ui_observers import (
    register_ui_observers,
    notify_process_start,
    notify_process_complete,
    notify_process_error,
    notify_process_stop,
    MockObserverManager
)

# Ekspor handler utama
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handler import (
    handle_preprocessing_button_click,
    execute_preprocessing,
    get_preprocessing_config_from_ui,
    confirm_preprocessing
)

from smartcash.ui.dataset.preprocessing.handlers.stop_handler import (
    handle_stop_button_click,
    stop_preprocessing
)

from smartcash.ui.dataset.preprocessing.handlers.reset_handler import (
    handle_reset_button_click,
    reset_preprocessing_config
)

from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import (
    handle_cleanup_button_click,
    execute_cleanup,
    cleanup_preprocessed_files,
    start_progress,
    reset_ui_after_cleanup
)

from smartcash.ui.dataset.preprocessing.handlers.save_handler import (
    handle_save_button_click,
    save_preprocessing_config
)

from smartcash.ui.dataset.preprocessing.handlers.config_handler import (
    update_ui_from_config
)

from smartcash.ui.dataset.preprocessing.handlers.setup_handlers import (
    setup_preprocessing_handlers
)

__all__ = [
    # Inisialisasi
    'initialize_dataset_preprocessing_ui',
    'initialize_preprocessing_ui',
    
    # Logging dan UI State
    'log_message',
    'is_initialized',
    'setup_ui_logger',
    'update_status_panel',
    'update_ui_state',
    'reset_ui_after_preprocessing',
    'update_ui_before_preprocessing',
    'is_preprocessing_running',
    'set_preprocessing_state',
    'show_confirmation',
    'ensure_confirmation_area',
    'reset_after_operation',
    
    # Progress
    'update_progress',
    'reset_progress_bar',
    'start_progress',
    'complete_progress',
    'create_progress_callback',
    
    # Observer
    'register_ui_observers',
    'notify_process_start',
    'notify_process_complete',
    'notify_process_error',
    'notify_process_stop',
    'MockObserverManager',
    
    # Handlers - Preprocessing
    'handle_preprocessing_button_click',
    'execute_preprocessing',
    'get_preprocessing_config_from_ui',
    'confirm_preprocessing',
    
    # Handlers - Stop
    'handle_stop_button_click',
    'stop_preprocessing',
    
    # Handlers - Reset
    'handle_reset_button_click',
    'reset_preprocessing_config',
    
    # Handlers - Cleanup
    'handle_cleanup_button_click',
    'execute_cleanup',
    'cleanup_preprocessed_files',
    'reset_ui_after_cleanup',
    
    # Handlers - Save
    'handle_save_button_click',
    'save_preprocessing_config',
    
    # Handlers - Config
    'update_ui_from_config',
    
    # Handlers - Setup
    'setup_preprocessing_handlers'
]
