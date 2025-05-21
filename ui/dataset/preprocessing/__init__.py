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

# Ekspor notification manager
from smartcash.ui.dataset.preprocessing.utils.notification_manager import (
    NotificationManager,
    get_notification_manager
)

# Ekspor handler yang dipecah menjadi modul-modul yang lebih kecil
from smartcash.ui.dataset.preprocessing.handlers import (
    # Button handlers
    handle_preprocessing_button_click,
    
    # Config handlers
    get_preprocessing_config_from_ui,
    update_ui_from_config,
    
    # Confirmation handlers
    confirm_preprocessing,
    
    # Execution handlers
    execute_preprocessing,
    
    # Stop handlers
    handle_stop_button_click,
    stop_preprocessing,
    
    # Reset handlers
    handle_reset_button_click,
    reset_preprocessing_config,
    
    # Cleanup handlers
    handle_cleanup_button_click,
    execute_cleanup,
    cleanup_preprocessed_files,
    
    # Save handlers
    handle_save_button_click,
    save_preprocessing_config,
    
    # Setup handlers
    setup_preprocessing_handlers
)

# Ekspor dari utils atau handlers yang dibutuhkan
from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import reset_ui_after_cleanup

__all__ = [
    # Fungsi inisialisasi
    'initialize_dataset_preprocessing_ui',
    'initialize_preprocessing_ui',
    
    # Utilitas logging
    'log_message',
    'is_initialized',
    'setup_ui_logger',
    
    # Utilitas UI state
    'update_status_panel',
    'update_ui_state',
    'reset_ui_after_preprocessing',
    'update_ui_before_preprocessing',
    'is_preprocessing_running',
    'set_preprocessing_state',
    'show_confirmation',
    'ensure_confirmation_area',
    'reset_after_operation',
    
    # Utilitas progress
    'update_progress',
    'reset_progress_bar',
    'start_progress',
    'complete_progress',
    'create_progress_callback',
    
    # Utilitas observer
    'register_ui_observers',
    'notify_process_start',
    'notify_process_complete',
    'notify_process_error',
    'notify_process_stop',
    'MockObserverManager',
    
    # Notification manager
    'NotificationManager',
    'get_notification_manager',
    
    # Button handlers
    'handle_preprocessing_button_click',
    
    # Config handlers
    'get_preprocessing_config_from_ui',
    'update_ui_from_config',
    
    # Confirmation handlers
    'confirm_preprocessing',
    
    # Execution handlers
    'execute_preprocessing',
    
    # Stop handlers
    'handle_stop_button_click',
    'stop_preprocessing',
    
    # Reset handlers
    'handle_reset_button_click',
    'reset_preprocessing_config',
    
    # Cleanup handlers
    'handle_cleanup_button_click',
    'execute_cleanup',
    'cleanup_preprocessed_files',
    'reset_ui_after_cleanup',
    
    # Save handlers
    'handle_save_button_click',
    'save_preprocessing_config',
    
    # Setup handlers
    'setup_preprocessing_handlers'
]
