"""
File: smartcash/ui/dataset/preprocessing/utils/__init__.py
Deskripsi: Fixed utils module dengan complete function exports dan missing implementations
"""

# Import dari ui_utils
from .ui_utils import (
    clear_outputs, handle_ui_error, show_ui_success, log_to_accordion,
    log_preprocessing_config, display_preprocessing_results,
    show_preprocessing_success, is_milestone_step
)

# Import dari progress_utils
from .progress_utils import (
    create_dual_progress_callback, setup_dual_progress_tracker,
    complete_progress_tracker, error_progress_tracker, reset_progress_tracker,
    ProgressBridgeManager, register_progress_callback_to_backend,
    create_progress_reporter, sync_progress_with_config_operations
)

# Import dari button_manager
from .button_manager import (
    disable_all_buttons, enable_all_buttons, with_backend_operation_management,
    BackendAwareButtonManager, setup_backend_button_management,
    notify_backend_operation_start, notify_backend_operation_complete
)

# Import dari backend_utils
from .backend_utils import (
    validate_dataset_ready, check_preprocessed_exists,
    create_backend_preprocessor, create_backend_checker,
    create_backend_cleanup_service, _convert_ui_to_backend_config
)

# Import dari confirmation_utils
from .confirmation_utils import (
    show_cleanup_confirmation, show_preprocessing_confirmation,
    clear_confirmation_area
)

# Missing functions implementation
def setup_backend_integration(ui_components, config):
    """🔗 Setup complete backend integration"""
    from .backend_utils import setup_backend_integration as backend_setup
    return backend_setup(ui_components, config)

def test_backend_connectivity(ui_components, config):
    """🧪 Test backend connectivity"""
    from .backend_utils import test_backend_connectivity as backend_test
    return backend_test(ui_components, config)

# Export semua functions
__all__ = [
    # UI Utils
    'clear_outputs', 'handle_ui_error', 'show_ui_success', 'log_to_accordion',
    'log_preprocessing_config', 'display_preprocessing_results',
    'show_preprocessing_success', 'is_milestone_step',
    
    # Progress Utils
    'create_dual_progress_callback', 'setup_dual_progress_tracker',
    'complete_progress_tracker', 'error_progress_tracker', 'reset_progress_tracker',
    'ProgressBridgeManager', 'register_progress_callback_to_backend',
    'create_progress_reporter', 'sync_progress_with_config_operations',
    
    # Button Manager
    'disable_all_buttons', 'enable_all_buttons', 'with_backend_operation_management',
    'BackendAwareButtonManager', 'setup_backend_button_management',
    'notify_backend_operation_start', 'notify_backend_operation_complete',
    
    # Backend Utils
    'validate_dataset_ready', 'check_preprocessed_exists',
    'create_backend_preprocessor', 'create_backend_checker',
    'create_backend_cleanup_service', '_convert_ui_to_backend_config',
    'setup_backend_integration', 'test_backend_connectivity',
    
    # Confirmation Utils
    'show_cleanup_confirmation', 'show_preprocessing_confirmation',
    'clear_confirmation_area'
]