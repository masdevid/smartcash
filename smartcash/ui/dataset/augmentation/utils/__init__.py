"""
File: smartcash/ui/dataset/augmentation/utils/__init__.py
Deskripsi: Utils module exports dengan enhanced integration dan backward compatibility
"""

# Core UI utilities
from smartcash.ui.dataset.augmentation.utils.ui_utils import (
    log_to_ui, log_to_accordion, get_widget_value_safe, extract_augmentation_types,
    validate_form_inputs, update_button_states, clear_ui_outputs, show_validation_errors,
    handle_ui_error, create_config_summary_html, show_config_summary
)

# Progress utilities
from smartcash.ui.dataset.augmentation.utils.progress_utils import (
    create_unified_progress_manager, create_backend_communicator, 
    UnifiedProgressManager, BackendProgressCommunicator
)

# Backend integration
from smartcash.ui.dataset.augmentation.utils.backend_utils import (
    create_service_from_ui, create_service_config, validate_service_config, get_dataset_status
)

# Operation utilities
from smartcash.ui.dataset.augmentation.utils.operation_utils import (
    execute_augmentation, execute_enhanced_check, execute_cleanup_with_progress
)

# Button management
try:
    from smartcash.ui.dataset.augmentation.utils.button_manager import (
        ButtonManager, OperationManager
    )
    BUTTON_MANAGER_AVAILABLE = True
except ImportError:
    BUTTON_MANAGER_AVAILABLE = False

# Dialog utilities
try:
    from smartcash.ui.dataset.augmentation.utils.dialog_utils import (
        show_cleanup_confirmation, show_reset_confirmation, show_validation_errors_dialog,
        show_operation_progress, update_operation_progress, complete_operation_progress
    )
    DIALOG_UTILS_AVAILABLE = True
except ImportError:
    DIALOG_UTILS_AVAILABLE = False

# Backend communicator
try:
    from smartcash.ui.dataset.augmentation.utils.backend_communicator import (
        BackendProgressCommunicator, ServiceIntegrator, create_backend_communicator, create_service_integrator
    )
    BACKEND_COMMUNICATOR_AVAILABLE = True
except ImportError:
    BACKEND_COMMUNICATOR_AVAILABLE = False

# One-liner utilities untuk common operations
safe_get_value = lambda ui_components, key, default=None: get_widget_value_safe(ui_components, key, default)
safe_log = lambda ui_components, msg, level='info': log_to_ui(ui_components, msg, level)
clear_outputs = lambda ui_components: clear_ui_outputs(ui_components)
validate_form = lambda ui_components: validate_form_inputs(ui_components)
show_error = lambda ui_components, msg: handle_ui_error(ui_components, msg)

# Progress management one-liners
create_progress_manager = lambda ui_components: create_unified_progress_manager(ui_components)
create_communicator = lambda ui_components: create_backend_communicator(ui_components) if BACKEND_COMMUNICATOR_AVAILABLE else None

# Service integration one-liners
create_service = lambda ui_components: create_service_from_ui(ui_components)
validate_service = lambda ui_components: validate_service_config(create_service_config(ui_components))

# Operation execution one-liners
run_augmentation = lambda ui_components: execute_augmentation(ui_components)
run_check = lambda ui_components: execute_enhanced_check(ui_components)
run_cleanup = lambda ui_components: execute_cleanup_with_progress(ui_components)

# Button management one-liners (jika tersedia)
if BUTTON_MANAGER_AVAILABLE:
    create_button_manager = lambda ui_components: ButtonManager(ui_components)
    create_operation_manager = lambda ui_components: OperationManager(ui_components)
else:
    create_button_manager = lambda ui_components: None
    create_operation_manager = lambda ui_components: None

# Dialog utilities one-liners (jika tersedia)
if DIALOG_UTILS_AVAILABLE:
    show_cleanup_dialog = lambda ui_components, on_confirm, on_cancel=None: show_cleanup_confirmation(ui_components, on_confirm, on_cancel)
    show_reset_dialog = lambda ui_components, on_confirm, on_cancel=None: show_reset_confirmation(ui_components, on_confirm, on_cancel)
    show_validation_dialog = lambda ui_components, validation_result: show_validation_errors_dialog(ui_components, validation_result)
else:
    show_cleanup_dialog = lambda ui_components, on_confirm, on_cancel=None: None
    show_reset_dialog = lambda ui_components, on_confirm, on_cancel=None: None
    show_validation_dialog = lambda ui_components, validation_result: None

# Availability info
def get_utils_availability() -> Dict[str, bool]:
    """Get availability status dari semua utils modules"""
    return {
        'button_manager': BUTTON_MANAGER_AVAILABLE,
        'dialog_utils': DIALOG_UTILS_AVAILABLE,
        'backend_communicator': BACKEND_COMMUNICATOR_AVAILABLE,
        'core_utils': True,
        'progress_utils': True,
        'backend_utils': True,
        'operation_utils': True
    }

# Comprehensive utilities export
__all__ = [
    # Core utilities
    'log_to_ui', 'log_to_accordion', 'get_widget_value_safe', 'extract_augmentation_types',
    'validate_form_inputs', 'update_button_states', 'clear_ui_outputs', 'show_validation_errors',
    'handle_ui_error', 'create_config_summary_html', 'show_config_summary',
    
    # Progress utilities
    'create_unified_progress_manager', 'create_backend_communicator', 
    'UnifiedProgressManager', 'BackendProgressCommunicator',
    
    # Backend utilities
    'create_service_from_ui', 'create_service_config', 'validate_service_config', 'get_dataset_status',
    
    # Operation utilities
    'execute_augmentation', 'execute_enhanced_check', 'execute_cleanup_with_progress',
    
    # One-liner utilities
    'safe_get_value', 'safe_log', 'clear_outputs', 'validate_form', 'show_error',
    'create_progress_manager', 'create_communicator', 'create_service', 'validate_service',
    'run_augmentation', 'run_check', 'run_cleanup',
    'create_button_manager', 'create_operation_manager',
    'show_cleanup_dialog', 'show_reset_dialog', 'show_validation_dialog',
    
    # Availability info
    'get_utils_availability',
    'BUTTON_MANAGER_AVAILABLE', 'DIALOG_UTILS_AVAILABLE', 'BACKEND_COMMUNICATOR_AVAILABLE'
]

# Conditional exports berdasarkan availability
if BUTTON_MANAGER_AVAILABLE:
    __all__.extend(['ButtonManager', 'OperationManager'])

if DIALOG_UTILS_AVAILABLE:
    __all__.extend([
        'show_cleanup_confirmation', 'show_reset_confirmation', 'show_validation_errors_dialog',
        'show_operation_progress', 'update_operation_progress', 'complete_operation_progress'
    ])

if BACKEND_COMMUNICATOR_AVAILABLE:
    __all__.extend([
        'BackendProgressCommunicator', 'ServiceIntegrator', 'create_backend_communicator', 'create_service_integrator'
    ])