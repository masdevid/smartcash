# File: smartcash/ui/setup/dependency/utils/__init__.py  
# Deskripsi: Dependency utilities exports tanpa circular import

# UI dependencies (lightweight imports)
from smartcash.ui.setup.dependency.utils.ui_deps import LogLevel, with_logging, requires, get_optional

# Button state utilities
from smartcash.ui.setup.dependency.utils.button_state import create_button_state_handler

# Package utilities (core functionality)
from smartcash.ui.setup.dependency.utils.package_utils import (
    get_installed_packages_dict,
    check_package_installation_status,
    filter_uninstalled_packages,
    install_single_package,
    batch_check_packages_status,
    parse_package_requirement,
    extract_package_name_from_requirement,
    get_package_detailed_info
)

# UI state management (tanpa import circular)
from smartcash.ui.setup.dependency.utils.ui_state_utils import (
    create_operation_context,
    update_status_panel,
    log_to_ui_safe,
    ProgressSteps,
    update_progress_step,
    show_progress_tracker_safe,
    reset_progress_tracker_safe,
    complete_operation_with_message,
    error_operation_with_message,
    with_button_context,
    clear_ui_outputs,
    reset_ui_logger,
    create_progress_tracker,
    create_logger_bridge
)

# Package status utilities (terpisah dari UI state)
from smartcash.ui.setup.dependency.utils.package_status_utils import (
    update_package_status,
    update_package_status_by_name,
    batch_update_package_status,
    get_package_status_from_ui,
    get_all_package_statuses,
    reset_all_package_statuses,
    filter_packages_by_status,
    count_packages_by_status,
    sync_package_status_with_system
)

# System info utilities
from smartcash.ui.setup.dependency.utils.system_info_utils import (
    get_comprehensive_system_info,
    check_system_requirements,
    format_system_info_html
)

# Report generation utilities
from smartcash.ui.setup.dependency.utils.report_generator_utils import (
    generate_comprehensive_status_report,
    generate_installation_summary_report,
    generate_analysis_summary_report,
    generate_system_compatibility_report
)

# Package selector utilities
from smartcash.ui.setup.dependency.utils.package_selector_utils import (
    create_package_selector_grid,
    get_package_categories,
    get_selected_packages,
    reset_package_selections,
    _create_category_widget_improved
)

# Constants
from smartcash.ui.setup.dependency.utils.constants import (
    STATUS_CONFIGS,
    PACKAGE_STATUS_MAPPING,
    INSTALLATION_DEFAULTS,
    ANALYSIS_DEFAULTS,
    get_status_config,
    get_ui_message
)

__all__ = [
    # UI dependencies
    'LogLevel',
    'with_logging', 
    'requires',
    'get_optional',
    
    # Button state
    'create_button_state_handler',
    
    # Package utilities
    'get_installed_packages_dict',
    'check_package_installation_status',
    'filter_uninstalled_packages',
    'install_single_package',
    'batch_check_packages_status',
    'parse_package_requirement',
    'extract_package_name_from_requirement',
    'get_package_detailed_info',
    
    # UI state management
    'create_operation_context',
    'update_status_panel',
    'log_to_ui_safe',
    'ProgressSteps',
    'update_progress_step',
    'show_progress_tracker_safe',
    'reset_progress_tracker_safe',
    'complete_operation_with_message',
    'error_operation_with_message',
    'with_button_context',
    'clear_ui_outputs',
    'reset_ui_logger',
    'create_progress_tracker',
    'create_logger_bridge',
    
    # Package status utilities
    'update_package_status',
    'update_package_status_by_name',
    'batch_update_package_status',
    'get_package_status_from_ui',
    'get_all_package_statuses',
    'reset_all_package_statuses',
    'filter_packages_by_status',
    'count_packages_by_status',
    'sync_package_status_with_system',
    
    # System info
    'get_comprehensive_system_info',
    'check_system_requirements',
    'format_system_info_html',
    
    # Report generation
    'generate_comprehensive_status_report',
    'generate_installation_summary_report',
    'generate_analysis_summary_report',
    'generate_system_compatibility_report',
    
    # Package selector utilities
    'create_package_selector_grid',
    'get_package_categories',
    'get_selected_packages',
    'reset_package_selections',
    '_create_category_widget_improved',
    
    # Constants
    'STATUS_CONFIGS',
    'PACKAGE_STATUS_MAPPING',
    'INSTALLATION_DEFAULTS',
    'ANALYSIS_DEFAULTS',
    'get_status_config',
    'get_ui_message'
]