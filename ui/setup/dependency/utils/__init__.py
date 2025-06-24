"""
File: smartcash/ui/setup/dependency/utils/__init__.py
Deskripsi: Dependency utilities exports dengan resolved circular imports
"""

# Core utilities - no dependencies
from smartcash.ui.setup.dependency.utils.constants import (
    STATUS_CONFIGS,
    PACKAGE_STATUS_MAPPING,
    INSTALLATION_DEFAULTS,
    ANALYSIS_DEFAULTS,
    get_status_config,
    get_ui_message
)

from smartcash.ui.setup.dependency.utils.ui_deps import LogLevel, with_logging, requires, get_optional

# Package utilities
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

# UI state utilities
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
    update_package_status_by_name,
    batch_update_package_status,
    with_button_context
)

# System info utilities
from smartcash.ui.setup.dependency.utils.system_info_utils import (
    get_comprehensive_system_info,
    check_system_requirements,
    format_system_info_html
)

# Package selector utilities - import after others to avoid circular dependency
from smartcash.ui.setup.dependency.utils.package_selector_utils import (
    create_package_selector_grid,
    get_package_categories,
    update_package_status,
    get_selected_packages,
    reset_package_selections,
    _create_category_widget_improved
)

# Button state utilities
from smartcash.ui.setup.dependency.utils.button_state import create_button_state_handler

# Report generator - import last to avoid circular dependency
# Use lazy import to break circular dependency
def get_report_generators():
    """Lazy import untuk report generators"""
    from smartcash.ui.setup.dependency.utils.report_generator_utils import (
        generate_comprehensive_status_report,
        generate_installation_summary_report,
        generate_analysis_summary_report,
        generate_system_compatibility_report
    )
    return {
        'generate_comprehensive_status_report': generate_comprehensive_status_report,
        'generate_installation_summary_report': generate_installation_summary_report,
        'generate_analysis_summary_report': generate_analysis_summary_report,
        'generate_system_compatibility_report': generate_system_compatibility_report
    }

__all__ = [
    # Constants
    'STATUS_CONFIGS',
    'PACKAGE_STATUS_MAPPING', 
    'INSTALLATION_DEFAULTS',
    'ANALYSIS_DEFAULTS',
    'get_status_config',
    'get_ui_message',
    
    # UI Dependencies
    'LogLevel',
    'with_logging',
    'requires',
    'get_optional',
    
    # Package utilities
    'get_installed_packages_dict',
    'check_package_installation_status',
    'filter_uninstalled_packages',
    'install_single_package',
    'batch_check_packages_status',
    'parse_package_requirement',
    'extract_package_name_from_requirement',
    'get_package_detailed_info',
    
    # UI state utilities
    'create_operation_context',
    'update_status_panel',
    'log_to_ui_safe',
    'ProgressSteps',
    'update_progress_step',
    'show_progress_tracker_safe',
    'reset_progress_tracker_safe',
    'complete_operation_with_message',
    'error_operation_with_message',
    'update_package_status_by_name',
    'batch_update_package_status',
    'with_button_context',
    
    # System info utilities
    'get_comprehensive_system_info',
    'check_system_requirements',
    'format_system_info_html',
    
    # Package selector utilities
    'create_package_selector_grid',
    'get_package_categories',
    'update_package_status',
    'get_selected_packages',
    'reset_package_selections',
    '_create_category_widget_improved',
    
    # Button state
    'create_button_state_handler',
    
    # Report generators (lazy loaded)
    'get_report_generators'
]