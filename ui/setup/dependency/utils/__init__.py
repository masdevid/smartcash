"""
File: smartcash/ui/setup/dependency/utils/__init__.py
Deskripsi: Dependency utilities exports
"""

from .package_utils import (
    get_installed_packages_dict,
    check_package_installation_status,
    filter_uninstalled_packages,
    install_single_package,
    batch_check_packages_status,
    parse_package_requirement,
    extract_package_name_from_requirement
)

from .ui_state_utils import (
    create_operation_context,
    update_status_panel,
    log_to_ui_safe,
    ProgressSteps
)

from .system_info_utils import (
    get_comprehensive_system_info,
    check_system_requirements,
    format_system_info_html
)

from .report_generator_utils import (
    generate_comprehensive_status_report,
    generate_installation_summary_report,
    generate_analysis_summary_report
)

from .constants import (
    STATUS_CONFIGS,
    PACKAGE_STATUS_MAPPING,
    INSTALLATION_DEFAULTS,
    ANALYSIS_DEFAULTS,
    get_status_config,
    get_ui_message
)

__all__ = [
    # Package utilities
    'get_installed_packages_dict',
    'check_package_installation_status',
    'filter_uninstalled_packages',
    'install_single_package',
    'batch_check_packages_status',
    'parse_package_requirement',
    'extract_package_name_from_requirement',
    
    # UI state management
    'create_operation_context',
    'update_status_panel',
    'log_to_ui_safe',
    'ProgressSteps',
    
    # System info
    'get_comprehensive_system_info',
    'check_system_requirements',
    'format_system_info_html',
    
    # Report generation
    'generate_comprehensive_status_report',
    'generate_installation_summary_report',
    'generate_analysis_summary_report',
    
    # Constants
    'STATUS_CONFIGS',
    'PACKAGE_STATUS_MAPPING',
    'INSTALLATION_DEFAULTS',
    'ANALYSIS_DEFAULTS',
    'get_status_config',
    'get_ui_message'
]