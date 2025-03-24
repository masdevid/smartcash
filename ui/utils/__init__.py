"""
File: smartcash/ui/utils/__init__.py
Deskripsi: Import utilitas dasar untuk komponen UI dan notebook dengan integrasi logging dan cell yang ditingkatkan
"""

from smartcash.ui.utils.cell_utils import (
    setup_notebook_environment, 
    setup_ui_component, 
    create_default_ui_components,
    cleanup_resources,
    register_cleanup_resource,
    display_ui
)
from smartcash.ui.utils.logging_utils import (
    setup_ipython_logging, 
    log_to_ui
)
from smartcash.ui.utils.visualization_utils import (
    create_metric_display, 
    create_class_distribution_plot, 
    create_confusion_matrix_plot, 
    create_metrics_dashboard, 
    create_metrics_history_plot, 
    create_model_comparison_plot
)
from smartcash.ui.utils.file_utils import (
    display_file_info, 
    directory_tree, 
    create_file_upload_widget,
    save_uploaded_file, 
    create_file_browser, 
    backup_file,
    list_files, 
    is_image_file, 
    is_video_file, 
    get_file_info,
    create_file_download_link
)

from smartcash.ui.utils.drive_utils import (
    detect_drive_mount,
    sync_drive_to_local,
    async_sync_drive
)
from smartcash.ui.utils.fallback_utils import (
    import_with_fallback,
    get_logger_safely,
    get_status_widget,
    create_status_message,
    show_status,
    update_status_panel,
    load_config_safely,
    get_dataset_manager,
    handle_download_status
)

from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert, create_info_box
from smartcash.ui.utils.header_utils import create_header, create_section_title

from smartcash.ui.utils.layout_utils import (
    STANDARD_LAYOUTS, MAIN_CONTAINER, OUTPUT_WIDGET, BUTTON,
    HIDDEN_BUTTON, TEXT_INPUT, TEXT_AREA, SELECTION,
    HORIZONTAL_GROUP, VERTICAL_GROUP, DIVIDER, CARD,
    TABS, ACCORDION, create_divider
)
from smartcash.ui.utils.metric_utils import (
    create_metric_display, create_result_table, plot_statistics, styled_html
)
from smartcash.ui.utils.validator_utils import (
    create_validation_message, show_validation_message, clear_validation_messages,
    validate_required, validate_numeric, validate_integer, validate_min_value,
    validate_max_value, validate_range, validate_min_length, validate_max_length,
    validate_regex, validate_email, validate_url, validate_file_exists,
    validate_directory_exists, validate_file_extension, validate_api_key,
    validate_form, create_validator, combine_validators
)
__all__ = [
    # Cell Utils
    'setup_notebook_environment',
    'setup_ui_component',
    'create_default_ui_components',
    'cleanup_resources',
    'register_cleanup_resource',
    'display_ui',
    
    # Logging Utils
    'setup_ipython_logging',
    'create_dummy_logger',
    'log_to_ui',
    'alert_to_ui'
    
    # Visualization Utils
    'create_metric_display',
    'create_class_distribution_plot',
    'create_confusion_matrix_plot',
    'create_metrics_dashboard',
    'create_metrics_history_plot',
    'create_model_comparison_plot',
    
    # File Utils
    'display_file_info',
    'directory_tree',
    'create_file_upload_widget',
    'save_uploaded_file',
    'create_file_browser',
    'backup_file',
    'list_files',
    'is_image_file',
    'is_video_file',
    'get_file_info',
    'create_file_download_link',
    
    # Drive Detector
    'detect_drive_mount',
    'sync_drive_to_local',
    'async_sync_drive',
    
    # Fallback Utilities
    'import_with_fallback',
    'get_logger_safely',
    'get_status_widget',
    'create_status_message',
    'show_status',
    'update_status_panel',
    'load_config_safely',
    'get_dataset_manager',
    'handle_download_status'

    # Alerts
    'create_status_indicator', 'create_info_alert', 'create_info_box',
    
    # Headers
    'create_header', 'create_section_title',
    
    # Layouts
    'STANDARD_LAYOUTS', 'MAIN_CONTAINER', 'OUTPUT_WIDGET', 'BUTTON',
    'HIDDEN_BUTTON', 'TEXT_INPUT', 'TEXT_AREA', 'SELECTION',
    'HORIZONTAL_GROUP', 'VERTICAL_GROUP', 'DIVIDER', 'CARD',
    'TABS', 'ACCORDION', 'create_divider',
    
    # Metrics
    'create_metric_display', 'create_result_table', 'plot_statistics', 'styled_html',
    
    # Validators
    'create_validation_message', 'show_validation_message', 'clear_validation_messages',
    'validate_required', 'validate_numeric', 'validate_integer', 'validate_min_value',
    'validate_max_value', 'validate_range', 'validate_min_length', 'validate_max_length',
    'validate_regex', 'validate_email', 'validate_url', 'validate_file_exists',
    'validate_directory_exists', 'validate_file_extension', 'validate_api_key',
    'validate_form', 'create_validator', 'combine_validators',
]