"""
File: smartcash/ui/utils/__init__.py
Deskripsi: Utilitas dasar untuk komponen UI dan notebook dengan integrasi logging dan cell yang ditingkatkan
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
    UILogger, 
    UILogHandler, 
    create_dummy_logger, 
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
    format_file_size, 
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
from smartcash.ui.utils.ui_helpers import (
    set_active_theme,
    inject_css_styles,
    create_loading_indicator,
    create_confirmation_dialog,
    create_button_group,
    create_progress_updater,
    update_output_area,
    create_divider,
    create_spacing,
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
    'UILogger',
    'UILogHandler',
    'create_dummy_logger',
    'log_to_ui',
    
    # Visualization Utils
    'create_metric_display',
    'create_class_distribution_plot',
    'create_confusion_matrix_plot',
    'create_metrics_dashboard',
    'create_metrics_history_plot',
    'create_model_comparison_plot',
    
    # File Utils
    'format_file_size',
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
    
    # UI Helpers
    'set_active_theme',
    'inject_css_styles',
    'create_loading_indicator',
    'create_confirmation_dialog',
    'create_button_group',
    'create_progress_updater',
    'update_output_area',
    'create_divider',
    'create_spacing',
]