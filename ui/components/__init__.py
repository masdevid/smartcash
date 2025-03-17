"""
File: smartcash/ui/components/__init__.py
Deskripsi: Import semua komponen dari subdirektori untuk memudahkan akses
"""

# Import dari subdirektori komponen
from smartcash.ui.components.alerts import create_status_indicator, create_info_alert, create_info_box
from smartcash.ui.components.headers import create_header, create_component_header, create_section_title
from smartcash.ui.components.helpers import (
    create_tab_view, create_loading_indicator, update_output_area,
    register_observer_callback, display_file_info, create_progress_updater,
    create_button_group, create_confirmation_dialog
)
from smartcash.ui.components.layouts import (
    STANDARD_LAYOUTS, MAIN_CONTAINER, OUTPUT_WIDGET, BUTTON,
    HIDDEN_BUTTON, TEXT_INPUT, TEXT_AREA, SELECTION,
    HORIZONTAL_GROUP, VERTICAL_GROUP, DIVIDER, CARD,
    TABS, ACCORDION, create_divider
)
from smartcash.ui.components.metrics import (
    create_metric_display, create_result_table, plot_statistics, styled_html
)
from smartcash.ui.components.validators import (
    create_validation_message, show_validation_message, clear_validation_messages,
    validate_required, validate_numeric, validate_integer, validate_min_value,
    validate_max_value, validate_range, validate_min_length, validate_max_length,
    validate_regex, validate_email, validate_url, validate_file_exists,
    validate_directory_exists, validate_file_extension, validate_api_key,
    validate_form, create_validator, combine_validators
)
from smartcash.ui.components.widget_layouts import (
    create_layout, CONTAINER_LAYOUTS, CONTENT_LAYOUTS, INPUT_LAYOUTS,
    BUTTON_LAYOUTS, GROUP_LAYOUTS, COMPONENT_LAYOUTS, 
    create_divider as create_layout_divider, create_spacing, create_grid_layout,
    main_container, card_container, section_container, output_area,
    status_area, button, small_button, hidden_button, text_input,
    slider_input, checkbox, horizontal_group, vertical_group
)

__all__ = [
    # Alerts
    'create_status_indicator', 'create_info_alert', 'create_info_box',
    
    # Headers
    'create_header', 'create_component_header', 'create_section_title',
    
    # Helpers
    'create_tab_view', 'create_loading_indicator', 'update_output_area',
    'register_observer_callback', 'display_file_info', 'create_progress_updater',
    'create_button_group', 'create_confirmation_dialog',
    
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
    
    # Widget Layouts
    'create_layout', 'CONTAINER_LAYOUTS', 'CONTENT_LAYOUTS', 'INPUT_LAYOUTS',
    'BUTTON_LAYOUTS', 'GROUP_LAYOUTS', 'COMPONENT_LAYOUTS', 
    'create_spacing', 'create_grid_layout',
    'main_container', 'card_container', 'section_container', 'output_area',
    'status_area', 'button', 'small_button', 'hidden_button', 'text_input',
    'slider_input', 'checkbox', 'horizontal_group', 'vertical_group'
]