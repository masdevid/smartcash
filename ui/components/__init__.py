"""
File: smartcash/ui/components/__init__.py
Deskripsi: Package untuk Shared UI components dengan one-liner imports yang efisien
"""
# Core components that don't have external dependencies
# Lazy imports for container components to prevent circular imports
def __getattr__(name):
    """Lazy load components to prevent circular imports."""
    # Container components
    if name == 'ActionContainer':
        from smartcash.ui.components.action_container import ActionContainer
        return ActionContainer
    elif name == 'create_action_container':
        from smartcash.ui.components.action_container import create_action_container
        return create_action_container
    elif name == 'FooterContainer':
        from smartcash.ui.components.footer_container import FooterContainer
        return FooterContainer
    elif name == 'create_footer_container':
        from smartcash.ui.components.footer_container import create_footer_container
        return create_footer_container
    elif name == 'FormContainer':
        # FormContainer is not a class but a function that creates a form container
        from smartcash.ui.components.form_container import create_form_container
        return create_form_container
    elif name == 'create_form_container':
        from smartcash.ui.components.form_container import create_form_container
        return create_form_container
    elif name == 'HeaderContainer':
        from smartcash.ui.components.header_container import HeaderContainer
        return HeaderContainer
    elif name == 'create_header_container':
        from smartcash.ui.components.header_container import create_header_container
        return create_header_container
    elif name == 'MainContainer':
        from smartcash.ui.components.main_container import MainContainer
        return MainContainer
    elif name == 'create_main_container':
        from smartcash.ui.components.main_container import create_main_container
        return create_main_container
    elif name == 'OperationContainer':
        from smartcash.ui.components.operation_container import OperationContainer
        return OperationContainer
    elif name == 'create_operation_container':
        from smartcash.ui.components.operation_container import create_operation_container
        return create_operation_container
    elif name == 'SummaryContainer':
        from smartcash.ui.components.summary_container import SummaryContainer
        return SummaryContainer
    elif name == 'create_summary_container':
        from smartcash.ui.components.summary_container import create_summary_container
        return create_summary_container
        
    # Widget components
    elif name == 'create_dropdown':
        from smartcash.ui.components.widgets.dropdown import create_dropdown
        return create_dropdown
    elif name == 'create_checkbox':
        from smartcash.ui.components.widgets.checkbox import create_checkbox
        return create_checkbox
    elif name == 'create_text_input':
        from smartcash.ui.components.widgets.text_input import create_text_input
        return create_text_input
    elif name == 'create_log_slider':
        from smartcash.ui.components.widgets.log_slider import create_log_slider
        return create_log_slider
    elif name == 'create_slider':
        from smartcash.ui.components.widgets.slider import create_slider
        return create_slider
        
    raise AttributeError(f"module 'smartcash.ui.components' has no attribute '{name}'")

# Import card components
from smartcash.ui.components.card import (
    create_card,
    create_info_card,
    create_success_card,
    create_warning_card,
    create_error_card,
    create_card_row
)

# Import layout components to avoid circular imports
from smartcash.ui.components.layout.layout_components import (
    create_element,
    create_divider,
    create_responsive_container,
    create_responsive_two_column,
    get_responsive_config
)

# Import other components that might depend on layout components
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.action_section import create_action_section
from smartcash.ui.components.status_panel import create_status_panel, update_status_panel
from smartcash.ui.components.summary_panel import create_summary_panel, update_summary_panel


# Import log_accordion components from the new module structure
from smartcash.ui.components.log_accordion.legacy import (
    create_log_accordion, 
    update_log,
    log as log_message
)
from smartcash.ui.components.tabs import create_tab_widget, create_tabs
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

# Info components
from smartcash.ui.components.info import (
    create_info_accordion,
    style_info_content,
    create_tabbed_info
)

# Submodules
# Alerts
from smartcash.ui.components.alerts import (
    create_alert,
    create_alert_html,
    create_status_indicator,
    update_status_panel,
    create_info_box,
    update_status,
    constants as alerts_constants
)

# Dialog
from smartcash.ui.components.dialog import (
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible,
    create_confirmation_area
)

# Header
from smartcash.ui.components.header import (
    create_header,
    create_section_title
)

# Progress Tracker
from smartcash.ui.components.progress_tracker import (
    ProgressLevel,
    ProgressConfig,
    ProgressBarConfig,
    CallbackManager,
    TqdmManager,
    ProgressTracker,
    create_single_progress_tracker,
    create_dual_progress_tracker,
    create_triple_progress_tracker,
    create_flexible_tracker,
)

# Layout
from smartcash.ui.components.layout import (
    create_divider,
    create_responsive_container,
    create_responsive_two_column,
    get_responsive_button_layout,
)

__all__ = [
    # Core components
    'create_card',
    'create_action_buttons',
    'create_action_section',
    'create_info_card',
    'create_success_card',
    'create_warning_card',
    'create_error_card',
    'create_card_row',
    'create_status_panel',
    'update_status_panel',
    'create_log_accordion',
    'update_log',
    'create_tab_widget',
    'create_tabs',
    'create_save_reset_buttons',
    'create_info_accordion',
    'style_info_content',
    'create_tabbed_info',
    
    # Alert components
    'create_alert',
    'create_alert_html',
    'create_status_indicator',
    'update_status',
    'create_info_box',
    'alerts_constants',
    
    # Dialog components
    'show_confirmation_dialog',
    'show_info_dialog',
    'clear_dialog_area',
    'is_dialog_visible',
    'create_confirmation_area',
    
    # Header components
    'create_header',
    'create_section_title',
    'create_error_component',
    
    # Progress Tracker components
    'ProgressLevel',
    'ProgressConfig',
    'ProgressBarConfig',
    'CallbackManager',
    'TqdmManager',
    'ProgressTracker',
    'create_single_progress_tracker',
    'create_dual_progress_tracker',
    'create_triple_progress_tracker',
    'create_flexible_tracker',
    'create_three_progress_tracker',
    
    # Layout components
    'create_divider',
    'create_responsive_container',
    'create_responsive_two_column',
    'get_responsive_button_layout',
    
    # Widget components
    'create_slider',
    'create_dropdown',
    'create_checkbox',
    'create_text_input',
    'create_log_slider',
    
    # Submodules
    'alerts',
    'dialog',
    'header',
    'progress_tracker',
    'info',
    
    # Dialog components
    'create_confirmation_area'
]