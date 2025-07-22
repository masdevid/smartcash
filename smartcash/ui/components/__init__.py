"""
UI Components Package

This package provides shared UI components that are used across different modules.
Only components that are imported from outside the shared components folder are included here.
"""

# === CORE CONTAINER COMPONENTS ===
# These are the most commonly used container components across the UI modules

def __getattr__(name):
    """Lazy load components to prevent circular imports."""
    
    # Container creation functions - most commonly used
    if name == 'create_main_container':
        from smartcash.ui.components.main_container import create_main_container
        return create_main_container
    elif name == 'create_header_container':
        from smartcash.ui.components.header_container import create_header_container
        return create_header_container
    elif name == 'create_form_container':
        from smartcash.ui.components.form_container import create_form_container
        return create_form_container
    elif name == 'create_action_container':
        from smartcash.ui.components.action_container import create_action_container
        return create_action_container
    elif name == 'create_operation_container':
        from smartcash.ui.components.operation_container import create_operation_container
        return create_operation_container
    elif name == 'create_footer_container':
        from smartcash.ui.components.footer_container import create_footer_container
        return create_footer_container
    elif name == 'create_summary_container':
        from smartcash.ui.components.summary_container import create_summary_container
        return create_summary_container
    
    # Container classes - used in some UI modules
    elif name == 'ActionContainer':
        from smartcash.ui.components.action_container import ActionContainer
        return ActionContainer
    elif name == 'MainContainer':
        from smartcash.ui.components.main_container import MainContainer
        return MainContainer
    elif name == 'HeaderContainer':
        from smartcash.ui.components.header_container import HeaderContainer
        return HeaderContainer
    elif name == 'OperationContainer':
        from smartcash.ui.components.operation_container import OperationContainer
        return OperationContainer
    elif name == 'SummaryContainer':
        from smartcash.ui.components.summary_container import SummaryContainer
        return SummaryContainer
    elif name == 'FooterContainer':
        from smartcash.ui.components.footer_container import FooterContainer
        return FooterContainer
    
    # Dashboard and visualization components
    elif name == 'create_dashboard_cards':
        from smartcash.ui.components.stats_card import create_dashboard_cards
        return create_dashboard_cards
    elif name == 'StatsCard':
        from smartcash.ui.components.stats_card import StatsCard
        return StatsCard
    elif name == 'VisualizationContainer':
        from smartcash.ui.components.visualization_container import VisualizationContainer
        return VisualizationContainer
    
    # Widget creation functions - used across modules
    elif name == 'create_dropdown':
        from smartcash.ui.components.widgets.dropdown import create_dropdown
        return create_dropdown
    elif name == 'create_checkbox':
        from smartcash.ui.components.widgets.checkbox import create_checkbox
        return create_checkbox
    elif name == 'create_text_input':
        from smartcash.ui.components.widgets.text_input import create_text_input
        return create_text_input
    elif name == 'create_slider':
        from smartcash.ui.components.widgets.slider import create_slider
        return create_slider
    elif name == 'create_log_slider':
        from smartcash.ui.components.widgets.log_slider import create_log_slider
        return create_log_slider
        
    raise AttributeError(f"module 'smartcash.ui.components' has no attribute '{name}'")


# === DIRECT IMPORTS ===
# Components that are safe to import directly and commonly used
from smartcash.ui.components.indicator_panel import IndicatorPanel, create_indicator_panel

# Card components - widely used across info boxes and UI modules
from smartcash.ui.components.card import (
    create_card,
    create_info_card,
    create_success_card,
    create_warning_card,
    create_error_card,
    create_card_row
)

# Layout components - fundamental building blocks
from smartcash.ui.components.layout import (
    create_element,
    create_divider,
    create_responsive_container,
    create_responsive_two_column,
    get_responsive_config,
    get_responsive_button_layout
)

# Info components - used extensively in info boxes
from smartcash.ui.components.info import (
    create_info_accordion,
    style_info_content,
    create_tabbed_info
)

# Header components - used in UI modules
from smartcash.ui.components.header import (
    create_header,
    create_section_title
)

# Alert components - used for notifications and status indicators
from smartcash.ui.components.alerts import (
    create_alert,
    create_alert_html,
    create_status_indicator,
    create_info_box,
    constants as alerts_constants
)

# Dialog components - used for user interactions
from smartcash.ui.components.dialog import (
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible,
    create_confirmation_area
)

# Progress tracking - used in operation containers
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
    create_flexible_tracker
)

# Utility components
from smartcash.ui.components.action_buttons import create_action_buttons
from smartcash.ui.components.action_section import create_action_section
from smartcash.ui.components.summary_panel import create_summary_panel, update_summary_panel
from smartcash.ui.components.tabs import create_tab_widget, create_tabs
from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons


# === EXPORTS ===
__all__ = [
    # Container creation functions (most commonly used)
    'create_main_container',
    'create_header_container', 
    'create_form_container',
    'create_action_container',
    'create_operation_container',
    'create_footer_container',
    'create_summary_container',
    
    # Container classes
    'ActionContainer',
    'MainContainer',
    'HeaderContainer', 
    'OperationContainer',
    'SummaryContainer',
    'FooterContainer',
    
    # Dashboard and visualization
    'create_dashboard_cards',
    'StatsCard',
    'VisualizationContainer',
    
    # Widget creation functions
    'create_dropdown',
    'create_checkbox',
    'create_text_input',
    'create_slider',
    'create_log_slider',
    
    # Card components
    'create_card',
    'create_info_card',
    'create_success_card',
    'create_warning_card',
    'create_error_card',
    'create_card_row',
    
    # Layout components
    'create_element',
    'create_divider',
    'create_responsive_container',
    'create_responsive_two_column',
    'get_responsive_config',
    'get_responsive_button_layout',
    
    # Info components
    'create_info_accordion',
    'style_info_content',
    'create_tabbed_info',
    
    # Header components
    'create_header',
    'create_section_title',
    
    # Alert components
    'create_alert',
    'create_alert_html',
    'create_status_indicator',
    'create_info_box',
    'alerts_constants',
    
    # Dialog components
    'show_confirmation_dialog',
    'show_info_dialog',
    'clear_dialog_area',
    'is_dialog_visible',
    'create_confirmation_area',
    
    # Progress tracking components
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
    
    # Utility components
    'create_action_buttons',
    'create_action_section',
    'create_summary_panel',
    'update_summary_panel',
    'create_tab_widget',
    'create_tabs',
    'create_save_reset_buttons'
]