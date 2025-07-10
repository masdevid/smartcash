"""
File: [module]_ui.py
Description: UI components for [module] module following standardized container architecture

This module provides UI components for the [module] interface, built using
shared container components from the SmartCash UI library. It implements the visual
representation and layout of the [module] functionality.

Components:
- Header Container (Header + Status Panel)
- Form Container (Custom to each module)
- Action Container (Save/Reset | Primary | Action Buttons)
- Summary Container (Custom, Nice to have)
- Operation Container (Progress + Dialog + Log)
- Footer Container (Info Accordion + Tips)

The module follows the standardized container-based architecture with consistent
ordering and structure across all UI modules.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container

# Error handling import
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Module-specific imports (customize per module)
# from .module_forms import create_module_form_widget
# from .module_summary import create_module_summary_widget
# from ..constants import UI_CONFIG, BUTTON_CONFIG, VALIDATION_RULES

# === MODULE CONSTANTS ===
# These should be moved to a separate constants.py file in the module
UI_CONFIG = {
    'title': "Module Title",
    'subtitle': "Module description and purpose",
    'icon': "🔧",
    'module_name': "module_name",
    'parent_module': "parent_module",
    'version': "1.0.0"
}

# Button configuration - Choose ONE of these patterns:
# Note: You must choose either a single primary button OR multiple action buttons, but not both

# 1. Single primary action (recommended for single operation modules):
# BUTTON_CONFIG = {
#     'primary': {  # The key must be 'primary' for single action mode
#         'text': 'Start Processing',
#         'style': 'primary',  # Must be 'primary' for main action
#         'tooltip': 'Start the main operation',
#         'order': 1
#     }
# }

# 2. Multiple action buttons (for modules with multiple operations):
BUTTON_CONFIG = {
    'process': {  # Button ID (must be unique)
        'text': 'Process',
        'style': 'info',  # Must use non-primary style (info, success, warning, danger)
        'tooltip': 'Process the data',
        'order': 1,  # Determines left-to-right order
        'icon': 'play'  # Optional: Font Awesome icon name (without 'fa-' prefix)
    },
    'validate': {
        'text': 'Validate',
        'style': 'info',
        'tooltip': 'Validate the data',
        'order': 2,
        'icon': 'check'
    },
    'export': {
        'text': 'Export',
        'style': 'success',
        'tooltip': 'Export the results',
        'order': 3,
        'icon': 'download'
    }
}

# Save/Reset buttons are controlled separately via show_save_reset parameter
# when creating the action container

VALIDATION_RULES = {
    'example_field': {
        'min': 1,
        'max': 100,
        'required': True
    }
}


@handle_ui_errors(error_component_title="[Module] UI Creation Error")
def create_module_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create [module] UI using standardized container architecture.
    
    This function creates the complete UI for the [module] module following the
    standardized container order and structure. It provides consistent layout,
    styling, and functionality across all SmartCash UI modules.
    
    Container Order (standardized):
    1. Header Container (Header + Status Panel)
    2. Form Container (Custom to each module)
    3. Action Container (Save/Reset | Primary | Action Buttons)
    4. Summary Container (Custom, Nice to have)
    5. Operation Container (Progress + Dialog + Log)
    6. Footer Container (Info Accordion + Tips)
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments for customization
        
    Returns:
        Dictionary containing all UI components and widgets with standardized keys:
        - 'ui': Main UI widget
        - 'header_container': Header container widget
        - 'form_container': Form container widget
        - 'action_container': Action container widget
        - 'summary_container': Summary container widget (if enabled)
        - 'operation_container': Operation container widget
        - 'footer_container': Footer container widget
        - Individual widget references for handlers
        - Module metadata and configuration
    """
    config = config or {}
    ui_components = {}
    
    # === 1. HEADER CONTAINER ===
    # Header with title, subtitle, and status indicator
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon=UI_CONFIG['icon'],
        status_text="Ready"
    )
    ui_components['header_container'] = header_container.container
    
    # === 2. FORM CONTAINER ===
    # Custom form layout specific to the module
    form_container = create_form_container(
        title=f"⚙️ {UI_CONFIG['title']} Configuration",
        layout_type="column",  # or "row", "grid" based on needs
        container_padding="15px",
        gap="12px"
    )
    
    # Create module-specific form widgets
    form_widgets = _create_module_form_widgets(config)
    
    # Add form widgets to container
    if form_widgets:
        form_container['add_item'](form_widgets['ui'], width='100%')
    
    ui_components['form_container'] = form_container['container']
    ui_components['form_widgets'] = form_widgets
    
    # === 3. ACTION CONTAINER ===
    # Action buttons with save/reset, primary action, or multiple action buttons
    action_buttons = []
    has_primary = any(btn.get('style') == 'primary' for btn in BUTTON_CONFIG.values())
    
    # Validate button configuration
    if has_primary and len(BUTTON_CONFIG) > 1:
        raise ValueError(
            "Invalid button configuration: Cannot have multiple buttons when using a primary button. "
            "Use either a single primary button or multiple non-primary action buttons."
        )
    
    # Add module-specific action buttons
    for button_key, button_config in BUTTON_CONFIG.items():
        action_buttons.append({
            "id": button_key,
            "text": button_config['text'],
            "style": button_config['style'],
            "tooltip": button_config['tooltip'],
            "order": button_config['order']
        })
    
    # Create action container with appropriate configuration
    action_container = create_action_container(
        buttons=action_buttons,
        title=f"🚀 {UI_CONFIG['title']} Operations",
        alignment="left",  # or "center", "right"
        show_save_reset=config.get('show_save_reset', True)  # Configurable save/reset
    )
    ui_components['action_container'] = action_container['container']
    
    # Extract button references
    if has_primary:
        ui_components['primary_button'] = action_container['primary_button']
    
    # Save/Reset buttons (if enabled)
    if config.get('show_save_reset', True):
        ui_components['save_button'] = action_container.get('save_button')
        ui_components['reset_button'] = action_container.get('reset_button')
    
    # Extract action buttons (only if not using primary button)
    if not has_primary:
        for button_key in BUTTON_CONFIG.keys():
            btn = action_container.get_button(button_key)
            if btn:
                ui_components[f'{button_key}_button'] = btn
    
    # === 4. SUMMARY CONTAINER (Optional) ===
    # Custom summary container for displaying module status/results
    summary_enabled = config.get('show_summary', True)
    if summary_enabled:
        summary_container = create_summary_container(
            title=f"📊 {UI_CONFIG['title']} Summary",
            show_stats=True,
            show_preview=True
        )
        
        # Add module-specific summary content
        summary_content = _create_module_summary_content(config)
        if summary_content:
            summary_container['add_content'](summary_content)
        
        ui_components['summary_container'] = summary_container['container']
    
    # === 5. OPERATION CONTAINER ===
    # Progress tracking, dialog, and logging
    operation_container = create_operation_container(
        title=f"📊 {UI_CONFIG['title']} Status",
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name=UI_CONFIG['module_name']
    )
    ui_components['operation_container'] = operation_container['container']
    
    # Store operation container functions for handlers
    ui_components['log_message'] = operation_container['log_message']
    ui_components['update_progress'] = operation_container['update_progress']
    ui_components['show_dialog'] = operation_container['show_dialog']
    ui_components['show_info_dialog'] = operation_container['show_info_dialog']
    ui_components['clear_dialog'] = operation_container['clear_dialog']
    ui_components['progress_tracker'] = operation_container.get('progress_tracker')
    ui_components['log_output'] = operation_container.get('log_output')
    ui_components['log_accordion'] = operation_container.get('log_accordion')
    
    # === 6. FOOTER CONTAINER ===
    # Info accordion and helpful tips
    footer_container = create_footer_container(
        info_box=_create_module_info_box(),
        tips_box=_create_module_tips_box()
    )
    ui_components['footer_container'] = footer_container.container
    
    # === MAIN UI ASSEMBLY ===
    # Assemble all containers in standardized order
    main_components = [
        ui_components['header_container'],
        ui_components['form_container'],
        ui_components['action_container']
    ]
    
    # Add summary container if enabled
    if summary_enabled and 'summary_container' in ui_components:
        main_components.append(ui_components['summary_container'])
    
    # Add remaining containers
    main_components.extend([
        ui_components['operation_container'],
        ui_components['footer_container']
    ])
    
    # Create main UI container
    main_ui = widgets.VBox(
        main_components,
        layout=widgets.Layout(
            width='100%',
            margin='10px 0',
            border='1px solid #e0e0e0',
            border_radius='8px',
            background_color='#fafafa',
            padding='15px'
        )
    )
    
    ui_components['ui'] = main_ui
    ui_components['main_container'] = main_ui
    
    # === METADATA AND CONFIGURATION ===
    # Add module metadata and configuration
    ui_components.update({
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'logger_namespace': f"smartcash.ui.{UI_CONFIG['parent_module']}.{UI_CONFIG['module_name']}",
        'ui_initialized': True,
        'config': config,
        'version': UI_CONFIG['version']
    })
    
    return ui_components


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create module-specific form widgets.
    
    This function should be customized for each module to create the appropriate
    form elements based on the module's requirements.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        Dictionary containing form widgets and UI
    """
    # Example form widgets - customize per module
    example_input = widgets.Text(
        value=config.get('example_field', ''),
        description='Example Field:',
        placeholder='Enter example value',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    example_dropdown = widgets.Dropdown(
        options=[('Option 1', 'opt1'), ('Option 2', 'opt2')],
        value=config.get('example_option', 'opt1'),
        description='Example Option:',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    example_checkbox = widgets.Checkbox(
        value=config.get('example_flag', False),
        description='Example Flag',
        style={'description_width': 'initial'},
        layout={'width': '100%'}
    )
    
    # Create form layout
    form_ui = widgets.VBox([
        widgets.HTML(f"<h4>⚙️ {UI_CONFIG['title']} Configuration</h4>"),
        example_input,
        example_dropdown,
        example_checkbox,
        widgets.HTML(
            "<div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;'>"
            "<strong>💡 Configuration Tips:</strong><br>"
            "• Configure settings based on your specific requirements<br>"
            "• Use validation to ensure proper values<br>"
            "• Save configurations for future use"
            "</div>"
        )
    ])
    
    return {
        'ui': form_ui,
        'example_input': example_input,
        'example_dropdown': example_dropdown,
        'example_checkbox': example_checkbox
    }


def _create_module_summary_content(config: Dict[str, Any]) -> Optional[widgets.Widget]:
    """
    Create module-specific summary content.
    
    This function should be customized for each module to display relevant
    summary information, statistics, or previews.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        Widget containing summary content or None if not applicable
    """
    summary_html = f"""
    <div style="padding: 15px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 8px; color: white; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <h4 style="margin: 0 0 10px 0; font-size: 1.1rem;">📊 {UI_CONFIG['title']} Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.8rem; opacity: 0.8;">Status</div>
                <div style="font-size: 1rem; font-weight: 600;">Ready</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.8rem; opacity: 0.8;">Configuration</div>
                <div style="font-size: 1rem; font-weight: 600;">Default</div>
            </div>
        </div>
    </div>
    """
    
    return widgets.HTML(summary_html)


def _create_module_info_box() -> widgets.Widget:
    """
    Create module-specific info box for footer.
    
    Returns:
        Widget containing module information
    """
    info_html = f"""
    <div class="alert alert-info" style="font-size: 0.9em; padding: 12px; margin: 0;">
        <strong>💡 {UI_CONFIG['title']} Information:</strong>
        <ul style="margin: 8px 0 0 15px; padding: 0;">
            <li><strong>Purpose:</strong> Brief description of module functionality</li>
            <li><strong>Configuration:</strong> How to configure module settings</li>
            <li><strong>Operations:</strong> Available operations and their purposes</li>
            <li><strong>Tips:</strong> Best practices and recommendations</li>
        </ul>
    </div>
    """
    
    return widgets.HTML(info_html)


def _create_module_tips_box() -> widgets.Widget:
    """
    Create module-specific tips box for footer.
    
    Returns:
        Widget containing helpful tips
    """
    tips_html = f"""
    <div class="alert alert-success" style="font-size: 0.9em; padding: 12px; margin: 0;">
        <strong>💡 {UI_CONFIG['title']} Tips:</strong>
        <ul style="margin: 8px 0 0 15px; padding: 0;">
            <li>Start with default settings and adjust as needed</li>
            <li>Use validation to ensure proper configuration</li>
            <li>Save configurations for future use</li>
            <li>Monitor progress and logs during operations</li>
        </ul>
    </div>
    """
    
    return widgets.HTML(tips_html)


# === HELPER FUNCTIONS ===

def validate_module_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate module configuration against defined rules.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration validation fails
    """
    validated_config = config.copy()
    
    # Example validation logic - customize per module
    for field, rules in VALIDATION_RULES.items():
        if field in validated_config:
            value = validated_config[field]
            
            # Check required fields
            if rules.get('required', False) and not value:
                raise ValueError(f"Required field '{field}' is missing or empty")
            
            # Check value ranges
            if 'min' in rules and value < rules['min']:
                raise ValueError(f"Field '{field}' value {value} is below minimum {rules['min']}")
            
            if 'max' in rules and value > rules['max']:
                raise ValueError(f"Field '{field}' value {value} is above maximum {rules['max']}")
    
    return validated_config


def get_module_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the module.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'example_field': 'default_value',
        'example_option': 'opt1',
        'example_flag': False,
        'show_summary': True
    }


def update_module_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration from UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Updated configuration dictionary
    """
    config = {}
    
    # Extract values from form widgets
    form_widgets = ui_components.get('form_widgets', {})
    
    if 'example_input' in form_widgets:
        config['example_field'] = form_widgets['example_input'].value
    
    if 'example_dropdown' in form_widgets:
        config['example_option'] = form_widgets['example_dropdown'].value
    
    if 'example_checkbox' in form_widgets:
        config['example_flag'] = form_widgets['example_checkbox'].value
    
    return config


# === EXPORT FUNCTIONS ===
def create_module_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Alias for create_[module]_ui for backward compatibility.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components
    """
    return create_module_ui(config, **kwargs)

# === USAGE EXAMPLE ===

if __name__ == "__main__":
    # Example usage
    ui = create_module_ui()
    display(ui['ui'])
    print("Module UI Template")
    print("================")
    print()
    print("To use this template:")
    print("1. Copy this file to your module's components/ directory")
    print("2. Rename it to [module]_ui.py")
    print("3. Replace [module] with your actual module name")
    print("4. Customize the UI_CONFIG, BUTTON_CONFIG, and VALIDATION_RULES")
    print("5. Implement the _create_module_form_widgets() function")
    print("6. Customize the summary, info, and tips content")
    print("7. Update imports to match your module structure")
    print()
    print("Required customizations:")
    print("- Module name and configuration")
    print("- Form widgets and validation")
    print("- Button configurations")
    print("- Summary content (if applicable)")
    print("- Info and tips content")
    print()
    print("The template provides:")
    print("- Standardized container order")
    print("- Consistent styling and layout")
    print("- Error handling and validation")
    print("- Progress tracking and logging")
    print("- Modular and maintainable code structure")