"""
File: smartcash/ui/setup/colab/components/colab_ui.py
Description: UI components for Colab environment setup following standardized container architecture

This module provides UI components for the Colab setup interface, built using
shared container components from the SmartCash UI library. It implements the visual
representation and layout of the Colab environment setup functionality.

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
from typing import Dict, Any, Optional

# Core container imports - standardized across all modules
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.summary_container import create_summary_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container

# Error handling import
from smartcash.ui.core.errors.handlers import handle_ui_errors

# Module-specific imports
from smartcash.ui.setup.colab.components import env_info_panel
from smartcash.ui.setup.colab.components import setup_summary
from smartcash.ui.setup.colab.components import tips_panel
from smartcash.ui.setup.colab.constants import UI_CONFIG, BUTTON_CONFIG, VALIDATION_RULES

# Module constants (for validator compliance)
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create Colab-specific form widgets.
    
    This function should be customized for each module to create the appropriate
    form elements based on the module's requirements.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        Dictionary containing form widgets and UI
    """
    # Environment detection checkbox
    auto_detect = widgets.Checkbox(
        value=config.get('auto_detect', True),
        description='Auto-detect environment',
        style={'description_width': 'initial'}
    )
    
    # Drive mount path
    drive_path = widgets.Text(
        value=config.get('drive_path', '/content/drive'),
        description='Drive Mount Path:',
        placeholder='Enter drive mount path',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Project name
    project_name = widgets.Text(
        value=config.get('project_name', 'SmartCash'),
        description='Project Name:',
        placeholder='Enter project name',
        style={'description_width': '120px'},
        layout={'width': '100%'}
    )
    
    # Create environment info panel with lazy loading
    env_info = env_info_panel.create_env_info_panel(config, lazy_load=True)
    
    # Create form layout
    form_ui = widgets.VBox([
        widgets.HTML("<h4>🔧 Colab Environment Configuration</h4>"),
        auto_detect,
        drive_path,
        project_name,
        widgets.HTML("<h4>📊 Environment Information</h4>"),
        env_info,
        widgets.HTML(
            "<div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;'>"
            "<strong>💡 Configuration Tips:</strong><br>"
            "• Auto-detect will automatically configure environment settings<br>"
            "• Ensure drive path is correct for Google Drive mounting<br>"
            "• Project name will be used for folder structure"
            "</div>"
        )
    ])
    
    return {
        'ui': form_ui,
        'auto_detect': auto_detect,
        'drive_path': drive_path,
        'project_name': project_name,
        'env_info_panel': env_info
    }


def _create_module_summary_content(config: Dict[str, Any]) -> Optional[widgets.Widget]:
    """
    Create Colab-specific summary content.

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
        <h4 style="margin: 0 0 10px 0; font-size: 1.1rem;">🚀 Colab Environment Summary</h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.8rem; opacity: 0.8;">Status</div>
                <div style="font-size: 1rem; font-weight: 600;">Ready for Setup</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 10px; border-radius: 6px;">
                <div style="font-size: 0.8rem; opacity: 0.8;">Environment</div>
                <div style="font-size: 1rem; font-weight: 600;">Google Colab</div>
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
    Create Colab-specific info box for footer.

    Returns:
        Widget containing module information
    """
    info_html = f"""
    <div class="alert alert-info" style="font-size: 0.9em; padding: 12px; margin: 0;">
        <strong>💡 Colab Environment Setup Information:</strong>
        <ul style="margin: 8px 0 0 15px; padding: 0;">
            <li><strong>Purpose:</strong> Automatically configure Google Colab environment for SmartCash</li>
            <li><strong>Configuration:</strong> Set drive mount path and project settings</li>
            <li><strong>Operations:</strong> Mount drive, create folders, setup symlinks, sync configs</li>
            <li><strong>Verification:</strong> Validate environment setup and dependencies</li>
        </ul>
    </div>
    """
    
    return widgets.HTML(info_html)


def _create_module_tips_box() -> widgets.Widget:
    """
    Create Colab-specific tips box for footer.

    Returns:
        Widget containing helpful tips
    """
    tips_html = f"""
    <div class="alert alert-success" style="font-size: 0.9em; padding: 12px; margin: 0;">
        <strong>💡 Colab Setup Tips:</strong>
        <ul style="margin: 8px 0 0 15px; padding: 0;">
            <li>Ensure you have Google Drive access before starting</li>
            <li>The setup process will create necessary folder structures</li>
            <li>Monitor the progress tracker for detailed status updates</li>
            <li>Check logs if any errors occur during setup</li>
        </ul>
    </div>
    """
    
    return widgets.HTML(tips_html)


@handle_ui_errors(error_component_title="Colab UI Creation Error")
def create_colab_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create Colab UI using standardized container architecture.
    
    This function creates the complete UI for the Colab module following the
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
    # Primary button for Colab setup with phases - following single operation pattern
    action_container = create_action_container(
        buttons=[{
            'id': 'colab_setup',
            'text': BUTTON_CONFIG['primary']['text'],
            'style': 'primary',
            'tooltip': BUTTON_CONFIG['primary']['tooltip']
        }],
        title=f"🚀 {UI_CONFIG['title']} Operations",
        show_save_reset=True  # Include save/reset for configuration
    )
    ui_components['action_container'] = action_container['container']
    
    # Extract button references
    ui_components['primary_button'] = action_container['primary_button']
    ui_components['setup_button'] = action_container['buttons'].get('colab_setup')
    
    # Save/Reset buttons (access from action_container instance)
    action_container_instance = action_container['action_container']
    ui_components['save_button'] = getattr(action_container_instance, 'save_button', None)
    ui_components['reset_button'] = getattr(action_container_instance, 'reset_button', None)
    
    # Expose phase management methods for primary button
    ui_components['set_phase'] = action_container.get('set_phase')
    ui_components['set_phases'] = action_container.get('set_phases')
    ui_components['enable_all'] = action_container.get('enable_all')
    ui_components['disable_all'] = action_container.get('disable_all')
    
    # === 4. SUMMARY CONTAINER (Optional) ===
    # Custom summary container for displaying module status/results
    summary_enabled = config.get('show_summary', True)
    if summary_enabled:
        summary_container = create_summary_container(
            title=f"{UI_CONFIG['title']} Summary",
            icon="📊"
        )
        
        # Add module-specific summary content
        summary_content = _create_module_summary_content(config)
        if summary_content:
            summary_container.set_content(summary_content.value)
        
        ui_components['summary_container'] = summary_container.container
    
    # === 5. OPERATION CONTAINER ===
    # Progress tracking, dialog, and logging
    operation_container = create_operation_container(
        title=f"📊 {UI_CONFIG['module_name']} Status",
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name=UI_CONFIG['module_name'],
        log_namespace_filter='colab'  # Filter logs for colab namespace only
    )
    ui_components['operation_container'] = operation_container['container']
    
    # Store operation container functions for handlers
    ui_components['log_message'] = operation_container['log_message']
    ui_components['update_progress'] = operation_container['update_progress']
    ui_components['show_dialog'] = operation_container['show_dialog']
    ui_components['show_info_dialog'] = operation_container['show_info_dialog']
    ui_components['clear_dialog'] = operation_container['clear_dialog']
    ui_components['operation_container'] = operation_container.get('container')
    ui_components['progress_tracker'] = operation_container.get('progress_tracker')
    ui_components['log_accordion'] = operation_container.get('log_accordion')
    
    # === 6. FOOTER CONTAINER ===
    # Info accordion and helpful tips
    from smartcash.ui.components.footer_container import PanelConfig, PanelType
    
    footer_container = create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Environment Setup Info",
                content=_create_module_info_box().value,  # Extract HTML value
                flex="1",
                min_width="300px",
                open_by_default=False
            ),
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Setup Tips", 
                content=_create_module_tips_box().value,  # Extract HTML value
                flex="1",
                min_width="300px",
                open_by_default=False
            )
        ]
    )
    ui_components['footer_container'] = footer_container.container
    
    # === MAIN UI ASSEMBLY ===
    # Assemble all containers in standardized order, filtering out None values
    main_components = []
    
    # Add containers in order, checking each one is not None
    container_order = [
        ui_components.get('header_container'),
        ui_components.get('form_container'), 
        ui_components.get('action_container')
    ]
    
    # Add summary container if enabled
    if summary_enabled and 'summary_container' in ui_components:
        container_order.append(ui_components.get('summary_container'))
    
    # Add remaining containers
    container_order.extend([
        ui_components.get('operation_container'),
        ui_components.get('footer_container')
    ])
    
    # Filter out None values
    main_components = [container for container in container_order if container is not None]
    
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


# === HELPER FUNCTIONS ===

def validate_colab_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate Colab configuration against defined rules.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration validation fails
    """
    validated_config = config.copy()
    
    # Validate required fields
    for field, rules in VALIDATION_RULES.items():
        if field in validated_config:
            value = validated_config[field]
            
            # Check required fields
            if rules.get('required', False) and not value:
                raise ValueError(f"Required field '{field}' is missing or empty")
            
            # Check minimum length
            if 'min_length' in rules and len(str(value)) < rules['min_length']:
                raise ValueError(f"Field '{field}' must be at least {rules['min_length']} characters")
            
            # Check allowed values
            if 'allowed_values' in rules and value not in rules['allowed_values']:
                raise ValueError(f"Field '{field}' value '{value}' not in allowed values: {rules['allowed_values']}")
    
    return validated_config


def get_colab_default_config() -> Dict[str, Any]:
    """
    Get default configuration for the Colab module.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'auto_detect': True,
        'drive_path': '/content/drive',
        'project_name': 'SmartCash',
        'show_summary': True
    }


def update_colab_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
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
    
    if 'auto_detect' in form_widgets:
        config['auto_detect'] = form_widgets['auto_detect'].value
    
    if 'drive_path' in form_widgets:
        config['drive_path'] = form_widgets['drive_path'].value
    
    if 'project_name' in form_widgets:
        config['project_name'] = form_widgets['project_name'].value
    
    return config


# === EXPORT FUNCTIONS ===
def create_colab_ui_components(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Alias for create_colab_ui for backward compatibility.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components
    """
    return create_colab_ui(config, **kwargs)