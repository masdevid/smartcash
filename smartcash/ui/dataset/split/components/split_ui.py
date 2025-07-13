"""
Dataset Split UI Components Module.

This module provides UI components for the dataset split configuration interface,
built using shared container components from the SmartCash UI library.
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets

# Standard UI components
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container, LayoutType
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.core.decorators import handle_ui_errors
from smartcash.ui.core.errors.enums import ErrorLevel

# Local components
from .ratio_section import create_ratio_section
from .path_section import create_path_section
from .advanced_section import create_advanced_section

# Import module constants
from smartcash.ui.dataset.split.constants import (
    UI_CONFIG,
    BUTTON_CONFIG,
    DEFAULT_SPLIT_RATIOS,
    VALIDATION_RULES
)

# Re-export constants to maintain backward compatibility
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG

def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create form widgets for the split module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing form widgets
    """
    # Create form sections
    ratio_components = create_ratio_section(config)
    path_components = create_path_section(config)
    advanced_components = create_advanced_section(config)
    
    # Combine all components
    components = {**ratio_components, **path_components, **advanced_components}
    
    # Create form rows
    form_rows = [
        ratio_components["ratio_section"],
        path_components["path_section"],
        advanced_components["advanced_section"]
    ]
    
    return {
        'components': components,
        'form_rows': form_rows
    }


def _create_module_summary_content(components: Dict[str, Any]) -> widgets.Widget:
    """Create summary content for the module.
    
    Args:
        components: Dictionary of UI components
        
    Returns:
        Widget containing the summary content
    """
    summary = widgets.HTML(
        value="<h4>Dataset Split Summary</h4><p>Configured dataset split parameters.</p>"
    )
    return summary

def _create_module_info_box() -> widgets.Widget:
    """Create an info box with module documentation.
    
    Returns:
        Info box widget
    """
    info_text = """
    <h4>Dataset Split Help</h4>
    <p>Use this module to split your dataset into training, validation, and test sets.</p>
    <ul>
        <li><b>Ratios:</b> Define the split ratios between sets</li>
        <li><b>Paths:</b> Specify input and output directories</li>
        <li><b>Advanced:</b> Configure additional split options</li>
    </ul>
    """
    return widgets.HTML(info_text)

@handle_ui_errors(
    error_component_title=f"{UI_CONFIG['module_name']} Error",
    log_error=True,
    return_type=dict,
    level=ErrorLevel.ERROR,
    fail_fast=False,
    create_ui=True
)
def create_split_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create the dataset split configuration UI.
    
    This function creates a complete UI for configuring dataset splits with the following sections:
    - Ratio configuration
    - Path settings
    - Advanced options
    
    Args:
        config: Optional configuration dictionary with initial values
        **kwargs: Additional keyword arguments passed to container components
        
    Returns:
        Dictionary containing all UI components and containers
    """
    # Initialize config if not provided
    config = config or {}
    
    # Create UI components dictionary with error handling
    ui_components = {}
    
    try:
        # Create form widgets
        form_widgets = _create_module_form_widgets(config)
        components = form_widgets['components']
        form_rows = form_widgets['form_rows']
        
        # Create log accordion
        log_output = widgets.Output()
        log_accordion = widgets.Accordion(
            children=[log_output],
            selected_index=None
        )
        log_accordion.set_title(0, 'Log Messages')
        
        # Create header
        header_container = create_header_container(
            title=UI_CONFIG['title'],
            description=UI_CONFIG['description'],
            **kwargs
        )
        
        # Create form container
        form_container = create_form_container(
            form_rows=form_rows,
            layout_type=LayoutType.COLUMN,
            **kwargs
        )
        
        # Create action container with explicit save/reset buttons
        action_buttons = [
            {
                'name': 'save_button',
                'label': 'Simpan Konfigurasi',
                'button_style': 'success',
                'tooltip': 'Simpan konfigurasi split dataset',
                'icon': 'save'
            },
            {
                'name': 'reset_button',
                'label': 'Reset',
                'button_style': 'warning',
                'tooltip': 'Reset ke nilai default',
                'icon': 'undo'
            }
        ]
        
        action_container = create_action_container(
            buttons=action_buttons,
            show_save_reset=True,  # This will add default save/reset buttons if buttons list is empty
            **kwargs
        )
        
        # Create operation container (empty by default)
        operation_container = create_operation_container(
            children=[],
            **kwargs
        )
        
        # Create footer with log accordion
        footer_container = create_footer_container(
            log_accordion=log_accordion,
            show_progress=True,
            **kwargs
        )
        
        # Create main container
        main_container = create_main_container(
            header=header_container,
            body=widgets.VBox([
                form_container['container'],
                action_container['container']
            ]),
            footer=footer_container,
            **kwargs
        )
        
        # Create summary and info box
        summary_content = _create_module_summary_content(components)
        info_box = _create_module_info_box()
        
        # Get buttons from action container
        buttons = {}
        if hasattr(action_container, 'get'):
            # Try to get buttons from action container
            save_btn = action_container.get('save_button')
            reset_btn = action_container.get('reset_button')
            
            # If buttons weren't found in the container, try to create them
            if not save_btn or not reset_btn:
                from ipywidgets import Button
                
                if not save_btn:
                    save_btn = Button(
                        description='Simpan',
                        button_style='success',
                        icon='save',
                        tooltip='Simpan konfigurasi split dataset'
                    )
                
                if not reset_btn:
                    reset_btn = Button(
                        description='Reset',
                        button_style='warning',
                        icon='undo',
                        tooltip='Reset ke nilai default'
                    )
            
            buttons = {
                'save_button': save_btn,
                'reset_button': reset_btn
            }
        
        # Update components dictionary
        ui_components.update({
            # Form components
            'form_components': components,
            'form_container': form_container,
            'action_container': action_container,
            'operation_container': operation_container,
            'footer_container': footer_container,
            
            # Individual components
            'summary_content': summary_content,
            'info_box': info_box,
            'log_accordion': log_accordion,
            
            # Buttons
            'buttons': buttons,
            'save_button': buttons.get('save_button'),
            'reset_button': buttons.get('reset_button'),
            
            # Additional UI elements
            'summary_content': summary_content,
            'main_container': main_container
        })
        
        return ui_components
    
    except Exception as e:
        # The error will be handled by the @handle_ui_errors decorator
        raise
