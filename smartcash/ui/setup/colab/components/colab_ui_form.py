"""
Form widgets for Colab UI.

This module contains form widgets and configuration for the Colab setup interface.
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create Colab-specific form widgets with preprocess UI style.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        Dictionary containing form widgets and UI
    """
    # Common layout for form elements
    input_layout = widgets.Layout(
        width='auto',
        margin='5px 0',
        padding='5px 0'
    )
    
    checkbox_layout = widgets.Layout(
        width='auto',
        margin='8px 0',
        padding='5px 0'
    )
    
    # Create form widgets with default values from config
    auto_detect = widgets.Checkbox(
        value=config.get('auto_detect', True),
        description='Auto-detect environment',
        indent=False,
        layout=checkbox_layout,
        style={'description_width': 'initial'}
    )
    
    drive_path = widgets.Text(
        value=config.get('drive_path', '/content/drive/MyDrive'),
        description='Google Drive Path:',
        placeholder='Enter Google Drive mount path',
        layout=input_layout,
        style={'description_width': '140px'}
    )
    
    project_name = widgets.Text(
        value=config.get('project_name', 'SmartCash'),
        description='Project Name:',
        placeholder='Enter project name',
        layout=input_layout,
        style={'description_width': '140px'}
    )
    
    # Create form sections with two-column layout
    config_section = widgets.VBox([
        widgets.HTML("<h4 style='margin: 10px 0 5px 0;'>🔧 Environment Settings</h4>"),
        auto_detect,
        drive_path,
        project_name
    ], layout=widgets.Layout(width='100%', margin='0 0 10px 0'))
    
    # Create form UI with clean layout
    form_ui = widgets.VBox([
        config_section
    ], layout=widgets.Layout(
        width='100%',
        padding='10px 15px',
        border='1px solid #e0e0e0',
        border_radius='4px',
        margin='5px 0'
    ))
    
    # Store widgets for later access
    form_widgets = {
        'auto_detect': auto_detect,
        'drive_path': drive_path,
        'project_name': project_name,
        'form_ui': form_ui
    }
    
    return form_widgets

# For backward compatibility
_create_module_form_widgets = create_module_form_widgets
