"""
Form widgets for Colab UI.

This module contains form widgets and configuration for the Colab setup interface.
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create Colab-specific form widgets.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        Dictionary containing form widgets and UI
    """
    # Create form widgets with default values from config
    auto_detect = widgets.Checkbox(
        value=config.get('auto_detect', True),
        description='Auto-detect environment',
        indent=False,
        style={'description_width': 'initial'}
    )
    
    drive_path = widgets.Text(
        value=config.get('drive_path', '/content/drive/MyDrive'),
        description='Google Drive Path:',
        placeholder='Enter Google Drive mount path',
        style={'description_width': 'initial'}
    )
    
    project_name = widgets.Text(
        value=config.get('project_name', 'SmartCash'),
        description='Project Name:',
        placeholder='Enter project name',
        style={'description_width': 'initial'}
    )
    
    # Create form UI
    form_ui = widgets.VBox([
        widgets.HTML("<h4>🔧 Colab Environment Configuration</h4>"),
        auto_detect,
        drive_path,
        project_name,
        widgets.HTML(
            "<div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 4px; font-size: 0.9em; color: #666;'>"
            "<strong>💡 Configuration Tips:</strong><br>"
            "• Auto-detect will automatically configure environment settings<br>"
            "• Ensure drive path is correct for Google Drive mounting<br>"
            "• Project name will be used for folder structure"
            "</div>"
        )
    ], layout=widgets.Layout(width='100%', padding='10px 0'))
    
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
