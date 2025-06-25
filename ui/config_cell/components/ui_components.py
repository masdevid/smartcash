"""
File: smartcash/ui/config_cell/components/ui_components.py
Deskripsi: UI components for config cell with collapsible log accordion
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def _create_log_output() -> widgets.Output:
    """Create a styled output widget for logs.
    
    Returns:
        widgets.Output: Configured output widget for logs
    """
    output = widgets.Output()
    output.add_class('config-cell-log-output')
    output.layout = {
        'border': '1px solid #e0e0e0',
        'border_radius': '4px',
        'padding': '8px',
        'max_height': '200px',
        'overflow_y': 'auto',
        'background': '#f8f9fa',
        'font_family': 'monospace',
        'font_size': '12px',
    }
    return output

def create_config_cell_ui(module_name: str) -> Dict[str, Any]:
    """Create UI components for config cell with collapsible log accordion.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Dictionary of UI components including container, buttons, and log output
    """
    try:
        # Create header
        header = create_header(
            title=f"⚙️ {module_name.replace('_', ' ').title()}",
            subtitle="Configuration"
        )
        
        # Create action buttons

        
        # Create status panel
        status_bar = create_status_panel()
        
        # Create log output
        log_output = _create_log_output()
        
        # Create log accordion (closed by default)
        log_accordion = create_log_accordion(
            log_output=log_output,
            title='Logs',
            expanded=False,  # Start collapsed
            layout={
                'width': '100%',
                'margin': '5px 0 0 0'
            }
        )
        
        # Create container with all components
        container = widgets.VBox([
            header,
            widgets.HBox([save_btn, load_btn], layout={'margin': '0 0 5px 0'}),
            status_bar,
            log_accordion
        ], layout={
            'border': '1px solid #e0e0e0',
            'border_radius': '5px',
            'padding': '10px',
            'margin': '5px 0'
        })
        
        return {
            'header': header,
            'save_button': save_btn,
            'load_button': load_btn,
            'status_bar': status_bar,
            'log_output': log_output,
            'log_accordion': log_accordion,
            'container': container
        }
        
    except Exception as e:
        logger.error(f"Error creating config cell UI: {str(e)}")
        # Return minimal UI with error state
        error_msg = widgets.HTML(
            value=f'<div style="color: red; padding: 10px;">Error: {str(e)}</div>'
        )
        return {
            'container': widgets.VBox([error_msg]),
            'log_output': _create_log_output()
        }
