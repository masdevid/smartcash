"""
Environment container for Colab UI.

This module contains the environment information container and related functions.
"""

import ipywidgets as widgets
from typing import Dict, Any

from smartcash.ui.setup.colab.components import env_info_panel

def create_environment_container(config: Dict[str, Any]) -> widgets.VBox:
    """Create environment information container.
    
    Args:
        config: Module configuration dictionary
        
    Returns:
        VBox widget containing environment information and tips
    """
    # Create environment info panel with lazy loading
    env_info = env_info_panel.create_env_info_panel(config, lazy_load=True)
    
    # Create environment container
    return widgets.VBox([
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
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#ffffff'
    ))

# For backward compatibility
_create_environment_container = create_environment_container
