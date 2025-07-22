"""
Footer components for Colab UI.

This module contains footer components and info boxes for the Colab setup interface.
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_module_info_box() -> widgets.HTML:
    """Create Colab-specific info box for footer.
    
    Returns:
        Widget containing module information
    """
    info_content = """
    <div style='font-size: 0.9em; line-height: 1.5;'>
        <h4 style='margin-top: 0;'>About Colab Setup</h4>
        <p>This module helps you set up your Google Colab environment for SmartCash.</p>
        <p><strong>Version:</strong> 1.0.0</p>
        <p><strong>Requirements:</strong></p>
        <ul style='margin: 5px 0; padding-left: 20px;'>
            <li>Google Colab runtime</li>
            <li>Google Drive access</li>
            <li>Python 3.7+</li>
        </ul>
    </div>
    """
    return widgets.HTML(info_content)

def create_module_tips_box() -> widgets.HTML:
    """Create Colab-specific tips box for footer.
    
    Returns:
        Widget containing helpful tips
    """
    tips_content = """
    <div style='font-size: 0.9em; line-height: 1.5;'>
        <h4 style='margin-top: 0;'>Quick Tips</h4>
        <ul style='margin: 5px 0; padding-left: 20px;'>
            <li>Use auto-detect to automatically configure your environment</li>
            <li>Mount Google Drive to save your work</li>
            <li>Check the logs for detailed information</li>
            <li>Click the help icons for more information</li>
        </ul>
    </div>
    """
    return widgets.HTML(tips_content)

# For backward compatibility
_create_module_info_box = create_module_info_box
_create_module_tips_box = create_module_tips_box
