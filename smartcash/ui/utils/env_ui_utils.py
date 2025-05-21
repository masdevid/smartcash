"""
File: smartcash/ui/utils/env_ui_utils.py
Deskripsi: Utilitas UI untuk environment config
"""

from typing import Dict, Any
import ipywidgets as widgets
from datetime import datetime

def update_status(ui_components: Dict[str, Any], message: str, style: str = "info") -> None:
    """
    Update status panel with alert
    
    Args:
        ui_components: Dictionary UI components
        message: Status message
        style: Alert style (info, success, error)
    """
    from smartcash.ui.utils.alert_utils import create_info_box
    ui_components['status_panel'].value = create_info_box(
        "Environment Status",
        message,
        style=style
    ).value

def set_button_state(button: widgets.Button, disabled: bool, style: str = None) -> None:
    """
    Update button state
    
    Args:
        button: Button widget
        disabled: Whether button should be disabled
        style: Button style
    """
    button.disabled = disabled
    if style:
        button.button_style = style

def log_message(ui_components: Dict[str, Any], message: str, level: str = "info") -> None:
    """
    Log message to output
    
    Args:
        ui_components: Dictionary UI components
        message: Message to log
        level: Log level (info, error)
    """
    # Jika pesan kosong, jangan log
    if not message or not message.strip():
        return
        
    from IPython.display import display
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_widget = ui_components.get('log_output')  # Use Output widget for logging
    if log_widget is not None:
        with log_widget:
            if level == "error":
                print(f"[{timestamp}] ❌ {message}")
            else:
                print(f"[{timestamp}] ℹ️ {message}")
    else:
        # Fallback to print if log_output is missing
        print(f"[{timestamp}] {message}")

def update_progress(ui_components: Dict[str, Any], value: float, message: str = "") -> None:
    """
    Update progress bar and message
    
    Args:
        ui_components: Dictionary UI components
        value: Progress value (0-1)
        message: Progress message
    """
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = value
    
    if 'progress_message' in ui_components and message:  # Hanya update jika message tidak kosong
        ui_components['progress_message'].value = message 