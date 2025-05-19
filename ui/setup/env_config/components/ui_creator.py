"""
File: smartcash/ui/setup/env_config/components/ui_creator.py
Deskripsi: Creator untuk komponen UI environment config
"""

from typing import Dict, Any
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML, FloatProgress

from smartcash.ui.utils.alert_utils import create_info_box
from smartcash.ui.utils.header_utils import create_header
from smartcash.ui.utils.constants import COLORS

def create_env_config_ui() -> Dict[str, Any]:
    """
    Create the environment configuration UI components.
    Returns:
        dict: Dictionary of UI components
    """
    # Header
    header = HTML("<h2>Environment Configuration</h2>")

    # Buttons
    drive_button = Button(description="Connect to Google Drive", button_style="primary")
    directory_button = Button(description="Set Up Directories", button_style="primary")
    button_layout = HBox([drive_button, directory_button])

    # Status Panel
    status_panel = Output()

    # Log Panel
    log_panel = Output()

    # Progress Bar
    progress_bar = FloatProgress(min=0, max=100, value=0)
    progress_message = Label(value="")

    # Assemble UI components
    ui_components = {
        'header': header,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'status_panel': status_panel,
        'log_panel': log_panel,
        'progress_bar': progress_bar,
        'progress_message': progress_message,
        'button_layout': button_layout
    }

    # Create a VBox layout for the entire UI with consistent styling
    ui_layout = VBox([
        header,
        button_layout,
        status_panel,
        log_panel,
        progress_bar,
        progress_message
    ], layout=widgets.Layout(
        padding='10px',
        border='1px solid #ddd',
        background_color='#f9f9f9'
    ))

    # Add the layout to the components
    ui_components['ui_layout'] = ui_layout

    return ui_components 