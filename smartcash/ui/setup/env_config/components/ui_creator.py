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
from smartcash.ui.components.status_panel import create_status_panel
from smartcash.ui.components.progress_tracking import create_progress_tracking
from smartcash.ui.components.log_accordion import create_log_accordion
from smartcash.ui.utils.layout_utils import STANDARD_LAYOUTS

def create_env_config_ui() -> Dict[str, Any]:
    """
    Create the environment configuration UI components.
    Returns:
        dict: Dictionary of UI components
    """
    # Header
    header = create_header("Environment Configuration")

    # Buttons
    drive_button = Button(description="Connect to Google Drive", button_style="primary")
    directory_button = Button(description="Set Up Directories", button_style="primary")
    button_layout = HBox([drive_button, directory_button], layout=STANDARD_LAYOUTS['hbox'])

    # Status Panel
    status_panel = create_status_panel("Ready to configure environment", "info")

    # Log Panel
    log_panel = create_log_accordion("Environment Configuration Log")
    log_accordion = log_panel['log_accordion']  # Accordion widget for display
    log_output = log_panel['log_output']        # Output widget for logging

    # Progress Bar
    progress_components = create_progress_tracking(module_name="env_config")
    progress_components['progress_message'] = Label(value="")  # Ensure progress_message is included

    # Assemble UI components
    ui_components = {
        'header': header,
        'drive_button': drive_button,
        'directory_button': directory_button,
        'status_panel': status_panel,
        'log_panel': log_accordion,   # For display
        'log_output': log_output,     # For logging
        'progress_bar': progress_components['progress_bar'],
        'progress_message': progress_components['progress_message'],
        'button_layout': button_layout
    }

    # Create a VBox layout for the entire UI with consistent styling
    ui_layout = VBox([
        header,
        button_layout,
        status_panel,
        log_accordion,  # Use the actual widget
        progress_components['progress_container']
    ], layout=STANDARD_LAYOUTS['vbox'])

    # Add the layout to the components
    ui_components['ui_layout'] = ui_layout

    return ui_components 