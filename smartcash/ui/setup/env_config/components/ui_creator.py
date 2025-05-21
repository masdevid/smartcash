"""
File: smartcash/ui/setup/env_config/components/ui_creator.py
Deskripsi: Creator untuk komponen UI environment config
"""

from typing import Dict, Any
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Label, Button, Output, HTML, FloatProgress

from smartcash.ui.setup.env_config.components.ui_factory import UIFactory, create_ui_components

def create_env_config_ui() -> Dict[str, Any]:
    """
    Create the environment configuration UI components.
    Returns:
        dict: Dictionary of UI components
    """
    return UIFactory.create_ui_components() 