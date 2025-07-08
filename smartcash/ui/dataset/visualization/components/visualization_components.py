"""
File: smartcash/ui/dataset/visualization/visualization_components.py
Deskripsi: Komponen utama untuk visualisasi dataset
"""

from typing import Dict, Any, Optional, List, Callable
import ipywidgets as widgets
import pandas as pd
from IPython.display import display

# Import VisualizationUI from visualization_ui to avoid circular imports
from .visualization_ui import VisualizationUI

# Import visualization components from parent module to avoid circular imports
from ..visualization_initializer import (
    VisualizationInitializer,
    VisualizationHandler,
    initialize_visualization
)

# Import UI components
from smartcash.ui.components import (
    create_header,
    create_tab_widget as create_tab,
    create_section_title
)

# Import constants
from smartcash.ui.utils.constants import ICONS


def create_visualization_ui(data: pd.DataFrame, title: str = "Data Visualization") -> VisualizationUI:
    """
    Factory function untuk membuat UI visualisasi
    
    Args:
        data: DataFrame berisi data yang akan divisualisasikan
        title: Judul untuk UI
        
    Returns:
        Instance dari VisualizationUI
    """
    return VisualizationUI(data, title)
