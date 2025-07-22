"""
File: smartcash/ui/dataset/augmentation/components/form_widgets.py
Description: Form widgets for the augmentation UI module.
"""
from typing import Any, Dict

import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout

from .basic_opts_widget import create_basic_options_widget
from .advanced_opts_widget import create_advanced_options_widget
from .augtypes_opts_widget import create_augmentation_types_widget
from .live_preview_widget import create_live_preview_widget
from .form_containers import create_form_container


def create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets with live preview.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    # Create widget groups preserving original structure
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    augmentation_types = create_augmentation_types_widget()
    preview_widget = create_live_preview_widget()
    
    # Create 2x2 grid with original styling
    row1 = HBox([
        create_form_container(
            basic_options['container'], 
            "📋 Basic Options", 
            'basic_options', 
            '48%'
        ),
        create_form_container(
            advanced_options['container'], 
            "⚙️ Advanced Parameters", 
            'advanced_options', 
            '48%'
        )
    ], layout=Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='stretch',
        gap='15px',
        margin='8px 0'
    ))
    
    row2 = HBox([
        create_form_container(
            augmentation_types['container'], 
            "🔄 Augmentation Types", 
            'augmentation_types', 
            '48%'
        ),
        create_form_container(
            preview_widget['container'], 
            "👁️ Live Preview", 
            'preview_panel', 
            '48%'
        )
    ], layout=Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='stretch',
        gap='15px',
        margin='8px 0 15px 0'
    ))
    
    form_container = VBox([row1, row2])
    
    return {
        'container': form_container,
        'widgets': {
            **basic_options.get('widgets', {}),
            **advanced_options.get('widgets', {}),
            **augmentation_types.get('widgets', {}),
            'preview_widget': preview_widget
        }
    }
