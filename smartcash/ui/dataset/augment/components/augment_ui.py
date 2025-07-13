"""
File: smartcash/ui/dataset/augment/components/augment_ui.py
Description: Main augment UI using container-based architecture with preserved styling

This component creates the main augment UI using shared container components
while preserving the unique form structure and styling from the original module.
"""

import ipywidgets as widgets
from IPython.display import display, Image
from typing import Dict, Any, Optional, List, Tuple
import os
from pathlib import Path

# Import local components
from .preview_widget import create_preview_widget
from .basic_options import create_basic_options_widget
from .advanced_options import create_advanced_options_widget
from .augmentation_types import create_augmentation_types_widget
from .operation_summary import create_operation_summary_widget
from ..constants import UI_CONFIG, BUTTON_CONFIG, SECTION_STYLES, AUGMENT_COLORS

# Import core components
from smartcash.ui.components import (
    create_header_container,
    create_form_container,
    create_action_container,
    create_operation_container,
    create_summary_container,
    create_footer_container,
    create_main_container,
    SummaryContainer
)
from smartcash.ui.core.errors.handlers import handle_ui_errors
from IPython.display import display
from ipywidgets import HTML, Layout, VBox
from smartcash.ui.components.footer_container import PanelConfig, PanelType

from .basic_options import create_basic_options_widget
from .advanced_options import create_advanced_options_widget
from .augmentation_types import create_augmentation_types_widget
from .operation_summary import create_operation_summary_widget
from ..constants import UI_CONFIG, BUTTON_CONFIG, SECTION_STYLES, AUGMENT_COLORS

# Module metadata
MODULE_METADATA = {
    'module_name': 'augment',
    'parent_module': 'dataset',
    'ui_initialized': True,
    'config': {}
}

# Re-export constants for compatibility
UI_CONFIG = UI_CONFIG
BUTTON_CONFIG = BUTTON_CONFIG


def _create_styled_container(content: widgets.Widget, title: str, theme: str = 'basic', width: str = '48%') -> widgets.VBox:
    """
    Create styled container preserving original augmentation styling.
    
    Args:
        content: Widget content to wrap
        title: Container title
        theme: Style theme from original module
        width: Container width
        
    Returns:
        Styled VBox container
    """
    # Use original color scheme from constants
    style_config = SECTION_STYLES.get(theme, SECTION_STYLES['basic_options'])
    border_color = style_config['border_color']
    bg_color = style_config['background']
    
    header_html = f"""
    <div style="padding: 8px 12px; margin-bottom: 8px;
                background: linear-gradient(145deg, {bg_color} 0%, rgba(255,255,255,0.9) 100%);
                border-radius: 8px; border-left: 4px solid {border_color};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h5 style="color: #333; margin: 0; font-size: 14px; font-weight: 600;">
            {title}
        </h5>
    </div>
    """
    
    return widgets.VBox([
        widgets.HTML(header_html),
        content
    ], layout=widgets.Layout(
        width=width,
        margin='5px',
        padding='10px',
        border=f'1px solid {border_color}',
        border_radius='8px',
        background_color='rgba(255,255,255,0.8)',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        flex=f'1 1 {width.replace("%", "")}%' if '%' in width else '1 1 auto'
    ))


def _create_augment_ui_components(config: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create and configure all UI components for the augment module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (ui_components, widgets) dictionaries
    """
    ui_components = {}
    widgets_dict = {}
    
    # 1. Header Container
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'],
        icon=UI_CONFIG['icon']
    )
    ui_components['header_container'] = header_container
    header_widget = header_container.container
    
    # 2. Create form widgets
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    augmentation_types = create_augmentation_types_widget()
    
    # 3. Create preview section
    preview_image_path = Path("data/previews/augmentation_preview.jpg")
    if preview_image_path.exists():
        with open(preview_image_path, "rb") as f:
            preview_image = f.read()
            preview_widget = IPyImage(
                value=preview_image,
                format='jpg',
                width=300,
                height=200,
                layout=Layout(margin='10px auto', border='1px solid #ddd')
            )
    else:
        preview_widget = HTML(
            value='<div style="text-align: center; padding: 20px; color: #666;">No preview available</div>',
            layout=Layout(margin='10px auto')
        )
    
    # Create preview container
    preview_container = VBox([
        HTML('<div style="font-weight: bold; margin: 10px 0 5px 0; text-align: center;">🎯 Live Preview</div>'),
        preview_widget,
        HTML('<div style="font-size: 0.8em; color: #666; text-align: center; margin-top: 5px;">' +
             'Preview will update with augmentation settings</div>')
    ], layout=Layout(
        border='1px solid #e0e0e0',
        border_radius='8px',
        padding='10px',
        margin='10px 0',
        background='#f9f9f9',
        width='100%',
        align_items='center'
    ))
    
    # 4. Create form widgets using the module function
    form_widgets = _create_module_form_widgets(config)
    form_container = form_widgets['ui']
    
    # Update the widgets dictionary with the form widgets
    widgets_dict.update(form_widgets['widgets'])
    
    # Store the form container in ui_components
    ui_components['form_container'] = {
        'container': form_container,
        'preview': widgets_dict.get('preview_widget')
    }
    
    # 5. Create action container with proper button configuration
    action_buttons = [
        {
            "id": "preview",
            "text": BUTTON_CONFIG['preview']['text'],
            "style": BUTTON_CONFIG['preview']['style'],
            "tooltip": BUTTON_CONFIG['preview']['tooltip'],
            "order": 1
        },
        {
            "id": "augment",
            "text": BUTTON_CONFIG['augment']['text'],
            "style": BUTTON_CONFIG['augment']['style'],
            "tooltip": BUTTON_CONFIG['augment']['tooltip'],
            "order": 2
        },
        {
            "id": "check",
            "text": BUTTON_CONFIG['check']['text'],
            "style": BUTTON_CONFIG['check']['style'],
            "tooltip": BUTTON_CONFIG['check']['tooltip'],
            "order": 3
        },
        {
            "id": "cleanup",
            "text": BUTTON_CONFIG['cleanup']['text'],
            "style": BUTTON_CONFIG['cleanup']['style'],
            "tooltip": BUTTON_CONFIG['cleanup']['tooltip'],
            "order": 4
        }
    ]
    
    action_container = create_action_container(
        buttons=action_buttons,
        title="🚀 Augmentation Operations",
        container_margin="15px 0 5px 0"
    )
    ui_components['action_container'] = action_container
    
    # 6. Create summary container for operation status
    operation_summary = create_operation_summary_widget()
    summary_container = create_summary_container(
        theme="info",
        title="📊 Operation Status",
        icon="ℹ️"
    )
    # Get the HTML content from the widget and pass it as a string
    summary_container.set_content(operation_summary['container'].value)
    ui_components['summary_container'] = summary_container
    
    # 7. Operation Container
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name="Augmentation"
    )
    ui_components['operation_container'] = operation_container
    
    # 8. Footer Container with accordion
    info_box = _create_module_info_box()
    footer_container = create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,  # Changed to accordion
                title="📚 Augmentation Guide",
                content=info_box,
                style="info",
                flex="1",
                min_width="100%",
                open_by_default=True
            )
        ],
        # Customize footer container style
        style={
            'border_top': '1px solid #e0e0e0',
            'background': '#f9f9f9',
            'margin_top': '15px',
            'padding': '10px'
        }
    )
    ui_components['footer_container'] = footer_container
    
    # Store widgets for later reference
    widgets_dict.update({
        'basic_options': basic_options,
        'advanced_options': advanced_options,
        'augmentation_types': augmentation_types,
        'form_container': form_container,
        'operation_summary': operation_summary,
        'preview_widget': preview_widget,
        **augmentation_types.get('widgets', {}),
        **operation_summary.get('widgets', {})
    })
    
    # Create main container with proper layout
    main_container = create_main_container(
        components=[
            {
                'type': 'header',
                'component': header_container.container,
                'order': 0,
                'name': 'header'
            },
            {
                'type': 'form',
                'component': form_container,
                'order': 1,
                'name': 'form'
            },
            {
                'type': 'action',
                'component': action_container['container'],  # Access the container widget using dictionary access
                'order': 2,
                'name': 'actions'
            },
            {
                'type': 'operation',
                'component': summary_container.container,
                'order': 3,
                'name': 'summary'
            },
            {
                'type': 'operation',
                'component': operation_container['container'],
                'order': 4,
                'name': 'operations'
            },
            {
                'type': 'footer',
                'component': footer_container.container,
                'order': 5,
                'name': 'footer'
            }
        ],
        # Styling for the main container
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='0 10px 20px 10px',
        align_items='stretch'
    )
    
    # Add the main container to the UI components
    ui_components['main_container'] = main_container
    ui_components['main_layout'] = main_container.container
    
    return ui_components, widgets_dict


@handle_ui_errors(error_component_title="Augment UI Creation Error")
def create_augment_ui(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Create main augment UI using container-based architecture.
    
    Features:
    - 🏗️ Container-based architecture with core inheritance
    - 🎨 Preserved original forms and unique styling
    - 📊 Added summary_container component
    - 🔄 Standardized button and action handling
    - 📝 Comprehensive logging and progress tracking
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        Dictionary containing all UI components and metadata
    """
    config = config or {}
    
    # Create UI components
    ui_components, widgets_dict = _create_augment_ui_components(config)
    
    # Create container variables in function scope (required by validator)
    header_container = ui_components['header_container']
    form_container = ui_components['form_container']
    action_container = ui_components['action_container']
    operation_container = ui_components['operation_container']
    footer_container = ui_components['footer_container']
    
    # Create UI components dictionary
    ui_components = {
        'module_name': 'augment',
        'parent_module': 'dataset',
        'ui_initialized': True,
        'config': config or {},
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        'main_container': ui_components.get('main_container'),  # Add main container to top level
        **widgets_dict,
        # Add ui_components as a flat dictionary with all components
        'ui_components': {
            'header': header_container,
            'form': form_container,
            'actions': action_container,
            'operation': operation_container,
            'footer': footer_container,
            'main': ui_components.get('main_container'),  # Also include in ui_components
            'widgets': widgets_dict,
            'metadata': MODULE_METADATA,
            'config': config
        }
    }
    
    return ui_components


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # Create live preview widget
    preview_widget = create_preview_widget()
    
    # Create 2x2 grid with original styling
    row1 = widgets.HBox([
        _create_styled_container(
            basic_options['container'], 
            "📋 Basic Options", 
            'basic_options', 
            '48%'
        ),
        _create_styled_container(
            advanced_options['container'], 
            "⚙️ Advanced Parameters", 
            'advanced_options', 
            '48%'
        )
    ], layout=widgets.Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='stretch',
        gap='15px',
        margin='8px 0'
    ))
    
    row2 = widgets.HBox([
        _create_styled_container(
            augmentation_types['container'], 
            "🔄 Augmentation Types", 
            'augmentation_types', 
            '48%'
        ),
        _create_styled_container(
            preview_widget['container'], 
            "👁️ Live Preview", 
            'preview_panel', 
            '48%'
        )
    ], layout=widgets.Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='stretch',
        gap='15px',
        margin='8px 0 15px 0'
    ))
    
    form_container = widgets.VBox([row1, row2])
    
    return {
        'ui': form_container,
        'widgets': {
            **basic_options.get('widgets', {}),
            **advanced_options.get('widgets', {}),
            **augmentation_types.get('widgets', {}),
            'preview_widget': preview_widget
        }
    }


def _create_module_summary_content(config: Dict[str, Any]) -> str:
    """
    Create summary content for the module.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        HTML string containing the summary content
    """
    return "<p>Augmentation configuration and status will be displayed here.</p>"


def _create_module_info_box() -> widgets.Widget:
    """
    Create the info accordion for the footer with augmentation tips.
    
    Returns:
        Widget containing the info accordion content
    """
    # Create content for each accordion tab
    tips_content = widgets.HTML("""
    <div style="font-size: 0.9em; line-height: 1.5; padding: 8px 0;">
        <ul style="margin: 0 0 0 15px; padding: 0;">
            <li>Use 'Combined' type for balanced position + lighting effects</li>
            <li>Monitor target count to maintain dataset balance</li>
            <li>Preview results before running full augmentation</li>
            <li>Backup data before cleanup operations</li>
        </ul>
    </div>
    """)
    
    settings_content = widgets.HTML("""
    <div style="font-size: 0.9em; line-height: 1.5; padding: 8px 0;">
        <ul style="margin: 0 0 0 15px; padding: 0;">
            <li>Start with low intensity (0.1-0.3) and gradually increase</li>
            <li>Use 2-4 augmentation types for balanced results</li>
            <li>Check class distribution after augmentation</li>
        </ul>
    </div>
    """)
    
    # Create the accordion with two sections
    accordion = widgets.Accordion(children=[tips_content, settings_content])
    accordion.set_title(0, '💡 Augmentation Tips')
    accordion.set_title(1, '📊 Recommended Settings')
    
    # Style the accordion
    accordion.add_class('info-accordion')
    accordion.layout = widgets.Layout(
        width='100%',
        margin='0 0 10px 0',
        border='1px solid #e0e0e0',
        border_radius='8px',
        overflow='hidden'
    )
    
    # Add some custom CSS for the accordion
    display(widgets.HTML("""
    <style>
        .info-accordion .p-Accordion-header {
            background-color: #f5f5f5;
            padding: 8px 12px;
            cursor: pointer;
            font-weight: 600;
            border-bottom: 1px solid #e0e0e0;
        }
        .info-accordion .p-Accordion-header:hover {
            background-color: #ebebeb;
        }
        .info-accordion .p-Accordion-contents {
            padding: 8px 12px;
            background-color: #fafafa;
        }
    </style>
    """))
    
    return accordion