"""
File: smartcash/ui/dataset/augment/components/augment_ui.py
Description: Main augment UI using container-based architecture with preserved styling

This component creates the main augment UI using shared container components
while preserving the unique form structure and styling from the original module.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional, Tuple
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container, PanelConfig, PanelType

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
    header_widget = header_container.container  # Get the actual widget from the container
    
    # 2. Form Container
    form_container = create_form_container()
    ui_components['form_container'] = form_container
    form_container_widget = form_container['container']  # Get the actual widget from the container
    
    # 3. Action Container
    action_buttons = [
        {
            "id": "augment",
            "text": BUTTON_CONFIG['augment']['text'],
            "style": BUTTON_CONFIG['augment']['style'],
            "tooltip": BUTTON_CONFIG['augment']['tooltip'],
            "order": BUTTON_CONFIG['augment']['order']
        },
        {
            "id": "check",
            "text": BUTTON_CONFIG['check']['text'],
            "style": BUTTON_CONFIG['check']['style'],
            "tooltip": BUTTON_CONFIG['check']['tooltip'],
            "order": BUTTON_CONFIG['check']['order']
        },
        {
            "id": "cleanup",
            "text": BUTTON_CONFIG['cleanup']['text'],
            "style": BUTTON_CONFIG['cleanup']['style'],
            "tooltip": BUTTON_CONFIG['cleanup']['tooltip'],
            "order": BUTTON_CONFIG['cleanup']['order']
        },
        {
            "id": "preview",
            "text": BUTTON_CONFIG['preview']['text'],
            "style": BUTTON_CONFIG['preview']['style'],
            "tooltip": BUTTON_CONFIG['preview']['tooltip'],
            "order": BUTTON_CONFIG['preview']['order']
        }
    ]
    
    action_container = create_action_container(
        buttons=action_buttons,
        title="🚀 Augmentation Operations",
        alignment="left"
    )
    ui_components['action_container'] = action_container
    action_widget = action_container['container']  # Get the actual widget from the container
    
    # 4. Operation Container
    operation_container = create_operation_container(
        show_progress=True,
        show_dialog=True,
        show_logs=True,
        log_module_name="Augmentation"
    )
    ui_components['operation_container'] = operation_container
    operation_widget = operation_container['container']  # Get the actual widget from the container
    
    # 5. Footer Container
    info_box = _create_module_info_box()
    footer_container = create_footer_container(
        panels=[
            PanelConfig(
                panel_type=PanelType.INFO_BOX,
                title="ℹ️ Information",
                content=info_box,
                flex="1",
                min_width="100%"
            )
        ]
    )
    ui_components['footer_container'] = footer_container
    footer_widget = footer_container.container  # Get the actual widget from the container
    
    # Create form widgets
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    augmentation_types = create_augmentation_types_widget()
    operation_summary = create_operation_summary_widget()
    
    # Create form layout
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
            operation_summary['container'],
            "📊 Operation Summary",
            'live_preview',
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
    
    # Update form container
    form_container['add_item'](widgets.VBox([row1, row2]))
    
    # Assemble main UI
    main_ui = widgets.VBox([
        header_widget,  # Use the widget directly
        form_container_widget,  # Use the widget directly
        action_widget,  # Use the widget directly
        operation_widget,  # Use the widget directly
        footer_widget  # Use the widget directly
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#fafafa'
    ))
    
    ui_components['ui'] = main_ui
    
    # Collect all widgets
    widgets_dict.update({
        # Widget groups
        **basic_options.get('widgets', {}),
        **advanced_options.get('widgets', {}),
        **augmentation_types.get('widgets', {}),
        **operation_summary.get('widgets', {}),
        
        # Action buttons - access from the action_container dictionary
        'augment_button': action_container['buttons'].get('augment'),
        'check_button': action_container['buttons'].get('check'),
        'cleanup_button': action_container['buttons'].get('cleanup'),
        'preview_button': action_container['buttons'].get('preview'),
        
        # Operation widgets - access from the operation_container dictionary
        'operation_status': operation_container.get('status_widget'),
        'operation_progress': operation_container.get('progress_widget'),
        'log_output': operation_container.get('log_output')
    })
    
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
        'ui': ui_components['ui'],
        'module_name': 'augment',
        'parent_module': 'dataset',
        'ui_initialized': True,
        'config': config or {},
        'header_container': header_container,
        'form_container': form_container,
        'action_container': action_container,
        'operation_container': operation_container,
        'footer_container': footer_container,
        **widgets_dict,
        # Add ui_components as a flat dictionary with all components
        'ui_components': {
            'header': header_container,
            'form': form_container,
            'actions': action_container,
            'operation': operation_container,
            'footer': footer_container,
            'widgets': widgets_dict,
            'metadata': MODULE_METADATA,
            'config': config
        }
    }
    
    return ui_components


def _create_module_form_widgets(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create module-specific form widgets.
    
    Args:
        config: Configuration dictionary for the form widgets
        
    Returns:
        Dictionary containing the form UI and widget references
    """
    # Create widget groups preserving original structure
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    augmentation_types = create_augmentation_types_widget()
    operation_summary = create_operation_summary_widget()
    
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
            operation_summary['container'], 
            "📊 Operation Summary", 
            'live_preview', 
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
            **operation_summary.get('widgets', {})
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
    Create the info box content for the footer.
    
    Returns:
        Widget containing the info box content
    """
    return widgets.HTML(
        value=f"""
        <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px; 
             background: {SECTION_STYLES['basic_options']['background']}; 
             border-left: 4px solid {AUGMENT_COLORS['info']}; border-radius: 4px;">
            <strong>💡 Augmentation Tips:</strong>
            <ul style="margin: 5px 0 0 15px; padding: 0; line-height: 1.4;">
                <li>Use 'Combined' type for balanced position + lighting effects</li>
                <li>Monitor target count to maintain dataset balance</li>
                <li>Preview results before running full augmentation</li>
                <li>Backup data before cleanup operations</li>
            </ul>
        </div>
        """
    )