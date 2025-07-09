"""
File: smartcash/ui/dataset/augment/components/augment_ui.py
Description: Main augment UI using container-based architecture with preserved styling

This component creates the main augment UI using shared container components
while preserving the unique form structure and styling from the original module.
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional
from smartcash.ui.core.errors.handlers import handle_ui_errors
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.action_container import create_action_container
from smartcash.ui.components.operation_container import create_operation_container
from smartcash.ui.components.footer_container import create_footer_container

from .basic_options import create_basic_options_widget
from .advanced_options import create_advanced_options_widget
from .augmentation_types import create_augmentation_types_widget
from .operation_summary import create_operation_summary_widget
from ..constants import UI_CONFIG, BUTTON_CONFIG, SECTION_STYLES, AUGMENT_COLORS


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


@handle_ui_errors(error_component_title="Augment UI Creation Error")
def create_augment_ui(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
        
    Returns:
        Dictionary containing all UI components and widgets
    """
    config = config or {}
    ui_components = {}
    
    # === CORE CONTAINERS ===
    
    # 1. Header Container
    header_container = create_header_container(
        title=UI_CONFIG['title'],
        subtitle=UI_CONFIG['subtitle'], 
        icon=UI_CONFIG['icon']
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Form Container (will contain custom form layout)
    form_container = create_form_container()
    
    # 3. Action Container with augment-specific buttons
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
    ui_components['action_container'] = action_container['container']
    
    # 4. Operation Container for status, progress, and logging
    operation_container = create_operation_container(
        title="📊 Augmentation Status",
        show_progress=True,
        show_logs=True,
        log_module_name="Augment"
    )
    ui_components['operation_container'] = operation_container['container']
    
    # Store operation container functions for use by handlers
    ui_components['log_message'] = operation_container['log_message']
    ui_components['update_progress'] = operation_container['update_progress']
    ui_components['show_dialog'] = operation_container['show_dialog']
    ui_components['show_info_dialog'] = operation_container['show_info_dialog']
    ui_components['clear_dialog'] = operation_container['clear_dialog']
    
    # 5. Footer Container with info only
    info_html = f"""
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
    
    footer_container = create_footer_container(
        info_box=widgets.HTML(info_html)
    )
    ui_components['footer_container'] = footer_container.container
    
    # === CUSTOM FORM LAYOUT (PRESERVED) ===
    
    # Create widget groups preserving original structure
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    augmentation_types = create_augmentation_types_widget()
    operation_summary = create_operation_summary_widget()  # NEW: Summary container
    
    # Create 2x2 grid with original styling (PRESERVED LAYOUT)
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
            "📊 Operation Summary",  # NEW: Summary container 
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
        margin='8px 0'
    ))
    
    # Place custom form layout in form container
    form_container['container'].children = (widgets.VBox([row1, row2]),)
    ui_components['form_container'] = form_container['container']
    
    # === MAIN UI ASSEMBLY ===
    
    # Create main container following UI structure guidelines
    main_ui = widgets.VBox([
        header_container.container,
        form_container['container'],
        action_container['container'],
        operation_container['container'],
        footer_container.container
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#fafafa'
    ))
    
    ui_components['ui'] = main_ui
    
    # === COMPONENT MAPPING ===
    
    # Extract buttons from action container
    augment_button = action_container.get_button('augment')
    check_button = action_container.get_button('check')
    cleanup_button = action_container.get_button('cleanup')
    preview_button = action_container.get_button('preview')
    
    # Update ui_components with all widgets
    ui_components.update({
        # Widget groups
        **basic_options.get('widgets', {}),
        **advanced_options.get('widgets', {}),
        **augmentation_types.get('widgets', {}),
        **operation_summary.get('widgets', {}),  # NEW: Summary widgets
        
        # Action buttons
        'augment_button': augment_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'preview_button': preview_button,
        
        # Legacy compatibility
        'download_button': augment_button,  # For backward compatibility
        
        # Logging and progress
        'log_accordion': log_components.get('log_accordion'),
        'log_output': log_components.get('log_output'),
        'operation_status': operation_container.get_status_widget(),
        'operation_progress': operation_container.get_progress_widget(),
        
        # Metadata
        'module_name': UI_CONFIG['module_name'],
        'parent_module': UI_CONFIG['parent_module'],
        'logger_namespace': f"smartcash.ui.dataset.{UI_CONFIG['module_name']}",
        'augment_initialized': True,
        'config': config,
        'version': UI_CONFIG['version']
    })
    
    return ui_components