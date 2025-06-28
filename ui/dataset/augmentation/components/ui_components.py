"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Updated UI components dengan live preview integration dan cleanup target
"""

from IPython.display import display, HTML
import ipywidgets as widgets
from typing import Dict, Any

# Internal components
from smartcash.ui import components
from smartcash.ui.dataset.augmentation.utils.style_utils import (
    styled_container, flex_layout
)
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components import (
    create_header,
    create_confirmation_area,
    create_action_buttons,
    create_status_panel,
    create_log_accordion,
    create_save_reset_buttons,
    create_dual_progress_tracker
)

def _create_section_header(title: str, color: str) -> widgets.HTML:
    """Create styled section header"""
    return widgets.HTML(f"""
    <h4 style="color: #333; margin: 10px 0 8px 0; border-bottom: 2px solid {color}; 
               font-size: 14px; padding-bottom: 4px;">
        {title}
    </h4>
    """)

def _create_basic_options_group() -> Dict[str, Any]:
    """Basic options group dengan cleanup target integration"""
    from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
    return create_basic_options_widget()

def _create_advanced_options_group() -> Dict[str, Any]:
    """Advanced options group dengan HSV parameters"""
    from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
    return create_advanced_options_widget()

def _create_augmentation_types_group() -> Dict[str, Any]:
    """Augmentation types group"""
    from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
    return create_augmentation_types_widget()

def _create_live_preview_group() -> Dict[str, Any]:
    """Live preview group"""
    from smartcash.ui.dataset.augmentation.components.live_preview_widget import create_live_preview_widget
    return create_live_preview_widget()

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Main UI dengan live preview integration dan cleanup target update"""
    
    # Initialize ui_components dictionary
    ui_components = {}
    
    # Create confirmation area first with ui_components
    confirmation_area = create_confirmation_area(ui_components=ui_components)
    ui_components['confirmation_area'] = confirmation_area  # Store reference in ui_components
    
    # Header and status panel with consistent styling
    header = create_header(
        "Dataset Augmentation",
        "Pipeline augmentasi dengan live preview dan backend integration",
        "🔄"
    )
    status_panel = create_status_panel("✅ Pipeline augmentasi siap", "success")
    
    # Widget groups dengan live preview integration
    basic_options = _create_basic_options_group()
    advanced_options = _create_advanced_options_group()
    augmentation_types = _create_augmentation_types_group()
    live_preview = _create_live_preview_group()  # CHANGED: Menggantikan normalization
    
    # Progress tracker and buttons
    progress_tracker = create_dual_progress_tracker("Augmentation Pipeline", auto_hide=True)
    
    # Save/Reset buttons
    config_buttons = create_save_reset_buttons(
        save_label="Simpan", 
        reset_label="Reset",
        with_sync_info=True, 
        sync_message="Konfigurasi disinkronkan dengan backend"
    )
    
    # Action buttons with new API
    action_buttons = create_action_buttons(
        primary_button={
            "label": "🚀 Jalankan Augmentasi",
            "style": "success",
            "width": "220px"
        },
        secondary_buttons=[
            {
                "label": "🔍 Cek Data",
                "style": "info",
                "width": "220px"
            },
            {
                "label": "🗑️ Bersihkan Hasil",
                "style": "warning",
                "tooltip": "Hapus hasil augmentasi sebelumnya",
                "width": "220px"
            }
        ]
    )
    
    # Get buttons from the new action buttons component
    augment_button = action_buttons.get('primary')
    check_button = action_buttons.get('secondary_0')
    cleanup_button = action_buttons.get('secondary_1')
    button_container = action_buttons['container']
    
    # Fallback button creation if any button is missing
    if augment_button is None:
        print("[WARNING] Augment button not found, creating fallback")
        augment_button = widgets.Button(description='🚀 Jalankan Augmentasi', 
                                     button_style='success')
        augment_button.layout = widgets.Layout(width='220px')
    
    if check_button is None:
        print("[WARNING] Check button not found, creating fallback")
        check_button = widgets.Button(description='🔍 Cek Data')
        check_button.style.button_color = '#f0f0f0'
        check_button.layout = widgets.Layout(width='220px')
    
    if cleanup_button is None:
        print("[WARNING] Cleanup button not found, creating fallback")
        cleanup_button = widgets.Button(description='🗑️ Bersihkan Hasil',
                                      button_style='warning')
        cleanup_button.layout = widgets.Layout(width='220px')
    
    # Update action_buttons for backward compatibility
    action_buttons = {
        'container': button_container,
        'primary': augment_button,
        'secondary_0': check_button,
        'secondary_1': cleanup_button,
        'augment_button': augment_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'download_button': augment_button,  # For backward compatibility
        'buttons': [augment_button, check_button, cleanup_button] if cleanup_button else [augment_button, check_button]
    }
    
    # Initialize UI components dictionary
    ui_components = {}
    
    # Get confirmation area from ui_components
    confirmation_area = ui_components.get('confirmation_area')
    
    # Log accordion
    log_components = create_log_accordion('augmentation', '250px')
    
    # Import style utilities
    from smartcash.ui.dataset.augmentation.utils.style_utils import styled_container
    
    # 2x2 Grid with original styling and gradients
    row1 = widgets.HBox([
        styled_container(basic_options['container'], "📋 Opsi Dasar", 'basic', '48%'),
        styled_container(advanced_options['container'], "⚙️ Parameter Lanjutan", 'advanced', '48%')
    ], layout=widgets.Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='stretch',
        gap='15px',
        margin='8px 0',
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    row2 = widgets.HBox([
        styled_container(augmentation_types['container'], "🔄 Jenis Augmentasi", 'types', '48%'),
        styled_container(live_preview['container'], "🎬 Live Preview", 'normalization', '48%')
    ], layout=widgets.Layout(
        width='100%',
        display='flex',
        flex_flow='row wrap',
        justify_content='space-between',
        align_items='stretch',
        gap='15px',
        margin='8px 0',
        overflow='hidden',
        box_sizing='border-box'
    ))
    
    # Import shared action section component
    from smartcash.ui.components.action_section import create_action_section
    
    # Create action section using shared component
    action_section = create_action_section(
        action_buttons=action_buttons,
        confirmation_area=confirmation_area,
        title="🚀 Operations",
        status_label="📋 Status & Konfirmasi:",
        show_status=True,
        ui_components=ui_components
    )
    
    # Config section with consistent styling
    config_section = widgets.VBox([
        widgets.Box([config_buttons['container']], 
            layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
    ], layout=widgets.Layout(margin='8px 0'))
    
    # Main UI assembly with consistent styling
    ui = widgets.VBox([
        header,
        status_panel,
        row1,
        row2,
        config_section,
        action_section,
        progress_tracker.container if hasattr(progress_tracker, 'container') else widgets.VBox([]),
        log_components['log_accordion']
    ], layout=widgets.Layout(
        width='100%',
        max_width='1200px',
        margin='0 auto',
        padding='15px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        box_shadow='0 2px 4px rgba(0,0,0,0.05)'
    ))
    
    # Update ui_components with all components
    ui_components.update({
        'ui': ui, 
        'header': header, 
        'status_panel': status_panel,
        'confirmation_area': confirmation_area,
        **basic_options.get('widgets', {}), 
        **advanced_options.get('widgets', {}),
        **augmentation_types.get('widgets', {}), 
        **live_preview.get('widgets', {}),  # Live preview widgets
        'augment_button': augment_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'download_button': augment_button,  # For backward compatibility
        'save_button': config_buttons.get('save_button'),
        'reset_button': config_buttons.get('reset_button'),
        'progress_tracker': progress_tracker,
        'log_accordion': log_components.get('log_accordion'),
        'log_output': log_components.get('log_output'),
        'status': log_components.get('log_output'),
        'backend_ready': True, 
        'service_integration': True,
        'module_name': 'augmentation',
        'logger_namespace': 'smartcash.ui.dataset.augmentation',
        'augmentation_initialized': True,
        'config': config or {}
    })
    
    # Make sure confirmation area is properly initialized
    if 'confirmation_area' not in ui_components:
        ui_components['confirmation_area'] = confirmation_area
    
    # Add action buttons to ui_components
    ui_components.update({
        'augment_button': action_buttons.get('primary_button'),
        'check_button': next((btn for btn in action_buttons.get('secondary_buttons', []) if btn.get('label') == '🔍 Cek Data'), None),
        'cleanup_button': next((btn for btn in action_buttons.get('secondary_buttons', []) if btn.get('label') == '🗑️ Bersihkan Hasil'), None),
        'download_button': action_buttons.get('primary_button')  # For backward compatibility
    })
    
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(ui_components)
    
    return ui_components
    return ui_components
