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
    

    # Header dan status panel
    header = create_header(
        f"{ICONS.get('augmentation', '🔄')} Dataset Augmentation", 
        "Pipeline augmentasi dengan live preview dan backend integration"
    )
    status_panel = create_status_panel("✅ Pipeline augmentasi siap", "success")
    
    # Widget groups dengan live preview integration
    basic_options = _create_basic_options_group()
    advanced_options = _create_advanced_options_group()
    augmentation_types = _create_augmentation_types_group()
    live_preview = _create_live_preview_group()  # CHANGED: Menggantikan normalization
    
    # Progress tracker dan buttons
    progress_tracker = create_dual_progress_tracker("Augmentation Pipeline", auto_hide=True)
    config_buttons = create_save_reset_buttons(
        save_label="Simpan", reset_label="Reset",
        with_sync_info=True, sync_message="Konfigurasi disinkronkan dengan backend"
    )
    action_buttons = create_action_buttons(
        primary_label="🚀 Jalankan Augmentasi", primary_icon="play",
        secondary_buttons=[("🔍 Cek Data", "search", "info")],
        cleanup_enabled=True, 
        button_width="220px"
    )
    
    # Confirmation area untuk dialog integration
    confirmation_area, _ = create_confirmation_area()  # Unpack the tuple, we only need the widget
    
    # Log accordion
    log_components = create_log_accordion('augmentation', '250px')
    
    # 2x2 Grid dengan live preview
    row1 = widgets.HBox([
        styled_container(basic_options['container'], "📋 Opsi Dasar", 'basic', '47%'),
        styled_container(advanced_options['container'], "⚙️ Parameter Lanjutan", 'advanced', '47%')
    ], layout=widgets.Layout(
        width='100%', max_width='100%', display='flex',
        flex_flow='row wrap', justify_content='space-between',
        align_items='stretch', gap='6px', margin='8px 0',
        overflow='hidden', box_sizing='border-box'
    ))
    
    row2 = widgets.HBox([
        styled_container(augmentation_types['container'], "🔄 Jenis Augmentasi", 'types', '47%'),
        styled_container(live_preview['container'], "🎬 Live Preview", 'normalization', '47%')  # CHANGED
    ], layout=widgets.Layout(
        width='100%', max_width='100%', display='flex',
        flex_flow='row wrap', justify_content='space-between',
        align_items='stretch', gap='6px', margin='8px 0',
        overflow='hidden', box_sizing='border-box'
    ))
    
    # Action section dengan confirmation area
    action_section = widgets.VBox([
        _create_section_header("🚀 Pipeline Operations", "#667eea"),
        action_buttons['container'],
        widgets.HTML("<div style='margin: 5px 0;'><strong>📋 Status & Konfirmasi:</strong></div>"),
        confirmation_area
    ], layout=widgets.Layout(
        width='100%',
        margin='10px 0',
        padding='10px',
        border='1px solid #e0e0e0',
        border_radius='8px',
        background_color='#f9f9f9'
    ))
    
    # Config section
    config_section = widgets.VBox([
        widgets.Box([config_buttons['container']], 
            layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
    ])
    
    # Main UI assembly dengan urutan yang benar
    ui = widgets.VBox([
        header, 
        status_panel, 
        row1, 
        row2, 
        config_section,
        action_section,
        progress_tracker.container,
        log_components['log_accordion']
    ], layout=widgets.Layout(
        width='100%', max_width='100%', display='flex',
        flex_flow='column', align_items='stretch'
    ))
    
    # Component mapping dengan live preview integration
    components = {
        'ui': ui, 
        'header': header, 
        'status_panel': status_panel,
        'confirmation_area': confirmation_area,
        **basic_options['widgets'], 
        **advanced_options['widgets'],
        **augmentation_types['widgets'], 
        **live_preview['widgets'],  # CHANGED: Live preview widgets
        'augment_button': action_buttons['download_button'],
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        'save_button': config_buttons['save_button'],
        'reset_button': config_buttons['reset_button'],
        'progress_tracker': progress_tracker,
        'log_output': log_components['log_output'],
        'status': log_components['log_output'],
        'backend_ready': True, 
        'service_integration': True,
        'module_name': 'augmentation',
        'logger_namespace': 'smartcash.ui.dataset.augmentation',
        'augmentation_initialized': True,
        'config': config or {}
    }

    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(components)
    return components


