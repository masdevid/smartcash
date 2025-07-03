"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Updated UI components dengan shared container components dan live preview integration
"""

from IPython.display import display, HTML
import ipywidgets as widgets
from typing import Dict, Any, Optional

# Import standard container components
from smartcash.ui.components.main_container import create_main_container
from smartcash.ui.components.header_container import create_header_container
from smartcash.ui.components.form_container import create_form_container
from smartcash.ui.components.footer_container import create_footer_container
from smartcash.ui.components.action_container import create_action_container

# Internal components
from smartcash.ui import components
from smartcash.ui.dataset.augmentation.utils.style_utils import (
    styled_container, flex_layout
)
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components import (
    create_log_accordion,
    create_save_reset_buttons
)
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressLevel

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
    """🎨 Create augmentation UI using shared container components while preserving unique form structure"""
    config = config or {}
    
    # Initialize UI components dictionary
    ui_components = {}
    
    # === CORE COMPONENTS ===
    
    # 1. Create Header Container
    header_container = create_header_container(
        title="Dataset Augmentation",
        subtitle="Pipeline augmentasi dengan live preview dan backend integration",
        icon="🔄"
    )
    ui_components['header_container'] = header_container.container
    
    # 2. Create Form Container (will be populated with custom form layout later)
    form_container = create_form_container()
    
    # 3. Create Footer Container with Log Accordion
    log_components = create_log_accordion('augmentation', '250px')
    footer_container = create_footer_container(
        log_output=log_components['log_output'],
        info_box=widgets.HTML(
            """
            <div class="alert alert-info" style="font-size: 0.9em; padding: 8px 12px;">
                <strong>Tips Augmentasi:</strong>
                <ul style="margin: 5px 0 0 15px; padding: 0;">
                    <li>Gunakan augmentasi yang sesuai dengan domain data</li>
                    <li>Pastikan preview menunjukkan hasil yang diharapkan</li>
                    <li>Cek hasil augmentasi sebelum menggunakannya untuk training</li>
                </ul>
            </div>
            """
        )
    )
    ui_components['footer_container'] = footer_container.container
    
    # 4. Create Action Container with standard approach
    action_container = create_action_container(
        buttons=[
            {
                "button_id": "augment",
                "text": "🚀 Jalankan Augmentasi",
                "style": "primary",
                "order": 1
            },
            {
                "button_id": "check",
                "text": "🔍 Cek Data",
                "style": "info",
                "order": 2
            },
            {
                "button_id": "cleanup",
                "text": "🗑️ Bersihkan Hasil",
                "style": "warning",
                "tooltip": "Hapus hasil augmentasi sebelumnya",
                "order": 3
            }
        ],
        title="🚀 Augmentation Operations",
        alignment="left"
    )
    ui_components['action_container'] = action_container.container
    
    # 5. Create Progress Tracker
    progress_tracker = ProgressTracker(
        title="Augmentation Pipeline",
        levels=[ProgressLevel.OVERALL, ProgressLevel.CURRENT],
        auto_hide=True
    )
    ui_components['progress_tracker'] = progress_tracker
    
    # 6. Create Save/Reset buttons for configuration
    config_buttons = create_save_reset_buttons(
        save_label="Simpan", 
        reset_label="Reset",
        with_sync_info=True, 
        sync_message="Konfigurasi disinkronkan dengan backend"
    )
    
    # === BUTTON MAPPING ===
    
    # Extract buttons from action container using standard approach
    augment_button = action_container.get_button('augment')
    check_button = action_container.get_button('check')
    cleanup_button = action_container.get_button('cleanup')
    
    # === CUSTOM FORM LAYOUT ===
    # Preserve the unique form structure of the augmentation module
    
    # Widget groups dengan live preview integration
    basic_options = _create_basic_options_group()
    advanced_options = _create_advanced_options_group()
    augmentation_types = _create_augmentation_types_group()
    live_preview = _create_live_preview_group()  # Live preview component
    
    # Import style utilities
    from smartcash.ui.dataset.augmentation.utils.style_utils import styled_container
    
    # 2x2 Grid with original styling and gradients - PRESERVED CUSTOM LAYOUT
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
    
    # Place custom form layout in the form container
    form_container['form_container'].children = (widgets.VBox([row1, row2]),)
    ui_components['form_container'] = form_container['container']
    
    # Config section with consistent styling
    config_section = widgets.VBox([
        widgets.Box([config_buttons['container']], 
            layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%'))
    ], layout=widgets.Layout(margin='8px 0'))
    
    # Already created the 2x2 grid with original styling and gradients above
    
    # === MAIN UI ASSEMBLY ===
    
    # Assemble Main Container with shared components
    main_container = create_main_container(
        header_container=header_container.container,
        form_container=ui_components['form_container'],
        footer_container=footer_container.container,
        action_container=action_container.container
    )
    
    # Add config section to the main container (preserve unique augmentation feature)
    main_container.children = list(main_container.children[:-1]) + [config_section] + [main_container.children[-1]]
    
    # Add progress tracker to the main container
    if hasattr(progress_tracker, 'container'):
        main_container.children = list(main_container.children) + [progress_tracker.container]
    
    # Store the main UI in ui_components
    ui_components['ui'] = main_container
    
    # Update ui_components with all components
    ui_components.update({
        # Core UI components
        **basic_options.get('widgets', {}), 
        **advanced_options.get('widgets', {}),
        **augmentation_types.get('widgets', {}), 
        **live_preview.get('widgets', {}),  # Live preview widgets
        
        # Buttons from action container
        'augment_button': augment_button,
        'check_button': check_button,
        'cleanup_button': cleanup_button,
        'download_button': augment_button,  # For backward compatibility
        
        # Config buttons
        'save_button': config_buttons.get('save_button'),
        'reset_button': config_buttons.get('reset_button'),
        
        # Progress and logging
        'progress_tracker': progress_tracker,
        'log_accordion': log_components.get('log_accordion'),
        'log_output': log_components.get('log_output'),
        
        # Metadata
        'module_name': 'augmentation',
        'logger_namespace': 'smartcash.ui.dataset.augmentation',
        'augmentation_initialized': True,
        'config': config or {}
    })
    
    # Log any missing components for debugging
    from smartcash.ui.utils.logging_utils import log_missing_components
    log_missing_components(ui_components)
    
    return ui_components
