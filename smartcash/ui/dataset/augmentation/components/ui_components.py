"""
File: smartcash/ui/dataset/augmentation/components/ui_components.py
Deskripsi: Main UI assembly dengan reuse komponen dan progress tracker baru
"""

import ipywidgets as widgets
from typing import Dict, Any

def create_augmentation_main_ui(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Main UI assembly dengan CommonInitializer pattern dan progress tracker baru
    
    Args:
        config: Konfigurasi untuk initialize UI values
        
    Returns:
        Dictionary berisi semua UI components
    """
    # Import reused components
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import create_divider
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    from smartcash.ui.components.progress_tracker import create_triple_progress_tracker
    
    # Import widget components
    from smartcash.ui.dataset.augmentation.components.basic_opts_widget import create_basic_options_widget
    from smartcash.ui.dataset.augmentation.components.advanced_opts_widget import create_advanced_options_widget
    from smartcash.ui.dataset.augmentation.components.augtypes_opts_widget import create_augmentation_types_widget
    
    # Header dengan module context
    header = create_header(
        f"{ICONS.get('augmentation', '🔄')} Dataset Augmentation", 
        "Augmentasi dataset dengan progress tracking dan service integration"
    )
    
    # Status panel initial
    status_panel = create_status_panel("✅ Augmentation UI siap digunakan", "success")
    
    # Widget groups
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()
    augmentation_types = create_augmentation_types_widget()
    
    # Progress tracking dengan triple level dan step mapping dari config
    progress_config = config.get('progress', {}) if config else {}
    operations_config = progress_config.get('operations', {})
    
    # Get augmentation operation config dengan fallback
    aug_operation = operations_config.get('augmentation', {
        'steps': ["prepare", "augment", "normalize", "verify"],
        'weights': {"prepare": 10, "augment": 50, "normalize": 30, "verify": 10},
        'auto_hide': True
    })
    
    progress_tracker = create_triple_progress_tracker(
        operation="Dataset Augmentation",
        steps=aug_operation.get('steps', ["prepare", "augment", "normalize", "verify"]),
        step_weights=aug_operation.get('weights', {"prepare": 10, "augment": 50, "normalize": 30, "verify": 10}),
        auto_hide=aug_operation.get('auto_hide', True)
    )
    
    # Control buttons dengan enhanced tooltips
    config_buttons = create_save_reset_buttons(
        save_label="Simpan Config", 
        reset_label="Reset Config",
        save_tooltip="Simpan konfigurasi augmentation dengan inheritance",
        reset_tooltip="Reset ke konfigurasi default dari base config",
        with_sync_info=True,
        sync_message="Konfigurasi akan tersimpan di configs/augmentation_config.yaml"
    )
    
    # Action buttons dengan enhanced operations
    action_buttons = create_action_buttons(
        primary_label="🎯 Run Augmentation Pipeline", 
        primary_icon="play",
        secondary_buttons=[
            ("🔍 Check Dataset Status", "search", "info"),
            ("🧹 Cleanup Augmentation Data", "trash", "warning")
        ],
        cleanup_enabled=True, 
        button_width="200px"
    )
    
    # Output areas
    confirmation_area = widgets.Output(layout=widgets.Layout(
        width='100%', margin='8px 0', max_height='200px', overflow='auto'
    ))
    
    # Log dan info
    log_components = create_log_accordion('augmentation', '200px')
    
    # Help info panel dengan augmentation guidance
    help_panel = _create_augmentation_help_panel()
    
    # Layout assembly dengan responsive design
    settings_row1 = widgets.HBox([
        _create_settings_section("Opsi Dasar", basic_options['container']),
        _create_settings_section("Opsi Lanjutan", advanced_options['container'])
    ], layout=widgets.Layout(width='100%', justify_content='flex-start'))
    
    settings_row2 = _create_settings_section(None, augmentation_types['container'], full_width=True)
    
    # Config controls
    config_container = widgets.Box([config_buttons['container']], 
        layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%', margin='8px 0'))
    
    # Settings container
    settings_container = widgets.VBox([
        settings_row1, 
        settings_row2, 
        config_container
    ], layout=widgets.Layout(width='100%', overflow='hidden'))
    
    # Action header
    action_header = widgets.HTML(f"""
    <h4 style='color: {COLORS.get('dark', '#333')}; margin: 15px 0 10px 0; font-size: 16px; 
               border-bottom: 2px solid {COLORS.get('success', '#28a745')}; padding-bottom: 6px;'>
        {ICONS.get('play', '▶️')} Actions
    </h4>
    """)
    
    # Main UI assembly
    ui = widgets.VBox([
        header,
        status_panel,
        settings_container,
        action_header,
        action_buttons['container'],
        confirmation_area,
        progress_tracker['container'],
        log_components['log_accordion'],
        help_panel
    ], layout=widgets.Layout(width='100%', overflow='hidden'))
    
    # Comprehensive component mapping untuk handlers
    return {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'settings_container': settings_container,
        'confirmation_area': confirmation_area,
        'help_panel': help_panel,
        
        # Widget mappings dari form components
        **basic_options['widgets'],
        **advanced_options['widgets'], 
        **augmentation_types['widgets'],
        
        # Button mappings
        'augment_button': action_buttons['download_button'],  # Primary action
        'check_button': action_buttons['check_button'],       # Secondary action
        'cleanup_button': action_buttons.get('cleanup_button'), # Cleanup action
        'save_button': config_buttons['save_button'],
        'reset_button': config_buttons['reset_button'],
        
        # Progress tracking dengan new API
        'progress_tracker': progress_tracker['tracker'],
        'progress_container': progress_tracker['container'],
        'show_for_operation': progress_tracker['show_container'],
        'update_overall': progress_tracker['update_overall'],
        'update_step': progress_tracker['update_step'],
        'update_current': progress_tracker['update_current'],
        'update_progress': progress_tracker['update_progress'],
        'complete_operation': progress_tracker['complete_operation'],
        'error_operation': progress_tracker['error_operation'],
        'reset_all': progress_tracker['reset_all'],
        
        # Log outputs
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        'status': log_components['log_output'],  # Compatibility
        
        # Metadata untuk initializer
        'module_name': 'augmentation',
        'logger_namespace': 'smartcash.ui.dataset.augmentation',
        'config': config or {}
    }

def _create_settings_section(title: str, content_widget, full_width: bool = False):
    """Create settings section dengan consistent styling"""
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    children = []
    if title:
        children.append(widgets.HTML(
            f"<h5 style='color: {COLORS.get('dark', '#333')}; margin: 5px 0; font-size: 14px;'>"
            f"{ICONS.get('settings', '⚙️')} {title}</h5>"
        ))
    children.append(content_widget)
    
    width = '100%' if full_width else '48%'
    margin = '8px 0' if full_width else '0 1% 0 0'
    
    return widgets.VBox(children, layout=widgets.Layout(
        width=width, padding='8px', margin=margin,
        border='1px solid #e0e0e0', border_radius='4px',
        background_color='rgba(248, 249, 250, 0.8)'
    ))

def _create_augmentation_help_panel() -> widgets.Widget:
    """Create help panel dengan augmentation guidance"""
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    help_content = f"""
    <div style="padding: 12px; background-color: {COLORS.get('bg_light', '#f8f9fa')}; 
                border-radius: 6px; margin: 10px 0;">
        <h6 style="color: {COLORS.get('dark', '#333')}; margin: 0 0 8px 0;">
            {ICONS.get('help', '💡')} Panduan Augmentasi Dataset
        </h6>
        <ul style="margin: 5px 0; padding-left: 20px; font-size: 12px; line-height: 1.4;">
            <li><strong>Combined Augmentation:</strong> Gabungan transformasi posisi dan pencahayaan (direkomendasikan)</li>
            <li><strong>Target Count:</strong> 500-1000 file per kelas untuk training yang optimal</li>
            <li><strong>Balance Classes:</strong> Aktifkan untuk menyeimbangkan Layer 1 & 2 (denominasi utama)</li>
            <li><strong>Parameter Moderat:</strong> Gunakan nilai default untuk hasil yang stabil</li>
            <li><strong>Progress Tracking:</strong> Monitor 4 tahap: Prepare → Augment → Normalize → Verify</li>
        </ul>
    </div>
    """
    
    help_accordion = widgets.Accordion([widgets.HTML(help_content)])
    help_accordion.set_title(0, f"{ICONS.get('help', '💡')} Panduan Augmentasi")
    help_accordion.selected_index = None  # Collapsed by default
    
    return help_accordion