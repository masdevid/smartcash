"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: Fixed komponen UI dengan proper communicator setup dan layout responsive yang tidak overflow
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Fixed UI creation dengan proper communicator integration dan responsive layout.

    Args:
        env: Environment manager
        config: Konfigurasi aplikasi

    Returns:
        Dictionary berisi widget UI dengan proper communicator setup
    """
    # Import komponen UI standar 
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS 
    from smartcash.ui.info_boxes.augmentation_info import get_augmentation_info
    from smartcash.ui.utils.layout_utils import create_divider

    # Import shared components
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking_container
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons

    # Import UI pure components
    from smartcash.ui.dataset.augmentation.components.basic_options_widget import create_basic_options_widget
    from smartcash.ui.dataset.augmentation.components.advanced_options_widget import create_advanced_options_widget
    from smartcash.ui.dataset.augmentation.components.augmentation_types_widget import create_augmentation_types_widget

    # Fixed header
    header = create_header(f"{ICONS['augmentation']} Dataset Augmentation", 
                          "Augmentasi dataset untuk meningkatkan performa model SmartCash")

    # Status panel dengan default message
    status_panel = create_status_panel("✅ Augmentation UI siap digunakan", "success")

    # Create UI pure widgets
    basic_options = create_basic_options_widget()
    advanced_options = create_advanced_options_widget()  
    augmentation_types = create_augmentation_types_widget()

    # Config buttons tanpa icon
    config_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi augmentasi",
        reset_tooltip="Reset ke konfigurasi default",
        with_sync_info=True,
        sync_message="Konfigurasi disinkronkan dengan Google Drive",
        button_width="110px",
        container_width="100%"
    )

    # Action buttons dengan proper naming
    action_buttons = create_action_buttons(
        primary_label="🚀 Run Augmentation",
        primary_icon="action", 
        secondary_buttons=[
            ("🔍 Check Dataset", "search", "info"),
            ("🧹 Cleanup Dataset", "cleanup", "warning")
        ],
        cleanup_enabled=True,
        button_width="160px"
    )

    # Fixed confirmation area positioning
    confirmation_area = widgets.Output(layout=widgets.Layout(
        width='100%', margin='8px 0', padding='0', overflow='hidden'
    ))

    # Progress tracking components
    progress_components = create_progress_tracking_container()

    # Log components 
    log_components = create_log_accordion('augmentation', '180px', '100%')

    # Help panel
    help_panel = get_augmentation_info()

    # Fixed responsive layout - no horizontal scroll
    settings_row1 = widgets.HBox([
        widgets.VBox([
            widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0; font-size: 14px;'>{ICONS['settings']} Opsi Dasar</h5>"),
            basic_options['container']
        ], layout=widgets.Layout(width='48%', padding='6px', margin='0 1% 0 0', 
                                border='1px solid #e0e0e0', border_radius='4px')),
        
        widgets.VBox([
            widgets.HTML(f"<h5 style='color: {COLORS['dark']}; margin: 5px 0; font-size: 14px;'>{ICONS['settings']} Opsi Lanjutan</h5>"),
            advanced_options['container']
        ], layout=widgets.Layout(width='48%', padding='6px', 
                                border='1px solid #e0e0e0', border_radius='4px'))
    ], layout=widgets.Layout(width='100%', max_width='100%', justify_content='flex-start'))

    # Fixed augmentation types row
    settings_row2 = widgets.VBox([
        augmentation_types['container']
    ], layout=widgets.Layout(width='100%', padding='6px', margin='8px 0', 
                            border='1px solid #e0e0e0', border_radius='4px'))

    # Config buttons container
    config_container = widgets.Box([config_buttons['container']], 
        layout=widgets.Layout(display='flex', justify_content='flex-end', 
                             width='100%', margin='8px 0'))

    # Main settings container
    settings_container = widgets.VBox([
        settings_row1, settings_row2, config_container
    ], layout=widgets.Layout(width='100%', max_width='100%', overflow='hidden'))

    # Fixed main UI assembly
    ui = widgets.VBox([
        header,
        status_panel, 
        settings_container,
        create_divider(),
        action_buttons['container'],
        confirmation_area,
        progress_components['container'],
        log_components['log_accordion'],
        help_panel
    ], layout=widgets.Layout(width='100%', max_width='100%', padding='0', 
                            margin='0', overflow='hidden'))

    # Fixed UI components mapping dengan proper communicator support
    ui_components = {
        'ui': ui,
        'header': header, 
        'status_panel': status_panel,
        
        # Core UI containers
        'settings_container': settings_container,
        'confirmation_area': confirmation_area,
        
        # Widget mappings untuk config extraction
        **basic_options['widgets'],
        **advanced_options['widgets'], 
        **augmentation_types['widgets'],
        
        # Fixed button mappings untuk handlers
        'augment_button': action_buttons['download_button'],  # Primary action
        'check_button': action_buttons['check_button'],
        'cleanup_button': action_buttons.get('cleanup_button'),
        'save_button': config_buttons['save_button'],
        'reset_button': config_buttons['reset_button'],
        
        # Progress tracking components
        **progress_components,
        
        # Log components
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion'],
        'status': log_components['log_output'],  # Backward compatibility
        
        # Module metadata untuk communicator
        'module_name': 'augmentation',
        'logger_namespace': 'smartcash.ui.dataset.augmentation',
        'data_dir': 'data',
        'augmented_dir': 'data/augmented',
        'preprocessed_dir': 'data/preprocessed',
        
        # Environment context
        'env': env,
        'config': config or {},
        
        # Communicator setup flags
        'communicator_ready': False,
        'ui_logger_ready': False,
        
        # Operation state
        'operation_running': False,
        'stop_requested': False,
        
        # Component references untuk debugging
        'component_refs': {
            'basic_options': basic_options,
            'advanced_options': advanced_options,
            'augmentation_types': augmentation_types,
            'action_buttons': action_buttons,
            'config_buttons': config_buttons,
            'progress_components': progress_components,
            'log_components': log_components
        }
    }

    return ui_components