"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: Komponen UI utama untuk augmentasi dataset dengan pemisahan logika bisnis yang bersih
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk augmentasi dataset tanpa logika bisnis.

    Args:
        env: Environment manager
        config: Konfigurasi aplikasi

    Returns:
        Dictionary berisi widget UI
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

    # Import komponen UI pure (tanpa logika bisnis)
    from smartcash.ui.dataset.augmentation.components.basic_options_widget import create_basic_options_widget
    from smartcash.ui.dataset.augmentation.components.advanced_options_widget import create_advanced_options_widget
    from smartcash.ui.dataset.augmentation.components.augmentation_types_widget import create_augmentation_types_widget

    # Header dengan komponen standar
    header = create_header(f"{ICONS['augmentation']} Dataset Augmentation", 
                          "Augmentasi dataset untuk meningkatkan performa model SmartCash")

    # Panel info status
    status_panel = create_status_panel("Konfigurasi augmentasi dataset", "info")

    # Buat widget UI pure (hanya tampilan, tanpa logika bisnis)
    basic_options_widget = create_basic_options_widget()
    advanced_options_widget = create_advanced_options_widget()
    augmentation_types_widget = create_augmentation_types_widget()

    # Buat tombol save dan reset menggunakan shared component
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi augmentasi dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi augmentasi ke default",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
        button_width="100px"
    )

    # Buat tombol-tombol augmentasi dengan shared component
    action_buttons = create_action_buttons(
        primary_label="Run Augmentation",
        primary_icon="random",
        cleanup_enabled=True
    )

    # Progress tracking dengan shared component
    progress_components = create_progress_tracking_container()

    # Log accordion dengan shared component
    log_components = create_log_accordion(
        module_name='augmentation',
        height='200px',
        width='100%'
    )

    # Help panel dengan komponen info_box standar
    help_panel = get_augmentation_info()

    # Layout dengan pemisahan yang jelas antara UI dan logika
    row1 = widgets.HBox([
        # Kolom 1: Basic Options
        widgets.VBox([
            widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 5px; margin-bottom: 5px;'>{ICONS['settings']} Opsi Dasar</h4>"),
            basic_options_widget['container']
        ], layout=widgets.Layout(width='48%', padding='5px', border='1px solid #eaeaea', border_radius='5px')),
        
        # Kolom 2: Advanced Options
        widgets.VBox([
            widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 5px; margin-bottom: 5px;'>{ICONS['settings']} Opsi Lanjutan</h4>"),
            advanced_options_widget['container']
        ], layout=widgets.Layout(width='48%', padding='5px', border='1px solid #eaeaea', border_radius='5px'))
    ], layout=widgets.Layout(width='100%', justify_content='space-between'))
    
    # Baris 2: Jenis Augmentasi & Split
    row2 = widgets.VBox([
        augmentation_types_widget['container']
    ], layout=widgets.Layout(width='100%', padding='5px', margin='10px 0', border='1px solid #eaeaea', border_radius='5px'))
    
    # Container untuk tombol save/reset
    button_container = widgets.Box(
        [save_reset_buttons['container']], 
        layout=widgets.Layout(display='flex', justify_content='flex-end', width='100%', margin='10px 0 5px 0')
    )
    
    # Settings container
    settings_container = widgets.VBox([
        row1,
        row2,
        button_container
    ], layout=widgets.Layout(width='100%', padding='0'))

    # Main UI assembly
    ui = widgets.VBox([
        header,
        status_panel,
        settings_container,
        create_divider(),
        action_buttons['container'],
        progress_components['progress_container'],
        log_components['log_accordion'],
        help_panel
    ], layout=widgets.Layout(width='100%', padding='0', margin='0'))

    # UI Components mapping (hanya untuk referensi widget, tanpa logika bisnis)
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        
        # Widget containers
        'basic_options_container': basic_options_widget['container'],
        'advanced_options_container': advanced_options_widget['container'],
        'augmentation_types_container': augmentation_types_widget['container'],
        
        # Individual widgets untuk parameter extraction
        **basic_options_widget['widgets'],
        **advanced_options_widget['widgets'],
        **augmentation_types_widget['widgets'],
        
        # Action buttons - Ensure handler compatibility
        'augment_button': action_buttons.get('primary_button') or action_buttons.get('main_button') or action_buttons.get('action_button'),
        'stop_button': action_buttons.get('stop_button'),
        'cleanup_button': action_buttons.get('cleanup_button'),
        
        # Handler expects these specific keys for state management
        'running_state': False,  # Handler uses this for tracking running state
        'stop_signal': False,    # Handler uses this for stop signal
        
        # Config buttons
        'save_button': save_reset_buttons['save_button'],
        'reset_button': save_reset_buttons['reset_button'],
        
        # Progress components - Handler expects these keys
        'progress_bar': progress_components.get('progress_bar'),
        'progress_container': progress_components.get('progress_container'),
        'current_progress': progress_components.get('current_progress'),
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label'),
        
        # Handler expects these progress-related keys
        'progress_callback': None,  # Handler sets this during execution
        
        # Log components - Handler expects these keys
        'status': log_components.get('log_output'),
        'log_output': log_components.get('log_output'),
        'log_accordion': log_components.get('log_accordion'),
        
        # Module info (untuk managers) - Handler expects these paths
        'module_name': 'augmentation',
        'data_dir': 'data',
        'augmented_dir': 'data/augmented',
        
        # Handler compatibility - Additional keys the handler might expect
        'env': env,  # Environment manager
        'config': config,  # Configuration
        
        # Add reference to full components for debugging and handler access
        'action_buttons': action_buttons,
        'save_reset_buttons': save_reset_buttons,
        'progress_components': progress_components,
        'log_components': log_components
    }

    return ui_components