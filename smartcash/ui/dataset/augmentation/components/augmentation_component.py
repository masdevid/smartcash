"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: Komponen UI utama untuk augmentasi dataset menggunakan shared components
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def create_augmentation_ui(env=None, config=None) -> Dict[str, Any]:
    """
    Buat komponen UI untuk augmentasi dataset.

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
    from smartcash.ui.components.split_selector import create_split_selector
    from smartcash.ui.components.action_buttons import create_action_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion
    from smartcash.ui.components.validation_options import create_validation_options
    from smartcash.ui.components.save_reset_buttons import create_save_reset_buttons
    from smartcash.ui.components.sync_info_message import create_sync_info_message
    # Tidak menggunakan split_config lagi sesuai permintaan

    # Import komponen submodules augmentasi
    from smartcash.ui.dataset.augmentation.components.augmentation_options import create_augmentation_options
    from smartcash.ui.dataset.augmentation.components.advanced_options import create_advanced_options

    # Header dengan komponen standar
    header = create_header(f"{ICONS['augmentation']} Dataset Augmentation", 
                          "Augmentasi dataset untuk meningkatkan performa model SmartCash")

    # Panel info status
    status_panel = create_status_panel("Konfigurasi augmentasi dataset", "info")

    # Augmentation options
    augmentation_options = create_augmentation_options(config)

    # Advanced options dalam accordion
    advanced_options = create_advanced_options(config)

    # Accordion untuk advanced options - selalu tertutup di awal
    advanced_accordion = widgets.Accordion(children=[advanced_options], selected_index=None)
    advanced_accordion.set_title(0, f"{ICONS['settings']} Advanced Options")

    # Tombol save dan reset dan sync info akan dibuat nanti untuk menghindari duplikasi
    
    # Split selector tetap digunakan untuk kompatibilitas
    split_selector = create_split_selector(
        selected_value='Train Only',
        description="Target Split:",
        width='100%',
        icon='split'
    )
    
    # Buat tombol save dan reset menggunakan shared component
    save_reset_buttons = create_save_reset_buttons(
        save_label="Simpan",
        reset_label="Reset",
        save_tooltip="Simpan konfigurasi augmentasi dan sinkronkan ke Google Drive",
        reset_tooltip="Reset konfigurasi augmentasi ke default",
        save_icon="save",
        reset_icon="reset",
        with_sync_info=True,
        sync_message="Konfigurasi akan otomatis disinkronkan dengan Google Drive saat disimpan atau direset.",
        button_width="100px"
    )
    
    # Gunakan sync_info dari save_reset_buttons
    sync_info = save_reset_buttons.get('sync_info', {})

    # Buat tombol-tombol augmentasi dengan shared component
    action_buttons = create_action_buttons(
        primary_label="Run Augmentation",
        primary_icon="random",
        cleanup_enabled=True
    )

    # Tidak lagi menggunakan tombol visualisasi lama

    # Progress tracking dengan shared component
    progress_components = create_progress_tracking(
        module_name='augmentation',
        show_step_progress=True,
        show_overall_progress=True,
        width='100%'
    )

    # Log accordion dengan shared component
    log_components = create_log_accordion(
        module_name='augmentation',
        height='200px',
        width='100%'
    )

    # Summary container dihapus sesuai permintaan untuk memisahkan fitur visualisasi ke cell lain

    # Help panel dengan komponen info_box standar
    help_panel = get_augmentation_info()

    # Rakit komponen UI dengan layout yang lebih compact
    # Gunakan tab opsi dasar, target split, dan jenis augmentasi dengan checkbox opsi di bawah jenis augmentasi
    from smartcash.ui.dataset.augmentation.components.augmentation_options import create_combined_options
    combined_options = create_combined_options(config)

    # Container untuk tombol save/reset dengan layout rata kanan
    # Sync info sudah termasuk dalam save_reset_buttons jika with_sync_info=True
    save_reset_container = widgets.VBox([
        save_reset_buttons['container']
    ], layout=widgets.Layout(align_items='flex-end', width='100%'))
    
    settings_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Pengaturan Augmentasi</h4>"),
        combined_options,
        save_reset_container,  # Gunakan container yang sudah diatur rata kanan
        split_selector,  # Tetap tampilkan untuk kompatibilitas
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Opsi Lanjutan</h4>"),
        advanced_options
    ], layout=widgets.Layout(width='100%'))

    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        settings_container,
        create_divider(),
        action_buttons['container'],
        progress_components['progress_container'],
        log_components['log_accordion'],
        help_panel
    ])

    # Komponen UI dengan konsolidasi semua referensi
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'augmentation_options': augmentation_options,
        'combined_options': combined_options,
        'advanced_options': advanced_options,
        'advanced_accordion': advanced_accordion,
        'split_selector': split_selector,
        'save_reset_buttons': save_reset_buttons,
        'sync_info': sync_info,
        'save_button': save_reset_buttons['save_button'],
        'reset_config_button': save_reset_buttons['reset_button'],
        'augment_button': action_buttons['primary_button'],
        'augmentation_button': action_buttons['primary_button'],  # Alias untuk kompatibilitas
        'stop_button': action_buttons['stop_button'],
        'reset_button': save_reset_buttons['reset_button'],  # Gunakan dari save_reset_buttons
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': save_reset_buttons['save_button'],  # Gunakan dari save_reset_buttons
        'save_reset_buttons': save_reset_buttons,  # Tambahkan referensi lengkap
        'button_container': action_buttons['container'],
        'module_name': 'augmentation',
        # Default dataset paths
        'data_dir': 'data',
        'augmented_dir': 'data/augmented'
    }

    # Tambahkan komponen progress tracking
    ui_components.update({
        'progress_bar': progress_components['progress_bar'],
        'progress_container': progress_components['progress_container'],
        'current_progress': progress_components.get('current_progress'),
        'overall_label': progress_components.get('overall_label'),
        'step_label': progress_components.get('step_label')
    })

    # Tambahkan komponen log
    ui_components.update({
        'status': log_components['log_output'],
        'log_output': log_components['log_output'],
        'log_accordion': log_components['log_accordion']
    })

    return ui_components
