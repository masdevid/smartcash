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
    from smartcash.ui.components.action_buttons import create_action_buttons, create_visualization_buttons
    from smartcash.ui.components.progress_tracking import create_progress_tracking
    from smartcash.ui.components.status_panel import create_status_panel
    from smartcash.ui.components.log_accordion import create_log_accordion

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

    # Split selector menggunakan shared component
    split_selector = create_split_selector(
        selected_split='train',
        description="Target Split:",
        width='100%',
        icon='split'
    )

    # Buat tombol-tombol augmentasi dengan shared component
    action_buttons = create_action_buttons(
        primary_label="Run Augmentation",
        primary_icon="random",
        cleanup_enabled=True
    )

    # Tombol-tombol visualisasi dengan shared component
    visualization_buttons = create_visualization_buttons(include_distribution=True)

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

    # Summary stats container dengan styling yang konsisten
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd', 
            padding='10px', 
            margin='10px 0', 
            display='none'
        )
    )

    # Help panel dengan komponen info_box standar
    help_panel = get_augmentation_info()

    # Rakit komponen UI dengan layout yang lebih compact
    # Gunakan tab opsi dasar, target split, dan jenis augmentasi dengan checkbox opsi di bawah jenis augmentasi
    from smartcash.ui.dataset.augmentation.components.augmentation_options import create_combined_options
    combined_options = create_combined_options(config)

    settings_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Pengaturan Augmentasi</h4>"),
        combined_options,
        split_selector,
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
        summary_container,
        visualization_buttons['container'],
        widgets.Output(layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0', display='none')),
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
        'augment_button': action_buttons['primary_button'],
        'augmentation_button': action_buttons['primary_button'],  # Alias untuk kompatibilitas
        'stop_button': action_buttons['stop_button'],
        'reset_button': action_buttons['reset_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': action_buttons['save_button'],
        'button_container': action_buttons['container'],
        'summary_container': summary_container,
        'visualization_buttons': visualization_buttons['container'],
        'visualize_button': visualization_buttons['visualize_button'],
        'compare_button': visualization_buttons['compare_button'],
        'distribution_button': visualization_buttons.get('distribution_button'),
        'visualization_container': widgets.Output(layout=widgets.Layout(border='1px solid #ddd', padding='10px', margin='10px 0', display='none')),
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
