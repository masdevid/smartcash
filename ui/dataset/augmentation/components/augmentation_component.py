"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: Komponen UI utama untuk augmentasi dataset
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
    from smartcash.ui.utils.layout_utils import OUTPUT_WIDGET
    from smartcash.ui.handlers.status_handler import create_status_panel
    
    # Import komponen submodules augmentasi
    from smartcash.ui.dataset.augmentation.components.augmentation_options import create_augmentation_options
    from smartcash.ui.dataset.augmentation.components.split_selector import create_split_selector
    from smartcash.ui.dataset.augmentation.components.advanced_options import create_advanced_options
    
    # Import komponen button khusus untuk augmentasi
    from smartcash.ui.dataset.augmentation.components.action_buttons import create_action_buttons
    from smartcash.ui.components.visualization_buttons import create_visualization_buttons
    
    # Header dengan komponen standar
    header = create_header(f"{ICONS['augmentation']} Dataset Augmentation", 
                          "Augmentasi dataset untuk meningkatkan performa model SmartCash")
    
    # Panel info status
    status_panel = create_status_panel("Konfigurasi augmentasi dataset", "info")
    
    # Augmentation options (split dari komponen besar)
    augmentation_options = create_augmentation_options(config)
    split_selector = create_split_selector()
    
    # Advanced options dalam accordion
    advanced_options = create_advanced_options(config)
    
    # Accordion untuk advanced options - selalu tertutup di awal
    advanced_accordion = widgets.Accordion(children=[advanced_options], selected_index=None)
    advanced_accordion.set_title(0, f"{ICONS['settings']} Advanced Options")
    
    # Buat tombol-tombol augmentasi dengan komponen khusus
    action_buttons = create_action_buttons()
    
    # Tombol-tombol visualisasi dengan komponen standar
    visualization_buttons = create_visualization_buttons(module_name="augmentation")
    
    # Progress tracking dengan styling standar
    progress_bar = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Total:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    current_progress = widgets.IntProgress(
        value=0, min=0, max=100, 
        description='Kelas:',
        bar_style='info', 
        orientation='horizontal',
        layout=widgets.Layout(visibility='hidden', width='100%')
    )
    
    # Labels untuk progress
    overall_label = widgets.HTML("", layout=widgets.Layout(margin='2px 0', visibility='hidden'))
    step_label = widgets.HTML("", layout=widgets.Layout(margin='2px 0', visibility='hidden'))
    
    # Container progress dengan styling yang lebih konsisten
    progress_container = widgets.VBox([
        widgets.HTML(f"<h4 style='color:{COLORS['dark']}'>{ICONS['stats']} Progress</h4>"), 
        progress_bar,
        overall_label,
        current_progress,
        step_label
    ])
    
    # Status output dengan layout standar
    status = widgets.Output(layout=OUTPUT_WIDGET)
    
    # Log accordion dengan styling standar - terbuka secara default (selected_index=0)
    log_accordion = widgets.Accordion(children=[status], selected_index=0)
    log_accordion.set_title(0, f"{ICONS['file']} Augmentation Logs")
    
    # Summary stats container dengan styling yang konsisten
    summary_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd', 
            padding='10px', 
            margin='10px 0', 
            display='none'
        )
    )
    
    # Container visualisasi
    visualization_container = widgets.Output(
        layout=widgets.Layout(
            border='1px solid #ddd', 
            padding='10px', 
            margin='10px 0', 
            display='none'
        )
    )
    
    # Help panel dengan komponen info_box standar
    help_panel = get_augmentation_info()
    
    # Layout UI dengan divider standar
    from smartcash.ui.utils.layout_utils import create_divider
    
    # Rakit komponen UI
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Augmentation Settings</h4>"),
        augmentation_options,
        split_selector,
        advanced_accordion,
        create_divider(),
        action_buttons['container'],
        progress_container,
        log_accordion,
        summary_container,
        visualization_buttons['container'],
        visualization_container,
        help_panel
    ])
    
    # Komponen UI dengan konsolidasi semua referensi
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'augmentation_options': augmentation_options,
        'advanced_options': advanced_options,
        'split_selector': split_selector,
        'advanced_accordion': advanced_accordion,
        'augment_button': action_buttons['primary_button'],
        'stop_button': action_buttons['stop_button'],
        'reset_button': action_buttons['reset_button'],
        'cleanup_button': action_buttons['cleanup_button'],
        'save_button': action_buttons['save_button'],
        'button_container': action_buttons['container'],
        'progress_bar': progress_bar,
        'current_progress': current_progress,
        'overall_label': overall_label,
        'step_label': step_label,
        'status': status,
        'log_accordion': log_accordion,
        'summary_container': summary_container,
        'visualization_buttons': visualization_buttons['container'],
        'visualize_button': visualization_buttons['visualize_button'],
        'compare_button': visualization_buttons['compare_button'],
        'distribution_button': visualization_buttons['distribution_button'],
        'visualization_container': visualization_container,
        'module_name': 'augmentation',
        # Default dataset paths
        'data_dir': 'data',
        'augmented_dir': 'data/augmented'
    }
    
    return ui_components
