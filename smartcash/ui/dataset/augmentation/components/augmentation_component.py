"""
File: smartcash/ui/dataset/augmentation/components/augmentation_component.py
Deskripsi: Komponen utama UI untuk augmentasi dataset
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
    # Import utilitas UI standar
    from smartcash.ui.utils.header_utils import create_header
    from smartcash.ui.utils.constants import COLORS, ICONS
    from smartcash.ui.utils.layout_utils import create_divider
    
    # Buat komponen-komponen UI
    from smartcash.ui.dataset.augmentation.components.augmentation_options import create_augmentation_options
    from smartcash.ui.dataset.augmentation.components.action_buttons import create_action_buttons
    from smartcash.ui.dataset.augmentation.components.progress_component import create_progress_component
    from smartcash.ui.dataset.augmentation.components.output_component import create_output_component
    from smartcash.ui.dataset.augmentation.components.info_component import create_info_component
    
    # Header dengan styling konsisten
    header = create_header(
        f"{ICONS['augmentation']} Dataset Augmentation", 
        "Menambah variasi data dari hasil preprocessing untuk balancing distribusi kelas"
    )
    
    # Status panel untuk menampilkan status konfigurasi
    from smartcash.ui.components.status_panel import create_status_panel
    status_panel = create_status_panel(
        "Konfigurasi augmentasi dataset dari data preprocessed", 
        "info"
    )
    
    # Buat komponen-komponen UI
    aug_options = create_augmentation_options(config)
    button_components = create_action_buttons()
    progress_components = create_progress_component()
    output_components = create_output_component()
    info_component = create_info_component()
    
    # Tentukan default path
    preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
    augmented_dir = config.get('augmentation', {}).get('output_dir', 'data/augmented')
    
    # Container utama
    ui = widgets.VBox([
        header,
        status_panel,
        widgets.HTML(f"<h4 style='color: {COLORS['dark']}; margin-top: 15px; margin-bottom: 10px;'>{ICONS['settings']} Augmentation Settings</h4>"),
        aug_options,
        create_divider(),
        button_components['container'],
        progress_components['container'],
        output_components['log_accordion'],
        output_components['summary_container'],
        button_components['visualization_container'],
        output_components['visualization_container'],
        info_component
    ], layout=widgets.Layout(width='100%', padding='10px'))
    
    # Gabungkan semua komponen dalam dictionary
    ui_components = {
        'ui': ui,
        'header': header,
        'status_panel': status_panel,
        'aug_options': aug_options,
        'module_name': 'augmentation',
        'preprocessed_dir': preprocessed_dir,
        'augmented_dir': augmented_dir,
        'augmentation_running': False,
    }
    
    # Tambahkan semua komponen dari sub-component
    ui_components.update(button_components)
    ui_components.update(progress_components)
    ui_components.update(output_components)
    
    return ui_components