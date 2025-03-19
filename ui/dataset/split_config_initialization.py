"""
File: smartcash/ui/dataset/split_config_initialization.py
Deskripsi: Inisialisasi komponen UI konfigurasi split dataset
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.alerts import create_status_indicator
from IPython.display import display
from smartcash.ui.dataset.split_config_utils import (
    load_dataset_config,
    update_ui_from_config
)
from smartcash.ui.dataset.split_config_visualization import (
    get_dataset_stats,
    get_class_distribution,
    update_stats_cards,
    show_class_distribution_visualization
)

def initialize_ui(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> None:
    """
    Initialize UI dengan data dari konfigurasi.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
    """
    try:
        # Load dataset config dan update config
        dataset_config = load_dataset_config()
        
        # Update config dengan dataset_config
        if 'data' in dataset_config:
            for key, value in dataset_config['data'].items():
                config['data'][key] = value
        
        # Update UI dari config
        update_ui_from_config(ui_components, config)
        
        # Update stats cards
        stats = get_dataset_stats(config, env, logger)
        update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
        
        # Show class distribution
        show_class_distribution_visualization(
            ui_components['output_box'],
            get_class_distribution(config, logger),
            COLORS
        )
        
        if logger: logger.info(f"✅ UI konfigurasi split berhasil diinisialisasi")
            
    except Exception as e:
        if logger: logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        
        # Tampilkan pesan error sederhana di output box
        with ui_components.get('output_box', None):
            display(create_status_indicator("error", f"{ICONS['error']} Error inisialisasi konfigurasi split: {str(e)}"))