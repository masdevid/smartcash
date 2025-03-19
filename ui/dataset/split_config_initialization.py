"""
File: smartcash/ui/dataset/split_config_initialization.py
Deskripsi: Inisialisasi komponen UI konfigurasi split dataset dengan dukungan Google Drive
"""

from typing import Dict, Any
from smartcash.ui.utils.constants import COLORS, ICONS
from smartcash.ui.components.alerts import create_status_indicator
from IPython.display import display

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
        # Load dataset config dan update UI
        from smartcash.ui.dataset.split_config_handlers import load_dataset_config, update_ui_from_config
        dataset_config = load_dataset_config()
        
        # Update config dengan dataset_config
        if 'data' in dataset_config:
            for key, value in dataset_config['data'].items():
                if 'data' not in config:
                    config['data'] = {}
                config['data'][key] = value
        
        # Update UI dari config
        update_ui_from_config(ui_components, config)
        
        # Update status panel berdasarkan drive
        use_drive = config.get('data', {}).get('use_drive', False)
        drive_mounted = False
        if 'drive_options' in ui_components:
            drive_mounted = not ui_components['drive_options'].children[0].disabled
        
        if use_drive and drive_mounted:
            drive_path = config.get('data', {}).get('drive_path', '')
            local_path = config.get('data', {}).get('local_clone_path', 'data_local')
            
            ui_components['status_panel'].value = f"""
            <div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                        color:{COLORS['alert_info_text']}; margin:10px 0; border-radius:4px; 
                        border-left:4px solid {COLORS['alert_info_text']};">
                <p style="margin:5px 0">{ICONS['info']} Menggunakan dataset dari Google Drive: <strong>{drive_path}</strong></p>
                <p style="margin:5px 0">Clone lokal: <strong>{local_path}</strong></p>
            </div>
            """
        
        # Update stats cards
        from smartcash.ui.dataset.split_config_visualization import (
            get_dataset_stats,
            update_stats_cards,
            get_class_distribution,
            show_class_distribution_visualization
        )
        
        stats = get_dataset_stats(config, env, logger)
        update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
        
        # Show class distribution dalam output box
        show_class_distribution_visualization(
            ui_components['output_box'],
            get_class_distribution(config, env, logger),
            COLORS,
            logger
        )
        
        if logger: logger.info(f"✅ UI konfigurasi split berhasil diinisialisasi")
            
    except Exception as e:
        if logger: logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        
        # Tampilkan pesan error sederhana di output box
        with ui_components.get('output_box', None):
            display(create_status_indicator("error", f"{ICONS['error']} Error inisialisasi konfigurasi split: {str(e)}"))