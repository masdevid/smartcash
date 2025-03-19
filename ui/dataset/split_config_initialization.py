"""
File: smartcash/ui/dataset/split_config_initialization.py
Deskripsi: Inisialisasi komponen UI konfigurasi split dataset dengan penanganan error yang lebih baik
"""

from typing import Dict, Any, Optional
import os
from pathlib import Path
from IPython.display import display, HTML

def load_dataset_config(config_path: str = 'configs/dataset_config.yaml') -> Dict[str, Any]:
    """
    Load konfigurasi dataset dari file dengan fallback ke default.
    
    Args:
        config_path: Path ke file konfigurasi
        
    Returns:
        Dictionary berisi konfigurasi dataset
    """
    default_config = {
        'data': {
            'split_ratios': {'train': 0.7, 'valid': 0.15, 'test': 0.15},
            'stratified_split': True,
            'random_seed': 42,
            'backup_before_split': True,
            'backup_dir': 'data/splits_backup',
            'use_drive': False,
            'drive_path': '',
            'local_clone_path': 'data_local',
            'sync_on_change': True
        }
    }
    
    try:
        if os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
                
            # Update dengan config yang dimuat
            if 'data' in loaded_config:
                for key, value in loaded_config['data'].items():
                    if key == 'split_ratios' and isinstance(value, dict):
                        default_config['data']['split_ratios'].update(value)
                    else:
                        default_config['data'][key] = value
    except Exception as e:
        print(f"⚠️ Gagal memuat konfigurasi dari {config_path}: {str(e)}")
    
    return default_config

def initialize_ui(ui_components: Dict[str, Any], config: Dict[str, Any], env=None, logger=None) -> Dict[str, Any]:
    """
    Initialize UI dengan data dari konfigurasi dan validasi komponen.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        logger: Logger untuk logging
        
    Returns:
        Dictionary berisi komponen UI yang telah diupdate
    """
    try:
        # Load dataset config
        dataset_config = load_dataset_config()
        
        # Update config dengan dataset_config
        if 'data' in dataset_config:
            for key, value in dataset_config['data'].items():
                if 'data' not in config:
                    config['data'] = {}
                config['data'][key] = value
        
        # Validasi dan update UI dari config
        _update_ui_from_config(ui_components, config)
        
        # Update status panel berdasarkan drive
        use_drive = config.get('data', {}).get('use_drive', False)
        drive_mounted = False
        
        # Cek status drive_options
        if 'drive_options' in ui_components and hasattr(ui_components['drive_options'], 'children') and len(ui_components['drive_options'].children) > 0:
            # Periksa checkbox use_drive
            drive_mounted = not ui_components['drive_options'].children[0].disabled
            
            if use_drive and drive_mounted:
                drive_path = config.get('data', {}).get('drive_path', '')
                local_path = config.get('data', {}).get('local_clone_path', 'data_local')
                
                # Update status panel
                if 'status_panel' in ui_components:
                    from smartcash.ui.utils.constants import COLORS, ICONS
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
            show_distribution_visualization
        )
        
        # Update statistik dataset
        stats = get_dataset_stats(config, env, logger)
        if 'current_stats_html' in ui_components:
            from smartcash.ui.utils.constants import COLORS
            update_stats_cards(ui_components['current_stats_html'], stats, COLORS)
        
        # Show class distribution dalam output box
        if 'output_box' in ui_components:
            class_distribution = get_class_distribution(config, env, logger)
            from smartcash.ui.utils.constants import COLORS
            show_class_distribution_visualization(
                ui_components['output_box'],
                class_distribution,
                COLORS,
                logger
            )
        
        if logger: logger.info(f"✅ UI konfigurasi split berhasil diinisialisasi")
            
    except Exception as e:
        if logger: logger.error(f"❌ Error saat inisialisasi UI: {str(e)}")
        
        # Tampilkan pesan error sederhana di output box
        if 'output_box' in ui_components:
            from smartcash.ui.utils.constants import ICONS
            with ui_components['output_box']:
                display(HTML(f"""
                <div style="padding:10px; background-color:#f8d7da; color:#721c24; border-radius:4px;">
                    <p>{ICONS.get('error', '❌')} Error inisialisasi konfigurasi split: {str(e)}</p>
                </div>
                """))
    
    return ui_components

def _update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Update komponen UI dari konfigurasi dengan validasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
    """
    # Validasi komponen yang diperlukan
    if 'split_sliders' not in ui_components or not ui_components['split_sliders'] or len(ui_components['split_sliders']) < 3:
        return
    
    # Split sliders dengan validasi
    split_ratios = config.get('data', {}).get('split_ratios', {'train': 0.7, 'valid': 0.15, 'test': 0.15})
    ui_components['split_sliders'][0].value = split_ratios.get('train', 0.7) * 100
    ui_components['split_sliders'][1].value = split_ratios.get('valid', 0.15) * 100
    ui_components['split_sliders'][2].value = split_ratios.get('test', 0.15) * 100
    
    # Stratified checkbox
    if 'stratified' in ui_components:
        ui_components['stratified'].value = config.get('data', {}).get('stratified_split', True)
    
    # Advanced options dengan validasi
    if ('advanced_options' in ui_components and hasattr(ui_components['advanced_options'], 'children') 
            and len(ui_components['advanced_options'].children) >= 3):
        ui_components['advanced_options'].children[0].value = config.get('data', {}).get('random_seed', 42)
        ui_components['advanced_options'].children[1].value = config.get('data', {}).get('backup_before_split', True)
        ui_components['advanced_options'].children[2].value = config.get('data', {}).get('backup_dir', 'data/splits_backup')
    
    # Drive options jika tersedia
    if ('drive_options' in ui_components and hasattr(ui_components['drive_options'], 'children') 
            and len(ui_components['drive_options'].children) >= 4):
        ui_components['drive_options'].children[0].value = config.get('data', {}).get('use_drive', False)
        ui_components['drive_options'].children[1].value = config.get('data', {}).get('drive_path', '')
        ui_components['drive_options'].children[2].value = config.get('data', {}).get('local_clone_path', 'data_local')
        ui_components['drive_options'].children[3].value = config.get('data', {}).get('sync_on_change', True)