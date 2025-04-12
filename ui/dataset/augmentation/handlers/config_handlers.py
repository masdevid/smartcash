"""
File: smartcash/ui/dataset/augmentation/handlers/config_handlers.py
Deskripsi: Handler konfigurasi untuk augmentasi dataset
"""

import os
import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Optional
from IPython.display import display
from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

def setup_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi augmentasi dataset.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary berisi komponen UI yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Load konfigurasi
    config = load_augmentation_config(ui_components=ui_components)
    
    # Update UI dari konfigurasi
    update_ui_from_config(ui_components, config)
    
    # Handler untuk save config
    def on_save_config(b):
        """Handler untuk tombol save config."""
        try:
            # Update config dari UI
            updated_config = update_config_from_ui(ui_components, config)
            
            # Simpan konfigurasi
            success = save_augmentation_config(updated_config)
            
            # Update UI
            status_type = 'success' if success else 'error'
            message = f"{ICONS['success'] if success else ICONS['error']} Konfigurasi {'berhasil' if success else 'gagal'} disimpan"
            
            # Tampilkan status
            with ui_components['status']: 
                display(create_status_indicator(status_type, message))
                
            # Update status panel
            from smartcash.ui.dataset.shared.status_panel import update_status_panel
            update_status_panel(ui_components, status_type, message)
            
            # Log
            if logger: 
                log_method = logger.success if success else logger.error
                log_method(message)
        except Exception as e:
            if logger: logger.error(f"{ICONS['error']} Error saat save config: {str(e)}")
            with ui_components['status']: 
                display(create_status_indicator("error", f"{ICONS['error']} Error: {str(e)}"))
    
    # Register handler ke save button
    ui_components['save_button'].on_click(on_save_config)
    
    # Tambahkan fungsi-fungsi ke ui_components
    ui_components.update({
        'on_save_config': on_save_config,
        'update_config_from_ui': update_config_from_ui,
        'save_augmentation_config': save_augmentation_config,
        'load_augmentation_config': load_augmentation_config,
        'update_ui_from_config': update_ui_from_config,
        'config': config
    })
    
    return ui_components


def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Update konfigurasi dari UI components.
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi existing
        
    Returns:
        Dictionary berisi konfigurasi yang diupdate
    """
    # Deep copy config untuk mencegah modifikasi tidak sengaja
    config = copy.deepcopy(config or {})
    
    # Pastikan section augmentation dan data tersedia
    if 'augmentation' not in config: config['augmentation'] = {}
    if 'data' not in config: config['data'] = {}
    
    # Map UI types to config types
    type_map = {
        'Combined (Recommended)': 'combined', 
        'Position Variations': 'position', 
        'Lighting Variations': 'lighting', 
        'Extreme Rotation': 'extreme_rotation'
    }
    
    # Ekstrak nilai dari UI
    aug_types = [type_map.get(t, 'combined') for t in ui_components['aug_options'].children[0].value]
    variations = ui_components['aug_options'].children[1].value
    prefix = ui_components['aug_options'].children[2].value
    process_bboxes = ui_components['aug_options'].children[3].value
    validate_results = ui_components['aug_options'].children[4].value
    num_workers = ui_components['aug_options'].children[5].value if len(ui_components['aug_options'].children) > 5 else 4
    target_balance = ui_components['aug_options'].children[6].value if len(ui_components['aug_options'].children) > 6 else False
    
    # Update config
    config['augmentation'].update({
        'enabled': True,
        'types': aug_types,
        'num_variations': variations,
        'output_prefix': prefix,
        'process_bboxes': process_bboxes,
        'validate_results': validate_results,
        'num_workers': num_workers,
        'target_balance': target_balance,
        'resume': False
    })
    
    # Update path lokasi preprocessed
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    augmented_dir = ui_components.get('augmented_dir', 'data/augmented')
    
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    config['preprocessing']['preprocessed_dir'] = preprocessed_dir
    config['augmentation']['output_dir'] = augmented_dir
    
    # Pastikan file_prefix preprocessed tersedia
    if 'file_prefix' not in config['preprocessing']:
        config['preprocessing']['file_prefix'] = 'rp'
    
    # Simpan ke ui_components
    ui_components['config'] = config
    
    return config


def save_augmentation_config(config: Dict[str, Any], config_path: str = "configs/augmentation_config.yaml") -> bool:
    """
    Simpan konfigurasi augmentasi ke file.
    
    Args:
        config: Konfigurasi yang akan disimpan
        config_path: Path file konfigurasi
        
    Returns:
        Boolean menandakan keberhasilan
    """
    try:
        # Coba dapatkan logger jika tersedia
        logger = None
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger("augmentation_config")
        except ImportError:
            pass
            
        # Deep copy untuk mencegah modifikasi tidak sengaja
        save_config = copy.deepcopy(config)
        
        # Pastikan direktori config ada
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Cek jika file sudah ada, merge dengan config yang ada
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                existing_config = yaml.safe_load(f) or {}
                
            # Merge existing config dengan config baru
            merged_config = copy.deepcopy(existing_config)
            
            # Update config dengan augmentation dan preprocessing settings baru
            for section in ['augmentation', 'preprocessing']:
                if section in save_config:
                    if section not in merged_config:
                        merged_config[section] = {}
                    merged_config[section].update(save_config[section])
                
            # Gunakan config yang sudah dimerge
            save_config = merged_config
        
        # Simpan ke file
        with open(config_path, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False)
        
        # Coba sync dengan drive jika tersedia
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Cek apakah path sama dengan realpath untuk mencegah error symlink
                if os.path.realpath(config_path) == os.path.realpath(drive_config_path):
                    if logger: logger.info(f"🔄 File lokal dan drive identik: {config_path}, melewati salinan")
                else:
                    # Buat direktori jika belum ada
                    os.makedirs(Path(drive_config_path).parent, exist_ok=True)
                    
                    # Salin file ke drive
                    with open(drive_config_path, 'w') as f:
                        yaml.dump(save_config, f, default_flow_style=False)
                    if logger: logger.info(f"📤 Konfigurasi disimpan ke drive: {drive_config_path}")
        except (ImportError, Exception) as e:
            if logger: logger.debug(f"ℹ️ Tidak dapat menyalin ke drive: {str(e)}")
        
        return True
    except Exception as e:
        if logger: logger.error(f"❌ Error saat menyimpan konfigurasi: {str(e)}")
        return False