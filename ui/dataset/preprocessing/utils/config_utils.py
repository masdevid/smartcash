"""
File: smartcash/ui/dataset/preprocessing/utils/config_utils.py
Deskripsi: Utilitas konfigurasi untuk preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
import yaml
import copy
from pathlib import Path
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_PREPROCESSED_DIR, DEFAULT_INVALID_DIR, DEFAULT_IMG_SIZE
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.constants import ICONS

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Ekstrak dan update konfigurasi dari UI dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary konfigurasi yang diupdate
    """
    logger = ui_components.get('logger')
    
    # Buat config baru jika belum ada
    if config is None:
        config = {}
    
    # Deep copy untuk menghindari modifikasi referensi
    config = copy.deepcopy(config)
    
    # Pastikan struktur config lengkap
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    if 'data' not in config:
        config['data'] = {}
    
    # Ekstrak nilai dari UI components
    preproc_options = ui_components.get('preprocess_options')
    validation_options = ui_components.get('validation_options')
    split_selector = ui_components.get('split_selector')
    
    # Update preprocessing config
    preproc_config = config['preprocessing']
    
    # Update image size
    if preproc_options and hasattr(preproc_options, 'children') and len(preproc_options.children) >= 1:
        img_size = preproc_options.children[0].value
        preproc_config['img_size'] = [img_size, img_size]  # Square aspect ratio
    
    # Update normalization options
    if preproc_options and hasattr(preproc_options, 'children') and len(preproc_options.children) >= 3:
        if 'normalization' not in preproc_config:
            preproc_config['normalization'] = {}
        
        preproc_config['normalization'].update({
            'enabled': preproc_options.children[1].value,
            'preserve_aspect_ratio': preproc_options.children[2].value
        })
    
    # Update cache dan workers
    if preproc_options and hasattr(preproc_options, 'children') and len(preproc_options.children) >= 5:
        preproc_config.update({
            'enabled': preproc_options.children[3].value,
            'num_workers': preproc_options.children[4].value
        })
    
    # Update validation options
    if validation_options and hasattr(validation_options, 'children') and len(validation_options.children) >= 4:
        if 'validate' not in preproc_config:
            preproc_config['validate'] = {}
        
        preproc_config['validate'].update({
            'enabled': validation_options.children[0].value,
            'fix_issues': validation_options.children[1].value,
            'move_invalid': validation_options.children[2].value,
            'invalid_dir': validation_options.children[3].value
        })
    
    # Update splits berdasarkan pilihan di UI
    if split_selector and hasattr(split_selector, 'value'):
        split_map = {
            'All Splits': DEFAULT_SPLITS,
            'Train Only': [DEFAULT_SPLITS[0]],
            'Validation Only': ['valid'],
            'Test Only': ['test']
        }
        preproc_config['splits'] = split_map.get(split_selector.value, DEFAULT_SPLITS)
    
    # Tambahkan output_dir jika belum ada
    if 'output_dir' not in preproc_config:
        preproc_config['output_dir'] = DEFAULT_PREPROCESSED_DIR
    
    # Log
    if logger: 
        logger.debug(f"🔄 Konfigurasi berhasil diupdate dari UI")
    
    return config

def save_preprocessing_config(config: Dict[str, Any], config_path: str = "configs/preprocessing_config.yaml") -> bool:
    """
    Simpan konfigurasi preprocessing dengan SimpleConfigManager.
    
    Args:
        config: Konfigurasi aplikasi
        config_path: Path file konfigurasi
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan config manager
        config_manager = get_config_manager()
        
        # Simpan ke SimpleConfigManager menggunakan metode save_module_config
        success = config_manager.save_module_config('preprocessing', config)
        
        if success:
            # Coba simpan juga ke file untuk kompatibilitas
            try:
                # Pastikan direktori ada
                config_dir = os.path.dirname(config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                
                # Simpan ke file
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            except Exception as e:
                # Gagal menyimpan ke file, tapi berhasil menyimpan ke ConfigManager
                # Jadi kita anggap sukses
                pass
                
            return True
        else:
            # Jika gagal menyimpan ke ConfigManager, coba fallback ke file
            try:
                # Pastikan direktori ada
                config_dir = os.path.dirname(config_path)
                if config_dir and not os.path.exists(config_dir):
                    os.makedirs(config_dir, exist_ok=True)
                
                # Simpan ke file
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                return True
            except Exception:
                return False
    except Exception as e:
        # Log error tapi jangan raise exception
        return False

def load_preprocessing_config(config_path: str = "configs/preprocessing_config.yaml", ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load konfigurasi preprocessing dengan SimpleConfigManager.
    
    Args:
        config_path: Path file konfigurasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi
    """
    logger = ui_components.get('logger') if ui_components else None
    
    try:
        # Dapatkan config manager dan load dari module config
        config_manager = get_config_manager()
        config = config_manager.get_module_config('preprocessing')
        
        if config:
            if logger: 
                logger.debug(f"{ICONS['info']} Konfigurasi berhasil diload dari SimpleConfigManager")
            return config
    except Exception as e:
        if logger: 
            logger.debug(f"{ICONS['info']} Tidak dapat load dari SimpleConfigManager: {str(e)}")
    
    # Default config yang akan digunakan jika tidak ada konfigurasi
    default_config = {
        'data': {
            'dir': 'data'
        },
        'preprocessing': {
            'img_size': DEFAULT_IMG_SIZE,
            'output_dir': DEFAULT_PREPROCESSED_DIR,
            'splits': DEFAULT_SPLITS,
            'enabled': True,
            'num_workers': 4,
            'normalization': {
                'enabled': True,
                'preserve_aspect_ratio': True
            },
            'validate': {
                'enabled': True,
                'fix_issues': True,
                'move_invalid': True,
                'invalid_dir': DEFAULT_INVALID_DIR
            }
        }
    }
    
    # Coba simpan default ke SimpleConfigManager
    try:
        config_manager = get_config_manager()
        config_manager.save_module_config('preprocessing', default_config)
        if logger: 
            logger.debug(f"{ICONS['info']} Default konfigurasi berhasil disimpan ke SimpleConfigManager")
    except Exception as e:
        if logger: 
            logger.debug(f"{ICONS['info']} Tidak dapat menyimpan default ke SimpleConfigManager: {str(e)}")
        
        # Fallback: Coba simpan ke file
        try:
            # Pastikan direktori ada
            config_dir = os.path.dirname(config_path)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
            
            # Simpan ke file
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            if logger: 
                logger.debug(f"{ICONS['info']} Default konfigurasi berhasil disimpan ke file {config_path}")
        except Exception as file_e:
            if logger: 
                logger.debug(f"{ICONS['warning']} Gagal menyimpan default ke file: {str(file_e)}")
    
    # Simpan default config ke ui_components jika tersedia
    if ui_components: 
        ui_components['config'] = default_config
    
    return default_config

def update_ui_from_config(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update komponen UI dari konfigurasi.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Update data paths dan simpan di UI components
    data_dir = config.get('data', {}).get('dir', 'data')
    preprocessed_dir = config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
    ui_components.update({'data_dir': data_dir, 'preprocessed_dir': preprocessed_dir})
    
    # Update preprocess_options dengan nilai dari config
    preproc_config = config.get('preprocessing', {})
    preproc_options = ui_components.get('preprocess_options')
    
    if preproc_options and hasattr(preproc_options, 'children'):
        # Update opsi dengan list comprehension
        if len(preproc_options.children) >= 5:
            # Ambil komponen pertama dari array img_size jika tuple/list
            img_size = preproc_config.get('img_size', DEFAULT_IMG_SIZE)
            img_size_value = img_size[0] if isinstance(img_size, (list, tuple)) else img_size
            
            # Definisi pasangan field dan nilai dari config
            option_values = [
                img_size_value,  # Image size (integer)
                preproc_config.get('normalization', {}).get('enabled', True),  # Normalize
                preproc_config.get('normalization', {}).get('preserve_aspect_ratio', True),  # Aspect ratio
                preproc_config.get('enabled', True),  # Cache enabled
                preproc_config.get('num_workers', 4)  # Workers
            ]
            
            # Update nilai options satu per satu untuk menghindari error
            for i, value in enumerate(option_values):
                if i < len(preproc_options.children):
                    preproc_options.children[i].value = value
    
    # Update validation options dengan data dari config
    validation_options = ui_components.get('validation_options')
    validation_config = preproc_config.get('validate', {})
    
    if validation_options and hasattr(validation_options, 'children'):
        if len(validation_options.children) >= 4:
            # Definisi pasangan field dan nilai dari config
            validation_values = [
                validation_config.get('enabled', True),  # Enable validation
                validation_config.get('fix_issues', True),  # Fix issues
                validation_config.get('move_invalid', True),  # Move invalid
                validation_config.get('invalid_dir', 'data/invalid')  # Invalid dir
            ]
            
            # Update nilai options satu per satu
            for i, value in enumerate(validation_values):
                if i < len(validation_options.children):
                    validation_options.children[i].value = value
    
    # Update split selector
    split_selector = ui_components.get('split_selector')
    if split_selector and hasattr(split_selector, 'value'):
        # Map dari config value ke UI value
        split_list = preproc_config.get('splits', DEFAULT_SPLITS)
        split_str = str(sorted(split_list))
        
        split_map = {
            str(sorted(DEFAULT_SPLITS)): 'All Splits',
            str([DEFAULT_SPLITS[0]]): 'Train Only',
            str(['valid']): 'Validation Only',
            str(['test']): 'Test Only'
        }
        
        split_selector.value = split_map.get(split_str, 'All Splits')
    
    # Simpan referensi config di ui_components untuk persistensi
    ui_components['config'] = config
    
    if logger: 
        logger.debug(f"🔄 UI berhasil diupdate dari konfigurasi")
    
    return ui_components
