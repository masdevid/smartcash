"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Handler konfigurasi untuk preprocessing dataset
"""

from typing import Dict, Any, Optional
import os
import yaml
import copy
from pathlib import Path
from IPython.display import display
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_PREPROCESSED_DIR, DEFAULT_INVALID_DIR, DEFAULT_IMG_SIZE
from smartcash.common.config import get_config_manager
from smartcash.ui.utils.constants import ICONS

def setup_preprocessing_config_handler(ui_components: Dict[str, Any], config: Dict[str, Any] = None, env=None) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi preprocessing dengan persistensi yang ditingkatkan.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        env: Environment manager
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Dapatkan instance ConfigManager
    config_manager = get_config_manager()
    
    # Register UI components untuk persistensi
    config_manager.register_ui_components('preprocessing', ui_components)
    
    # Load konfigurasi jika belum tersedia
    if config is None:
        # Coba dapatkan dari ConfigManager terlebih dahulu
        config = config_manager.get_module_config('preprocessing')
        if not config:
            # Fallback ke load dari file
            config = load_preprocessing_config(ui_components=ui_components)
    
    # Update UI dari konfigurasi
    ui_components = update_ui_from_config(ui_components, config)
    
    # Handler untuk tombol save config
    def on_save_config(b):
        from smartcash.ui.utils.alert_utils import create_status_indicator
        from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import ensure_ui_persistence
        
        try:
            # Update config dari UI
            updated_config = update_config_from_ui(ui_components, ui_components.get('config', config))
            
            # Simpan ke ConfigManager
            success = config_manager.save_module_config('preprocessing', updated_config)
            
            # Simpan juga ke file lokal untuk kompatibilitas
            local_success = save_preprocessing_config(updated_config)
            
            # Pastikan UI components terdaftar untuk persistensi
            ensure_ui_persistence(ui_components, updated_config)
            
            # Simpan kembali config yang diupdate ke ui_components
            ui_components['config'] = updated_config
            
            # Tampilkan status
            status_type = 'success' if success and local_success else 'error'
            message = f"{ICONS['success' if success else 'error']} Konfigurasi {'berhasil' if success else 'gagal'} disimpan"
            
            # Sinkronkan dengan drive
            try:
                from smartcash.ui.dataset.preprocessing.handlers.persistence_handler import sync_config_with_drive
                drive_sync = sync_config_with_drive(ui_components)
                if drive_sync and logger:
                    logger.info(f"{ICONS['success']} Konfigurasi berhasil disinkronkan dengan drive")
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS['warning']} Gagal menyinkronkan dengan drive: {str(e)}")
        except Exception as e:
            status_type = 'error'
            message = f"{ICONS['error']} Error saat menyimpan konfigurasi: {str(e)}"
            if logger: logger.error(message)
        
        # Update status
        with ui_components['status']: 
            display(create_status_indicator(status_type, message))
            
        # Update status panel
        from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel
        update_status_panel(ui_components, status_type, message)
        
        # Log
        if logger: 
            log_method = logger.success if success else logger.error
            log_method(message)
    
    # Register handler untuk tombol save
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(on_save_config)
    
    # Tambahkan referensi fungsi ke UI components
    ui_components.update({
        'update_config_from_ui': update_config_from_ui,
        'save_preprocessing_config': save_preprocessing_config,
        'load_preprocessing_config': load_preprocessing_config,
        'update_ui_from_config': update_ui_from_config,
        'on_save_config': on_save_config,
        'config': config  # Simpan referensi config di ui_components
    })
    
    return ui_components

def update_config_from_ui(ui_components: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Ekstrak dan update konfigurasi dari UI dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary konfigurasi yang diupdate
    """
    # Inisialisasi config dengan deep copy untuk mencegah modifikasi tidak sengaja
    config = copy.deepcopy(config or {})
    logger = ui_components.get('logger')
    
    # Pastikan section preprocessing dan data ada
    if 'preprocessing' not in config: config['preprocessing'] = {}
    if 'data' not in config: config['data'] = {}
    
    # Ekstrak paths dari ui_components
    data_dir = ui_components.get('data_dir', 'data')
    preprocessed_dir = ui_components.get('preprocessed_dir', 'data/preprocessed')
    
    # Update paths dalam config
    config['data']['dir'] = data_dir
    config['preprocessing']['output_dir'] = preprocessed_dir
    
    # Ekstrak nilai dari preprocess_options
    options = ui_components.get('preprocess_options', {})
    if hasattr(options, 'children') and len(options.children) >= 5:
        # Ekstrak semua nilai dengan list comprehension
        values = [child.value for child in options.children[:5]] 
        img_size, normalize, preserve_ratio, enable_cache, num_workers = values
        
        # Update konfigurasi preprocessing
        config['preprocessing'].update({
            'img_size': [img_size, img_size],
            'enabled': enable_cache,
            'num_workers': num_workers,
            'normalization': {
                'enabled': normalize,
                'preserve_aspect_ratio': preserve_ratio,
                'target_size': [img_size, img_size]
            }
        })
    
    # Ekstrak nilai validation
    validation = ui_components.get('validation_options')
    if hasattr(validation, 'children') and len(validation.children) >= 4:
        # Ekstrak nilai dengan list comprehension
        values = [child.value for child in validation.children[:4]]
        validate_enabled, fix_issues, move_invalid, invalid_dir = values
        
        # Update konfigurasi validation
        config['preprocessing']['validate'] = {
            'enabled': validate_enabled,
            'fix_issues': fix_issues,
            'move_invalid': move_invalid,
            'invalid_dir': invalid_dir
        }
    
    # Ekstrak split selector
    split_selector = ui_components.get('split_selector')
    if hasattr(split_selector, 'value'):
        # Map nilai UI ke konfigurasi dengan dictionary
        split_map = {
            'All Splits': DEFAULT_SPLITS,
            'Train Only': ['train'],
            'Validation Only': ['valid'],
            'Test Only': ['test']
        }
        config['preprocessing']['splits'] = split_map.get(split_selector.value, DEFAULT_SPLITS)
    
    # Simpan referensi config di ui_components untuk memastikan persistensi
    ui_components['config'] = config
    
    if logger: logger.debug(f"🔄 Konfigurasi preprocessing berhasil diupdate dari UI")
    
    return config

def save_preprocessing_config(config: Dict[str, Any], config_path: str = "configs/preprocessing_config.yaml") -> bool:
    """
    Simpan konfigurasi preprocessing dengan penanganan persistensi yang lebih baik.
    
    Args:
        config: Konfigurasi aplikasi
        config_path: Path file konfigurasi
        
    Returns:
        Boolean status keberhasilan
    """
    # Pastikan config tidak None
    if config is None:
        return False
    
    # Deep copy untuk mencegah modifikasi tidak sengaja
    config = copy.deepcopy(config)
    
    # Pastikan direktori config ada
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir, exist_ok=True)
        except Exception:
            return False
    
    # Simpan konfigurasi ke file
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Simpan juga ke ConfigManager untuk persistensi
        try:
            config_manager = get_config_manager()
            config_manager.save_module_config('preprocessing', config)
        except Exception:
            # Jika gagal menyimpan ke ConfigManager, tetap lanjutkan
            # karena sudah berhasil menyimpan ke file
            pass
            
        return True
    except Exception:
        return False

def load_preprocessing_config(config_path: str = "configs/preprocessing_config.yaml", ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Load konfigurasi preprocessing dengan persistensi yang disempurnakan.
    
    Args:
        config_path: Path file konfigurasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary konfigurasi
    """
    logger = ui_components.get('logger') if ui_components else None
    
    # Coba dapatkan dari ConfigManager terlebih dahulu
    try:
        config_manager = get_config_manager()
        config = config_manager.get_module_config('preprocessing')
        
        if config and isinstance(config, dict) and 'preprocessing' in config:
            if logger: logger.info(f"{ICONS['success']} Konfigurasi preprocessing dimuat dari ConfigManager")
            
            # Simpan ke ui_components jika tersedia
            if ui_components: ui_components['config'] = config
            
            return config
    except Exception as e:
        if logger: logger.debug(f"{ICONS['info']} Tidak dapat memuat dari ConfigManager: {str(e)}")
    
    # Coba load dari file jika tidak ada di ConfigManager
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validasi konfigurasi
            if config is None:
                config = {}
            
            # Pastikan struktur konfigurasi benar
            if 'preprocessing' not in config:
                config['preprocessing'] = {}
            if 'data' not in config:
                config['data'] = {}
                
            # Log info
            if logger: logger.info(f"{ICONS['success']} Konfigurasi preprocessing dimuat dari {config_path}")
            
            # Simpan ke ConfigManager untuk persistensi
            try:
                config_manager = get_config_manager()
                config_manager.save_module_config('preprocessing', config)
            except Exception as e:
                if logger: logger.debug(f"{ICONS['info']} Tidak dapat menyimpan ke ConfigManager: {str(e)}")
            
            # Simpan ke ui_components jika tersedia
            if ui_components: ui_components['config'] = config
            
            return config
    except Exception as e:
        if logger: logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi dari {config_path}: {str(e)}")
    
    # Jika gagal load dari file dan ConfigManager, gunakan default
    default_config = {
        "preprocessing": {
            "enabled": True,
            "output_dir": DEFAULT_PREPROCESSED_DIR,
            "img_size": DEFAULT_IMG_SIZE,
            "normalization": {
                "enabled": True,
                "preserve_aspect_ratio": True
            },
            "validate": {
                "enabled": True,
                "fix_issues": True,
                "move_invalid": True,
                "invalid_dir": DEFAULT_INVALID_DIR
            },
            "splits": DEFAULT_SPLITS,
            "num_workers": 4
        },
        "data": {
            "dir": "data"
        }
    }
    
    if logger: logger.info(f"{ICONS['info']} Menggunakan konfigurasi default")
    
    # Simpan default config ke ConfigManager
    try:
        config_manager = get_config_manager()
        config_manager.save_module_config('preprocessing', default_config)
    except Exception as e:
        if logger: logger.debug(f"{ICONS['info']} Tidak dapat menyimpan default ke ConfigManager: {str(e)}")
    
    # Simpan default config ke ui_components jika tersedia
    if ui_components: ui_components['config'] = default_config
    
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
    
    if logger: logger.debug(f"🔄 UI berhasil diupdate dari konfigurasi")
    
    return ui_components