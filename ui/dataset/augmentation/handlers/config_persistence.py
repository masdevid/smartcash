"""
File: smartcash/ui/dataset/augmentation/handlers/config_persistence.py
Deskripsi: Helper untuk persistensi konfigurasi augmentasi dengan ConfigManager
"""

from typing import Dict, Any, Optional, List, Union
import os
import yaml
from pathlib import Path
from IPython.display import display, HTML
from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.utils.constants import ICONS

# Dapatkan logger untuk modul ini
logger = get_logger('augmentation.config_persistence')

def get_config_manager_instance():
    """Dapatkan instance ConfigManager dengan pendekatan singleton."""
    return get_config_manager()

def validate_augmentation_params(value: Any, default_value: Any, 
                              valid_types: Optional[Union[type, List[type]]] = None, 
                              valid_values: Optional[List[Any]] = None) -> Any:
    """Validasi parameter augmentasi dengan fallback ke nilai default."""
    config_manager = get_config_manager_instance()
    return config_manager.validate_param(value, default_value, valid_types, valid_values)

def ensure_ui_persistence(ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None, module_name: str = 'augmentation') -> Dict[str, Any]:
    """Pastikan persistensi UI components dengan mendaftarkannya ke ConfigManager.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi tambahan (opsional)
        module_name: Nama modul untuk persistensi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    try:
        # Dapatkan logger jika tersedia
        local_logger = ui_components.get('logger', logger)
        
        # Dapatkan instance ConfigManager
        config_manager = get_config_manager_instance()
        
        # Register UI components untuk persistensi
        config_manager.register_ui_components(module_name, ui_components)
        
        # Dapatkan konfigurasi jika tidak disediakan
        if not config:
            config = config_manager.get_module_config(module_name)
            
            # Jika masih None, coba load dari file
            if not config:
                try:
                    config = get_augmentation_config()
                except Exception as e:
                    local_logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi: {str(e)}")
        
        # Update UI dari konfigurasi jika ada
        if config:
            try:
                from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_config_to_ui
                map_config_to_ui(ui_components, config)
                # Simpan referensi config di ui_components
                ui_components['config'] = config
            except Exception as e:
                local_logger.warning(f"{ICONS['warning']} Gagal update UI dari konfigurasi: {str(e)}")
        
        # Log info
        local_logger.info(f"{ICONS['success']} UI components berhasil terdaftar untuk persistensi")
        
        return ui_components
    except Exception as e:
        local_logger = ui_components.get('logger', logger)
        local_logger.error(f"{ICONS['error']} Error saat memastikan persistensi UI: {str(e)}")
        return ui_components

def get_persisted_ui_components(module_name: str = 'augmentation') -> Dict[str, Any]:
    """Dapatkan UI components yang tersimpan dari ConfigManager."""
    config_manager = get_config_manager_instance()
    return config_manager.get_ui_components(module_name)

def get_augmentation_config(default_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Dapatkan konfigurasi augmentasi dari ConfigManager atau file lokal.
    
    Args:
        default_config: Konfigurasi default jika tidak ditemukan
        
    Returns:
        Dictionary konfigurasi augmentasi
    """
    # Coba dapatkan dari ConfigManager terlebih dahulu
    try:
        config_manager = get_config_manager_instance()
        config = config_manager.get_module_config('augmentation')
        if config and isinstance(config, dict) and 'augmentation' in config:
            logger.info(f"{ICONS['success']} Konfigurasi augmentation berhasil dimuat dari ConfigManager")
            return config
    except Exception as e:
        logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi dari ConfigManager: {str(e)}")
    
    # Coba load dari file lokal
    try:
        config_path = "configs/augmentation_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config and isinstance(config, dict) and 'augmentation' in config:
                    logger.info(f"{ICONS['success']} Konfigurasi augmentation berhasil dimuat dari {config_path}")
                    return config
    except Exception as e:
        logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi dari file: {str(e)}")
    
    # Coba load dari Google Drive jika terpasang
    try:
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        
        if env_manager.is_drive_mounted:
            drive_config_path = str(env_manager.drive_path / 'configs' / 'augmentation_config.yaml')
            
            if os.path.exists(drive_config_path):
                with open(drive_config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    if config and isinstance(config, dict) and 'augmentation' in config:
                        logger.info(f"{ICONS['success']} Konfigurasi augmentation berhasil dimuat dari drive: {drive_config_path}")
                        return config
    except Exception as e:
        logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi dari drive: {str(e)}")
    
    # Fallback ke default jika semua gagal
    if default_config is None:
        default_config = get_default_augmentation_config()
        logger.info(f"{ICONS['info']} Menggunakan konfigurasi default untuk augmentation")
    
    return default_config

def save_augmentation_config(config: Dict[str, Any]) -> bool:
    """Simpan konfigurasi augmentasi ke ConfigManager dan file lokal.
    
    Args:
        config: Konfigurasi yang akan disimpan
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Validasi konfigurasi
        if not config or not isinstance(config, dict) or 'augmentation' not in config:
            logger.warning(f"{ICONS['warning']} Konfigurasi tidak valid untuk disimpan")
            return False
        
        # Simpan ke ConfigManager
        config_manager = get_config_manager_instance()
        success = config_manager.save_module_config('augmentation', config)
        
        if not success:
            logger.warning(f"{ICONS['warning']} Gagal menyimpan konfigurasi ke ConfigManager")
            return False
        
        # Simpan juga ke file lokal
        try:
            config_path = "configs/augmentation_config.yaml"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
                
            # Jika drive terpasang, simpan juga ke drive
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Buat direktori jika belum ada
                os.makedirs(os.path.dirname(drive_config_path), exist_ok=True)
                
                # Salin file ke Google Drive
                with open(drive_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                    
                logger.info(f"{ICONS['success']} Konfigurasi disimpan ke drive: {drive_config_path}")
        except Exception as e:
            logger.warning(f"{ICONS['warning']} Gagal menyimpan ke file: {str(e)}")
        
        logger.info(f"{ICONS['success']} Konfigurasi augmentation berhasil disimpan")
        return True
    except Exception as e:
        logger.error(f"{ICONS['error']} Error saat menyimpan konfigurasi: {str(e)}")
        return False

def register_config_observer(callback: callable) -> bool:
    """Register observer untuk notifikasi perubahan konfigurasi augmentasi."""
    config_manager = get_config_manager_instance()
    return config_manager.register_observer('augmentation', callback)

def ensure_valid_aug_types(aug_types: Any) -> List[str]:
    """Pastikan aug_types selalu valid dengan validasi yang kuat."""
    # Default jika tidak valid
    default_aug_types = ['combined']
    
    # Validasi None
    if aug_types is None:
        return default_aug_types
    
    # Validasi string
    if isinstance(aug_types, str):
        return [aug_types]
    
    # Validasi list
    if isinstance(aug_types, (list, tuple)):
        # Filter nilai None dan string kosong
        valid_types = [t for t in aug_types if t is not None and str(t).strip()]
        return valid_types if valid_types else default_aug_types
    
    # Fallback ke default jika tipe tidak dikenali
    return default_aug_types

def safe_convert_type(value: Any, target_type: type, default_value: Any = None) -> Any:
    """Konversi nilai ke tipe target dengan aman.
    
    Args:
        value: Nilai yang akan dikonversi
        target_type: Tipe target (str, int, float, bool)
        default_value: Nilai default jika konversi gagal
        
    Returns:
        Nilai yang sudah dikonversi atau default_value jika konversi gagal
    """
    if value is None:
        return default_value
        
    try:
        if target_type == bool and isinstance(value, str):
            # Konversi string ke boolean dengan lebih aman
            return value.lower() in ('true', 'yes', 'y', '1', 'on')
        return target_type(value)
    except (ValueError, TypeError):
        return default_value

def validate_ui_component_value(component: Any, expected_type: type, default_value: Any = None) -> Any:
    """Validasi nilai komponen UI dengan tipe yang diharapkan.
    
    Args:
        component: Komponen UI
        expected_type: Tipe yang diharapkan
        default_value: Nilai default jika validasi gagal
        
    Returns:
        Nilai yang sudah divalidasi
    """
    if component is None or not hasattr(component, 'value'):
        return default_value
        
    try:
        value = component.value
        return safe_convert_type(value, expected_type, default_value)
    except Exception:
        return default_value

def get_default_augmentation_config() -> Dict[str, Any]:
    """
    Mendapatkan konfigurasi default untuk augmentasi.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        'augmentation': {
            # Parameter utama untuk service
            'types': ['combined'],
            'output_prefix': 'aug_',
            'num_variations': 2,
            'split': 'train',
            'validate_results': True,
            'process_bboxes': True,
            'target_balance': True,
            'balance_classes': True,  # Untuk backward compatibility
            'num_workers': 4,
            'move_to_preprocessed': True,
            'target_count': 1000,
            'resume': False,
            
            # Parameter untuk UI dan backward compatibility
            'prefix': 'aug_',  # Untuk backward compatibility
            'factor': 2,  # Untuk backward compatibility
            
            # Parameter teknik augmentasi
            'techniques': {
                'flip': True,
                'rotate': True,
                'blur': False,
                'noise': False,
                'contrast': False,
                'brightness': False,
                'saturation': False,
                'hue': False,
                'cutout': False
            },
            'advanced': {
                'rotate_range': 15,
                'blur_limit': 7,
                'noise_var': 25,
                'contrast_limit': 0.2,
                'brightness_limit': 0.2,
                'saturation_limit': 0.2,
                'hue_shift_limit': 20,
                'cutout_size': 0.1,
                'cutout_count': 4
            }
        },
        # Data path untuk service
        'data': {
            'dataset_path': 'data/preprocessed'
        }
    }

def reset_config_to_default(ui_components: Dict[str, Any]) -> bool:
    """
    Mengatur ulang konfigurasi augmentasi ke nilai default.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Dapatkan logger jika tersedia
        local_logger = ui_components.get('logger', logger)
        
        # Dapatkan config manager
        config_manager = get_config_manager_instance()
        
        # Dapatkan konfigurasi default
        default_config = get_default_augmentation_config()
        
        # Validasi konfigurasi
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_validator import validate_augmentation_config
            default_config = validate_augmentation_config(default_config)
        except Exception as e:
            local_logger.warning(f"{ICONS['warning']} Gagal validasi konfigurasi default: {str(e)}")
        
        # Log konfigurasi default
        local_logger.debug(f"{ICONS['info']} Konfigurasi default yang akan digunakan: {default_config}")
        
        # Perbarui UI components
        from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_config_to_ui
        ui_components = map_config_to_ui(ui_components, default_config)
        
        # Daftarkan UI components untuk persistensi
        config_manager.register_ui_components('augmentation', ui_components)
        
        # Simpan konfigurasi default
        config_manager.save_module_config('augmentation', default_config)
        
        # Simpan juga ke file lokal
        try:
            config_path = os.path.join('configs', 'augmentation_config.yaml')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        except Exception as e:
            local_logger.warning(f"{ICONS['warning']} Gagal menyimpan default ke file: {str(e)}")
        
        # Tampilkan pesan sukses
        display(HTML('<div style="padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px;">' +
                     '<b>\u2705 Konfigurasi augmentasi berhasil diatur ulang ke nilai default!</b></div>'))
        
        return True
        
    except Exception as e:
        # Tampilkan pesan error
        display(HTML('<div style="padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px;">' +
                     f'<b>\u274c Gagal mengatur ulang konfigurasi augmentasi: {str(e)}</b></div>'))
        logger.error(f"{ICONS['error']} Error saat reset konfigurasi: {str(e)}")
        return False

def sync_config_with_drive(ui_components: Dict[str, Any]) -> bool:
    """
    Menyimpan konfigurasi augmentasi ke Google Drive.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Dapatkan logger jika tersedia
        local_logger = ui_components.get('logger', logger)
        
        # Dapatkan config manager
        config_manager = get_config_manager_instance()
        
        # Dapatkan konfigurasi dari UI
        from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_ui_to_config
        updated_config = map_ui_to_config(ui_components)
        
        # Validasi konfigurasi
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_validator import validate_augmentation_config
            updated_config = validate_augmentation_config(updated_config)
        except Exception as e:
            local_logger.warning(f"{ICONS['warning']} Konfigurasi tidak valid untuk disinkronkan: {str(e)}")
            return False
        
        # Pastikan struktur konfigurasi benar
        if 'augmentation' not in updated_config:
            updated_config['augmentation'] = {}
        
        # Pastikan parameter yang dibutuhkan oleh service tersedia
        aug_config = updated_config.get('augmentation', {})
        if 'types' not in aug_config:
            aug_config['types'] = ['combined']
        if 'output_prefix' not in aug_config:
            aug_config['output_prefix'] = aug_config.get('prefix', 'aug_')
        if 'num_variations' not in aug_config:
            aug_config['num_variations'] = int(aug_config.get('factor', 2))
        if 'validate_results' not in aug_config:
            aug_config['validate_results'] = True
        if 'process_bboxes' not in aug_config:
            aug_config['process_bboxes'] = True
        if 'target_balance' not in aug_config:
            aug_config['target_balance'] = aug_config.get('balance_classes', True)
        if 'balance_classes' not in aug_config:
            aug_config['balance_classes'] = aug_config.get('target_balance', True)
        if 'num_workers' not in aug_config:
            aug_config['num_workers'] = 4
        if 'move_to_preprocessed' not in aug_config:
            aug_config['move_to_preprocessed'] = True
        if 'target_count' not in aug_config:
            aug_config['target_count'] = 1000
        if 'resume' not in aug_config:
            aug_config['resume'] = False
            
        # Pastikan data path tersedia
        if 'data' not in updated_config:
            updated_config['data'] = {}
        if 'dataset_path' not in updated_config['data']:
            updated_config['data']['dataset_path'] = ui_components.get('data_dir', 'data/preprocessed')
        
        # Log konfigurasi sebelum disimpan
        local_logger.debug(f"{ICONS['info']} Konfigurasi yang akan disimpan: {updated_config}")
        
        # Simpan konfigurasi ke drive melalui ConfigManager
        success = save_augmentation_config(updated_config)
        
        # Perbarui UI components
        if success:
            try:
                from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_config_to_ui
                map_config_to_ui(ui_components, updated_config)
                ui_components['config'] = updated_config
                
                local_logger.info(f"{ICONS['success']} Konfigurasi berhasil disinkronkan dengan drive")
            except Exception as e:
                local_logger.warning(f"{ICONS['warning']} Gagal update UI dari konfigurasi: {str(e)}")
                success = False
        else:
            local_logger.error(f"{ICONS['error']} Gagal menyinkronkan konfigurasi dengan drive")
        
        # Tampilkan pesan sukses
        if success:
            display(HTML('<div style="padding: 10px; background-color: #d4edda; color: #155724; border-radius: 5px;">' +
                         '<b>\u2705 Konfigurasi augmentasi berhasil disimpan!</b></div>'))
        else:
            display(HTML('<div style="padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px;">' +
                         '<b>\u274c Gagal menyimpan konfigurasi augmentasi!</b></div>'))
        
        return success
    except Exception as e:
        # Tampilkan pesan error
        display(HTML('<div style="padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 5px;">' +
                     f'<b>\u274c Gagal menyimpan konfigurasi augmentasi: {str(e)}</b></div>'))
        logger.error(f"{ICONS['error']} Error saat menyinkronkan konfigurasi: {str(e)}")
        return False
