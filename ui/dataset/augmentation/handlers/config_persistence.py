"""
File: smartcash/ui/dataset/augmentation/handlers/config_persistence.py
Deskripsi: Helper untuk persistensi konfigurasi augmentasi dengan ConfigManager
"""

from typing import Dict, Any, Optional, List, Union
import os
import yaml
from pathlib import Path
from smartcash.common.config.manager import get_config_manager
from smartcash.ui.utils.constants import ICONS

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
        logger = ui_components.get('logger')
        
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
                    if logger:
                        logger.warning(f"{ICONS['warning']} Gagal memuat konfigurasi: {str(e)}")
        
        # Update UI dari konfigurasi jika ada
        if config:
            try:
                from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_config_to_ui
                map_config_to_ui(ui_components, config)
                # Simpan referensi config di ui_components
                ui_components['config'] = config
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS['warning']} Gagal update UI dari konfigurasi: {str(e)}")
        
        # Log info jika tersedia logger
        if logger:
            logger.info(f"{ICONS['success']} UI components berhasil terdaftar untuk persistensi")
        
        return ui_components
    except Exception as e:
        if logger:
            logger.error(f"{ICONS['error']} Error saat memastikan persistensi UI: {str(e)}")
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
            return config
    except Exception:
        pass
    
    # Coba load dari file jika tidak ada di ConfigManager
    try:
        config_path = "configs/augmentation_config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            # Validasi konfigurasi
            if config is None:
                config = {}
            
            # Pastikan struktur konfigurasi benar
            if 'augmentation' not in config:
                config['augmentation'] = {}
            
            # Simpan ke ConfigManager untuk persistensi
            try:
                config_manager = get_config_manager_instance()
                config_manager.save_module_config('augmentation', config)
            except Exception:
                pass
            
            return config
    except Exception:
        pass
    
    # Jika tidak ada konfigurasi yang ditemukan, gunakan default
    if default_config:
        return default_config
    
    # Jika tidak ada default yang disediakan, gunakan default bawaan
    return get_default_augmentation_config()

def save_augmentation_config(config: Dict[str, Any]) -> bool:
    """Simpan konfigurasi augmentasi ke ConfigManager dan file lokal.
    
    Args:
        config: Konfigurasi yang akan disimpan
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Validasi konfigurasi sebelum disimpan
        if not config or not isinstance(config, dict):
            return False
            
        # Pastikan struktur konfigurasi benar
        if 'augmentation' not in config:
            config['augmentation'] = {}
        
        # Simpan ke ConfigManager
        config_manager = get_config_manager_instance()
        success = config_manager.save_module_config('augmentation', config)
        
        # Simpan juga ke file lokal untuk kompatibilitas
        try:
            config_path = "configs/augmentation_config.yaml"
            # Pastikan direktori ada
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            # Simpan ke file
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception:
            # Jika gagal menyimpan ke file lokal, tetap lanjutkan
            pass
        
        # Coba sinkronkan dengan Google Drive jika tersedia
        try:
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            if env_manager.is_drive_mounted:
                drive_config_path = str(env_manager.drive_path / 'configs' / Path(config_path).name)
                
                # Buat direktori jika belum ada
                os.makedirs(os.path.dirname(drive_config_path), exist_ok=True)
                
                # Salin file ke Google Drive
                with open(drive_config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
        except Exception:
            # Jika gagal menyinkronkan dengan drive, tetap lanjutkan
            pass
        
        return success
    except Exception:
        return False

def register_config_observer(callback: callable) -> None:
    """Register observer untuk notifikasi perubahan konfigurasi augmentasi."""
    config_manager = get_config_manager_instance()
    config_manager.register_observer('augmentation', callback)

def ensure_valid_aug_types(aug_types: Any) -> List[str]:
    """Pastikan aug_types selalu valid dengan validasi yang kuat."""
    default_aug_types = ['Combined (Recommended)']
    
    # Validasi tipe data
    if aug_types is None:
        return default_aug_types
    
    # Konversi ke list jika string
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
    """Dapatkan konfigurasi default untuk augmentasi.
    
    Returns:
        Dictionary konfigurasi default
    """
    return {
        'augmentation': {
            'prefix': 'aug',
            'factor': 2,
            'types': ['Combined (Recommended)'],
            'split': 'train',
            'balance_classes': True,
            'num_workers': 4,
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
        }
    }

def reset_config_to_default(ui_components: Dict[str, Any]) -> bool:
    """Reset konfigurasi ke default dan perbarui UI.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger')
        
        # Dapatkan konfigurasi default
        default_config = get_default_augmentation_config()
        
        # Validasi konfigurasi default
        try:
            from smartcash.ui.dataset.augmentation.handlers.config_validator import validate_augmentation_config
            validated_config = validate_augmentation_config(default_config)
            default_config = validated_config
        except Exception as e:
            if logger:
                logger.warning(f"{ICONS['warning']} Gagal validasi konfigurasi default: {str(e)}")
        
        # Simpan ke ConfigManager
        config_manager = get_config_manager_instance()
        success = config_manager.save_module_config('augmentation', default_config)
        
        # Simpan juga ke file lokal
        try:
            config_path = "configs/augmentation_config.yaml"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
        except Exception as e:
            if logger:
                logger.warning(f"{ICONS['warning']} Gagal menyimpan default ke file: {str(e)}")
        
        # Update UI dari konfigurasi default
        if success and ui_components:
            try:
                from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_config_to_ui
                map_config_to_ui(ui_components, default_config)
                ui_components['config'] = default_config
            except Exception as e:
                if logger:
                    logger.warning(f"{ICONS['warning']} Gagal update UI dari default: {str(e)}")
                success = False
        
        return success
    except Exception as e:
        if logger:
            logger.error(f"{ICONS['error']} Error saat reset konfigurasi: {str(e)}")
        return False

def sync_config_with_drive(ui_components: Dict[str, Any]) -> bool:
    """Sinkronisasi konfigurasi dengan file di drive.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Dapatkan logger jika tersedia
        logger = ui_components.get('logger')
        
        # Dapatkan konfigurasi dari UI
        from smartcash.ui.dataset.augmentation.handlers.config_mapper import map_ui_to_config
        updated_config = map_ui_to_config(ui_components, ui_components.get('config', {}))
        
        # Validasi konfigurasi sebelum disimpan
        if not updated_config or not isinstance(updated_config, dict):
            if logger:
                logger.warning(f"{ICONS['warning']} Konfigurasi tidak valid untuk disinkronkan")
            return False
            
        # Pastikan struktur konfigurasi benar
        if 'augmentation' not in updated_config:
            updated_config['augmentation'] = {}
        
        # Simpan konfigurasi ke drive melalui ConfigManager
        success = save_augmentation_config(updated_config)
        
        # Log info
        if success:
            if logger:
                logger.info(f"{ICONS['success']} Konfigurasi berhasil disinkronkan dengan drive")
            # Simpan kembali config yang diupdate ke ui_components
            ui_components['config'] = updated_config
        else:
            if logger:
                logger.error(f"{ICONS['error']} Gagal menyinkronkan konfigurasi dengan drive")
        
        return success
    except Exception as e:
        if logger:
            logger.error(f"{ICONS['error']} Error saat menyinkronkan konfigurasi: {str(e)}")
        return False
