"""
File: smartcash/ui/training_config/backbone/handlers/config_handlers.py
Deskripsi: Handler untuk operasi konfigurasi backbone model
"""

from typing import Dict, Any, Optional
import yaml
from pathlib import Path
import os
from smartcash.ui.utils.constants import ICONS

def load_default_config() -> Dict[str, Any]:
    """
    Load konfigurasi default untuk backbone model.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    # Import ModelManager untuk mendapatkan model yang dioptimalkan
    try:
        from smartcash.model.manager import ModelManager
        
        # Default config berdasarkan model yang dioptimalkan
        default_model_type = 'efficient_basic'
        
        # Pastikan model type ada dalam OPTIMIZED_MODELS
        if hasattr(ModelManager, 'OPTIMIZED_MODELS') and default_model_type in ModelManager.OPTIMIZED_MODELS:
            default_model_config = ModelManager.OPTIMIZED_MODELS[default_model_type]
        else:
            # Fallback jika model tidak ditemukan
            default_model_config = {
                'backbone': 'efficientnet_b4',
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False
            }
        
        return {
            'model': {
                'type': default_model_type,
                'backbone': default_model_config['backbone'],
                'backbone_pretrained': True,
                'backbone_freeze': False,
                'use_attention': default_model_config.get('use_attention', False),
                'use_residual': default_model_config.get('use_residual', False),
                'use_ciou': default_model_config.get('use_ciou', False)
            }
        }
    except Exception as e:
        print(f"{ICONS['error']} Error saat memuat konfigurasi default backbone: {str(e)}")
        # Fallback ke konfigurasi minimal
        return {
            'model': {
                'type': 'efficient_basic',
                'backbone': 'efficientnet_b4',
                'backbone_pretrained': True,
                'backbone_freeze': False,
                'use_attention': False,
                'use_residual': False,
                'use_ciou': False
            }
        }

def load_config() -> Dict[str, Any]:
    """
    Load konfigurasi backbone model dari file YAML.
    
    Returns:
        Dictionary berisi konfigurasi backbone model
    """
    try:
        # Path ke file konfigurasi model
        config_path = Path("config/model_config.yaml")
        if not config_path.exists():
            # Coba cari di lokasi alternatif
            alt_path = Path("../config/model_config.yaml")
            if alt_path.exists():
                config_path = alt_path
            else:
                return load_default_config()
        
        # Load konfigurasi dari file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Pastikan struktur konfigurasi benar
        if not config:
            config = {}
        if 'model' not in config:
            config['model'] = {}
        
        return config
    except Exception as e:
        print(f"{ICONS['error']} Error saat memuat konfigurasi backbone model: {str(e)}")
        # Kembalikan konfigurasi default
        return load_default_config()

def save_config(config: Dict[str, Any]) -> bool:
    """
    Simpan konfigurasi backbone model ke file YAML.
    
    Args:
        config: Konfigurasi yang akan disimpan
        
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    try:
        # Path ke file konfigurasi model
        config_path = Path("config/model_config.yaml")
        
        # Buat direktori jika belum ada
        os.makedirs(config_path.parent, exist_ok=True)
        
        # Simpan konfigurasi ke file
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Coba sinkronkan dengan drive jika tersedia
        try:
            from google.colab import drive
            # Cek apakah drive sudah di-mount
            if os.path.exists('/content/drive'):
                # Copy file ke drive
                drive_path = Path("/content/drive/MyDrive/smartcash/config/model_config.yaml")
                os.makedirs(drive_path.parent, exist_ok=True)
                with open(drive_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"{ICONS['success']} Konfigurasi berhasil disinkronkan dengan Google Drive")
        except ImportError:
            # Bukan di Google Colab, abaikan
            pass
        
        return True
    except Exception as e:
        print(f"{ICONS['error']} Error saat menyimpan konfigurasi backbone model: {str(e)}")
        return False

def get_config_manager_instance():
    """
    Dapatkan instance ConfigManager jika tersedia.
    
    Returns:
        Instance ConfigManager atau None jika tidak tersedia
    """
    try:
        from smartcash.common.config.manager import get_config_manager
        return get_config_manager()
    except Exception as e:
        print(f"{ICONS['warning']} Gagal mendapatkan instance ConfigManager: {str(e)}")
        return None

def save_config_with_manager(config: Dict[str, Any], ui_components: Dict[str, Any], logger=None) -> bool:
    """
    Simpan konfigurasi menggunakan ConfigManager dengan fallback.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
        logger: Logger untuk logging
        
    Returns:
        Boolean yang menunjukkan keberhasilan penyimpanan
    """
    success = False
    
    # Coba simpan dengan ConfigManager terlebih dahulu
    config_manager = get_config_manager_instance()
    if config_manager:
        try:
            # Pastikan UI components terdaftar untuk persistensi
            config_manager.register_ui_components('backbone_model', ui_components)
            # Simpan konfigurasi
            success = config_manager.save_module_config('model', config)
            if logger: logger.debug(f"{ICONS['info']} Konfigurasi disimpan melalui ConfigManager: {success}")
        except Exception as e:
            if logger: logger.warning(f"{ICONS['warning']} Gagal menyimpan dengan ConfigManager: {str(e)}")
            # Fallback ke metode save_config tradisional
            success = save_config(config)
    else:
        # Fallback ke metode save_config tradisional
        success = save_config(config)
    
    return success

def update_config_from_ui(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update konfigurasi dari nilai UI.
    
    Args:
        config: Konfigurasi aplikasi
        ui_components: Dictionary komponen UI
        
    Returns:
        Konfigurasi yang diupdate
    """
    # Pastikan struktur konfigurasi benar
    if not config:
        config = {}
    if 'model' not in config:
        config['model'] = {}
    
    # Simpan model_type yang dipilih
    if 'model_type' in ui_components:
        config['model']['type'] = ui_components['model_type'].value
            
    # Simpan backbone dan pengaturan dasar
    if 'backbone_type' in ui_components:
        config['model']['backbone'] = ui_components['backbone_type'].value
    
    # Sesuaikan fitur optimasi berdasarkan model yang dipilih
    model_type = config['model'].get('type', 'efficient_basic')
    
    # Import ModelManager untuk mendapatkan model yang dioptimalkan
    from smartcash.model.manager import ModelManager
    
    # Default semua fitur optimasi ke False
    config['model']['use_attention'] = False
    config['model']['use_residual'] = False
    config['model']['use_ciou'] = False
    
    # Jika model_type ada dalam OPTIMIZED_MODELS, gunakan fitur optimasi dari model tersebut
    if hasattr(ModelManager, 'OPTIMIZED_MODELS') and model_type in ModelManager.OPTIMIZED_MODELS:
        model_config = ModelManager.OPTIMIZED_MODELS[model_type]
        config['model']['use_attention'] = model_config.get('use_attention', False)
        config['model']['use_residual'] = model_config.get('use_residual', False)
        config['model']['use_ciou'] = model_config.get('use_ciou', False)
    
    return config
