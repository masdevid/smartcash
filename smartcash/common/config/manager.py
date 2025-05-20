"""
File: smartcash/common/config/manager.py
Deskripsi: Manager konfigurasi dengan dukungan YAML, environment variables, dan dependency injection
"""

import copy
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, Callable, Tuple, List
import logging
import yaml

from smartcash.common.config.singleton import Singleton
from smartcash.common.config.base_manager import BaseConfigManager
from smartcash.common.config.module_manager import ModuleConfigManager
from smartcash.common.config.drive_manager import DriveConfigManager
from smartcash.common.config.dependency_manager import DependencyManager

# Type variable untuk dependency injection
T = TypeVar('T')

class ConfigManager(DriveConfigManager, DependencyManager):
    """
    Manager untuk konfigurasi aplikasi dengan dukungan untuk loading dari file, 
    environment variable overrides, dependency injection, dan sinkronisasi dengan Google Drive
    """
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager dengan base directory, file konfigurasi utama, dan prefix environment variable
        
        Args:
            base_dir: Direktori dasar
            config_file: File konfigurasi utama
            env_prefix: Prefix untuk environment variables
        """
        if base_dir is None:
            raise ValueError("base_dir must not be None. Please provide a valid base directory for configuration.")
            
        # Initialize logger first
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            
        # Inisialisasi DriveConfigManager
        DriveConfigManager.__init__(self, base_dir, config_file, env_prefix)
        
        # Inisialisasi DependencyManager
        DependencyManager.__init__(self)
        
        # Simpan config_file untuk referensi
        self._config_file = config_file

    def sync_config_with_drive(self, module_name: str) -> bool:
        """
        Sinkronisasi konfigurasi modul ke Google Drive dan pastikan persistensi antar lokal dan drive.
        Args:
            module_name (str): Nama modul konfigurasi (misal: 'training_strategy')
        Returns:
            bool: True jika sinkronisasi berhasil, False jika gagal
        """
        success, message = self.sync_to_drive(module_name)
        if self._logger:
            if success:
                self._logger.info(f"✅ Sinkronisasi konfigurasi '{module_name}' ke Google Drive berhasil: {message}")
            else:
                self._logger.warning(f"⚠️ Sinkronisasi konfigurasi '{module_name}' ke Google Drive gagal: {message}")
        return success

# Singleton instance
_config_manager = None

def get_default_config() -> Dict[str, Any]:
    """
    Dapatkan konfigurasi default untuk aplikasi.
    
    Returns:
        Dictionary berisi konfigurasi default
    """
    return {
        '_base_': 'base_config.yaml',
        'data': {
            'source': 'roboflow',
            'roboflow': {
                'api_key': '',
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3'
            },
            'split_ratios': {
                'train': 0.7,
                'valid': 0.15,
                'test': 0.15
            },
            'stratified_split': True,
            'random_seed': 42,
            'validation': {
                'enabled': True,
                'fix_issues': True,
                'move_invalid': True,
                'invalid_dir': 'data/invalid',
                'visualize_issues': False
            }
        },
        'dataset': {
            'backup': {
                'enabled': True,
                'dir': 'data/backup/dataset',
                'count': 3,
                'auto': False
            },
            'export': {
                'enabled': True,
                'formats': ['yolo', 'coco'],
                'dir': 'data/exported'
            },
            'import': {
                'allowed_formats': ['yolo', 'coco', 'voc'],
                'temp_dir': 'data/temp'
            }
        },
        'cache': {
            'enabled': True,
            'dir': '.cache/smartcash/dataset',
            'max_size_gb': 1.0,
            'ttl_hours': 24,
            'auto_cleanup': True
        }
    }

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance ConfigManager (singleton) dengan fallback untuk base_dir dan config_file.
    
    Args:
        base_dir: Direktori dasar (opsional, akan menggunakan fallback jika None)
        config_file: File konfigurasi utama (opsional, akan menggunakan fallback jika None)
        env_prefix: Prefix untuk environment variables
        
    Returns:
        Instance singleton ConfigManager
    """
    global _config_manager
    
    # Initialize logger dengan level ERROR saja
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)  # Set ke ERROR level
    
    try:
        # Cek apakah kita di Colab
        try:
            import google.colab
            from google.colab import userdata
            is_colab = True
        except ImportError:
            is_colab = False
            
        # Set default base_dir dan config_file berdasarkan environment
        if is_colab:
            # Di Colab, gunakan /content sebagai base_dir
            default_base_dir = '/content'
            default_config_file = str(Path(default_base_dir) / 'configs' / 'dataset_config.yaml')
        else:
            # Di lokal, gunakan root project
            default_base_dir = str(Path(__file__).resolve().parents[3])  # 3 levels up to reach project root
            default_config_file = str(Path(default_base_dir) / 'smartcash' / 'configs' / 'dataset_config.yaml')
            
        # Gunakan nilai default jika tidak disediakan
        base_dir = base_dir or default_base_dir
        config_file = config_file or default_config_file
        
        # Validasi base_dir dan config_file
        if not base_dir:
            raise ValueError("base_dir tidak boleh None atau kosong")
        if not config_file:
            raise ValueError("config_file tidak boleh None atau kosong")
            
        # Pastikan direktori configs ada
        config_dir = Path(config_file).parent
        if not config_dir.exists():
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Gagal membuat direktori configs: {str(e)}")
                
        # Pastikan file config ada dan load API_KEY dari Colab secret jika di Colab
        if not Path(config_file).exists():
            default_config = get_default_config()
            
            # Jika di Colab, coba load API_KEY dari secret
            if is_colab:
                try:
                    api_key = userdata.get('ROBOFLOW_API_KEY')
                    if api_key:
                        default_config['data']['roboflow']['api_key'] = api_key
                except Exception as e:
                    logger.error(f"❌ Gagal mengambil API_KEY dari Colab secret: {str(e)}")
            
            try:
                # Simpan default config
                with open(config_file, 'w') as f:
                    yaml.dump(default_config, f)
            except Exception as e:
                logger.error(f"❌ Gagal menyimpan default config: {str(e)}")
        else:
            # Jika file config sudah ada dan di Colab, update API_KEY dari secret
            if is_colab:
                try:
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    api_key = userdata.get('ROBOFLOW_API_KEY')
                    if api_key:
                        config['data']['roboflow']['api_key'] = api_key
                        with open(config_file, 'w') as f:
                            yaml.dump(config, f)
                except Exception as e:
                    logger.error(f"❌ Gagal update API_KEY dari Colab secret: {str(e)}")
                
        # Buat atau dapatkan instance ConfigManager
        if _config_manager is None:
            _config_manager = ConfigManager(base_dir, config_file, env_prefix)
            
        return _config_manager
        
    except Exception as e:
        logger.error(f"❌ Error saat membuat ConfigManager: {str(e)}")
        raise

# Tambahkan method get_instance sebagai staticmethod untuk kompatibilitas
ConfigManager.get_instance = staticmethod(get_config_manager)
