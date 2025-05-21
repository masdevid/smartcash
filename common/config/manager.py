"""
File: smartcash/common/config/manager.py
Deskripsi: Manager konfigurasi yang disederhanakan dengan dukungan symlink untuk Google Colab
"""

import os
import copy
import shutil
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, TypeVar, Callable, Tuple, List

from smartcash.common.constants.core import DEFAULT_CONFIG_DIR, APP_NAME
from smartcash.common.constants.paths import COLAB_PATH, DRIVE_PATH

# Type variable untuk dependency injection
T = TypeVar('T')

class SimpleConfigManager:
    """
    Manager konfigurasi yang disederhanakan dengan dukungan symlink untuk Google Colab
    """
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager
        
        Args:
            base_dir: Direktori dasar (default: project root)
            config_file: File konfigurasi utama (default: base_config.yaml)
            env_prefix: Prefix untuk environment variables
        """
        # Initialize logger first
        self._logger = logging.getLogger(__name__)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
        
        self.env_prefix = env_prefix
        
        if base_dir is None:
            self.base_dir = self._get_default_base_dir()
        else:
            self.base_dir = Path(base_dir)
            
        self.config_file = config_file or "base_config.yaml"
        
        # Dictionary untuk menyimpan konfigurasi modul (cache)
        self.config_cache = {}
        
        # Dictionary untuk menyimpan observer (untuk notifikasi perubahan config)
        self.observers = {}
        
        # Setup config directory dan connection ke Google Drive jika di Colab
        self.config_dir = self.base_dir / DEFAULT_CONFIG_DIR
        self.drive_config_dir = None
        self._setup_config_directory()
    
    def _get_default_base_dir(self) -> Path:
        """
        Dapatkan direktori dasar default berdasarkan environment
        
        Returns:
            Path direktori dasar
        """
        try:
            # Cek apakah kita di Colab
            import google.colab
            return Path(COLAB_PATH)
        except ImportError:
            # Jika bukan di Colab, gunakan directory relatif ke project root
            return Path(__file__).resolve().parents[3]
    
    def _setup_config_directory(self) -> None:
        """
        Setup direktori konfigurasi dengan symlink ke Google Drive jika di Colab
        """
        # Pastikan direktori config lokal ada
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Cek apakah kita di Colab
            import google.colab
            from google.colab import drive
            is_colab = True
        except ImportError:
            is_colab = False
        
        if is_colab:
            try:
                # Mount Google Drive jika belum
                if not Path('/content/drive').exists():
                    drive.mount('/content/drive')
                    self._logger.info("Google Drive berhasil di-mount")
                
                # Pastikan direktori konfigurasi ada di Google Drive
                self.drive_config_dir = Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR
                self.drive_config_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy file konfigurasi dari repo ke Drive jika Drive kosong
                repo_config_dir = Path('/content/smartcash/configs')
                if repo_config_dir.exists() and not any(self.drive_config_dir.glob('*.yaml')):
                    self._copy_config_files(repo_config_dir, self.drive_config_dir)
                    self._logger.info(f"File konfigurasi disalin dari {repo_config_dir} ke {self.drive_config_dir}")
                
                # Jika config directory bukan symlink, buat symlink ke drive
                if not self.config_dir.is_symlink():
                    # Hapus direktori lokal jika sudah ada
                    if self.config_dir.exists():
                        shutil.rmtree(self.config_dir)
                    
                    # Buat symlink dari Drive ke local
                    os.symlink(self.drive_config_dir, self.config_dir)
                    self._logger.info(f"Symlink dibuat dari {self.drive_config_dir} ke {self.config_dir}")
            except Exception as e:
                self._logger.warning(f"Gagal setup Google Drive symlink: {str(e)}")
    
    def _copy_config_files(self, src_path: Path, dst_path: Path) -> None:
        """
        Copy file konfigurasi dari satu lokasi ke lokasi lain
        
        Args:
            src_path: Path sumber
            dst_path: Path tujuan
        """
        try:
            # Pastikan direktori tujuan ada
            dst_path.mkdir(parents=True, exist_ok=True)
            
            # Copy semua file .yaml dan .yml
            for ext in ['.yaml', '.yml']:
                for src_file in src_path.glob(f'*{ext}'):
                    if src_file.is_file() and not src_file.name.startswith('.'):
                        dst_file = dst_path / src_file.name
                        shutil.copy2(src_file, dst_file)
                        self._logger.debug(f"File {src_file.name} disalin ke {dst_path}")
        except Exception as e:
            self._logger.error(f"Error saat menyalin file konfigurasi: {str(e)}")
    
    def get_config_path(self, config_name: str = None) -> Path:
        """
        Dapatkan path file konfigurasi
        
        Args:
            config_name: Nama konfigurasi (opsional)
            
        Returns:
            Path ke file konfigurasi
        """
        if config_name is None:
            config_name = self.config_file
        
        # Jika bukan diakhiri dengan .yaml atau .yml, tambahkan .yaml
        if not (config_name.endswith('.yaml') or config_name.endswith('.yml')):
            config_name = f"{config_name}_config.yaml"
        
        return self.config_dir / config_name
    
    def get_drive_config_path(self, config_name: str = None) -> Optional[Path]:
        """
        Dapatkan path file konfigurasi di Google Drive
        
        Args:
            config_name: Nama konfigurasi (opsional)
            
        Returns:
            Path ke file konfigurasi di Drive atau None jika tidak di Colab
        """
        if self.drive_config_dir is None:
            return None
            
        if config_name is None:
            config_name = self.config_file
        
        # Jika bukan diakhiri dengan .yaml atau .yml, tambahkan .yaml
        if not (config_name.endswith('.yaml') or config_name.endswith('.yml')):
            config_name = f"{config_name}_config.yaml"
        
        return self.drive_config_dir / config_name
    
    def load_config(self, config_name: str = None) -> Dict[str, Any]:
        """
        Load konfigurasi
        
        Args:
            config_name: Nama konfigurasi (opsional)
            
        Returns:
            Dictionary konfigurasi
        """
        config_path = self.get_config_path(config_name)
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                # Cache config
                cache_key = config_name or self.config_file
                self.config_cache[cache_key] = copy.deepcopy(config)
                return config
        except Exception as e:
            self._logger.error(f"Error saat memuat konfigurasi {config_path}: {str(e)}")
        
        return {}
    
    def save_config(self, config: Dict[str, Any], config_name: str = None) -> bool:
        """
        Simpan konfigurasi
        
        Args:
            config: Konfigurasi yang akan disimpan
            config_name: Nama konfigurasi (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Gunakan path Drive jika tersedia, jika tidak gunakan path lokal
        config_path = self.get_drive_config_path(config_name) or self.get_config_path(config_name)
        
        try:
            # Pastikan direktori ada
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simpan ke file
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            
            # Cache config
            cache_key = config_name or self.config_file
            self.config_cache[cache_key] = copy.deepcopy(config)
            
            # Notifikasi observer
            self._notify_observers(cache_key, config)
            
            self._logger.info(f"Konfigurasi berhasil disimpan ke {config_path}")
            return True
        except Exception as e:
            self._logger.error(f"Error saat menyimpan konfigurasi {config_path}: {str(e)}")
            return False
    
    def get_config(self, config_name: str = None, reload: bool = False) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi (dari cache atau load dari disk)
        
        Args:
            config_name: Nama konfigurasi (opsional)
            reload: Reload dari disk meskipun ada di cache
            
        Returns:
            Dictionary konfigurasi
        """
        cache_key = config_name or self.config_file
        
        # Jika perlu reload atau belum ada di cache, load dari disk
        if reload or cache_key not in self.config_cache:
            return self.load_config(config_name)
        
        # Gunakan cache
        return copy.deepcopy(self.config_cache.get(cache_key, {}))
    
    def update_config(self, update_dict: Dict[str, Any], config_name: str = None) -> bool:
        """
        Update konfigurasi (load, update, save)
        
        Args:
            update_dict: Dictionary dengan update
            config_name: Nama konfigurasi (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Load konfigurasi
        config = self.get_config(config_name, reload=True)
        
        # Update konfigurasi
        config.update(update_dict)
        
        # Simpan konfigurasi
        return self.save_config(config, config_name)
    
    def register_observer(self, config_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register observer untuk notifikasi perubahan konfigurasi
        
        Args:
            config_name: Nama konfigurasi
            callback: Fungsi callback yang dipanggil saat konfigurasi berubah
        """
        # Tambahkan observer ke dictionary
        if config_name not in self.observers:
            self.observers[config_name] = set()
        
        self.observers[config_name].add(callback)
    
    def unregister_observer(self, config_name: str, callback: Callable) -> None:
        """
        Unregister observer
        
        Args:
            config_name: Nama konfigurasi
            callback: Fungsi callback yang akan di-unregister
        """
        # Hapus observer dari dictionary
        if config_name in self.observers and callback in self.observers[config_name]:
            self.observers[config_name].remove(callback)
    
    def _notify_observers(self, config_name: str, config: Dict[str, Any]) -> None:
        """
        Notifikasi observer tentang perubahan konfigurasi
        
        Args:
            config_name: Nama konfigurasi
            config: Konfigurasi yang berubah
        """
        # Notifikasi observer
        if config_name in self.observers:
            for callback in self.observers[config_name]:
                try:
                    callback(config)
                except Exception as e:
                    self._logger.error(f"Error saat memanggil observer: {str(e)}")
    
    def get_available_configs(self, ignored_configs: List[str] = None) -> List[str]:
        """
        Dapatkan daftar konfigurasi yang tersedia
        
        Args:
            ignored_configs: List nama konfigurasi yang diabaikan (tidak perlu dilaporkan jika tidak ada)
            
        Returns:
            List nama konfigurasi
        """
        # Dapatkan semua file .yaml dan .yml di direktori konfigurasi
        config_files = []
        if self.config_dir.exists():
            for ext in ['.yaml', '.yml']:
                config_files.extend([f.name for f in self.config_dir.glob(f'*{ext}')])
        
        # Hilangkan ekstensi
        configs = [f.replace('_config.yaml', '').replace('_config.yml', '').replace('.yaml', '').replace('.yml', '') 
                for f in config_files]
        
        # Filter konfigurasi yang diabaikan
        if ignored_configs:
            # Jangan log error untuk konfigurasi yang diabaikan
            for config in ignored_configs:
                config_path = self.get_config_path(config)
                if not config_path.exists():
                    # Hapus log error untuk file yang memang tidak ada
                    pass
        
        return configs
                
    # Metode untuk kompatibilitas dengan kode lama
    def get_module_config(self, config_name: str = None, reload: bool = False) -> Dict[str, Any]:
        """
        Alias untuk get_config, untuk kompatibilitas dengan ConfigManager lama
        
        Args:
            config_name: Nama konfigurasi (opsional)
            reload: Reload dari disk meskipun ada di cache
            
        Returns:
            Dictionary konfigurasi
        """
        return self.get_config(config_name, reload)
        
    def save_module_config(self, config_name: str, config: Dict[str, Any]) -> bool:
        """
        Alias untuk save_config, untuk kompatibilitas dengan ConfigManager lama
        
        Args:
            config_name: Nama konfigurasi
            config: Konfigurasi yang akan disimpan
            
        Returns:
            True jika berhasil, False jika gagal
        """
        return self.save_config(config, config_name)


# Dictionary untuk menyimpan instance singleton
_INSTANCE = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance SimpleConfigManager (singleton)
    
    Args:
        base_dir: Direktori dasar (default: project root)
        config_file: File konfigurasi (default: base_config.yaml)
        env_prefix: Prefix untuk environment variables
        
    Returns:
        Instance singleton SimpleConfigManager
    """
    global _INSTANCE
    
    # Jika instance belum ada, buat instance baru
    if _INSTANCE is None:
        try:
            _INSTANCE = SimpleConfigManager(base_dir, config_file, env_prefix)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error saat membuat SimpleConfigManager: {str(e)}")
            raise
    
    return _INSTANCE

# Tambahkan method get_instance sebagai static method
SimpleConfigManager.get_instance = staticmethod(get_config_manager)
