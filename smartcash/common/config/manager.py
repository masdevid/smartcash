"""
File: smartcash/common/config/manager.py
Deskripsi: Manager konfigurasi dengan struktur symlink yang benar
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
    Manager konfigurasi dengan struktur symlink yang benar untuk Colab
    """
    
    def __init__(self, base_dir: Optional[str] = None, config_file: Optional[str] = None, env_prefix: str = 'SMARTCASH_'):
        """
        Inisialisasi config manager
        
        Args:
            base_dir: Direktori dasar (default: /content untuk Colab)
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
        
        # Setup struktur direktori yang benar
        self.config_dir = self.base_dir / DEFAULT_CONFIG_DIR  # /content/configs
        self.repo_config_dir = Path('/content/smartcash/configs')  # Template configs dari repo
        self.drive_config_dir = Path(DRIVE_PATH) / DEFAULT_CONFIG_DIR  # Actual storage di Drive
        
        self._setup_config_structure()
    
    def _get_default_base_dir(self) -> Path:
        """
        Dapatkan direktori dasar default berdasarkan environment
        
        Returns:
            Path direktori dasar (/content untuk Colab, project root untuk local)
        """
        try:
            # Cek apakah kita di Colab
            import google.colab
            return Path(COLAB_PATH)
        except ImportError:
            # Jika bukan di Colab, gunakan directory relatif ke project root
            return Path(__file__).resolve().parents[3]
    
    def _setup_config_structure(self) -> None:
        """
        Setup struktur direktori konfigurasi yang benar dengan symlink
        """
        try:
            # Cek apakah kita di Colab
            import google.colab
            from google.colab import drive
            is_colab = True
        except ImportError:
            is_colab = False
            # Untuk local development, buat direktori biasa
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"📁 Setup direktori konfigurasi lokal: {self.config_dir}")
            return
        
        if not is_colab:
            return
            
        try:
            # Mount Google Drive jika belum
            if not Path('/content/drive').exists():
                drive.mount('/content/drive')
                self._logger.info("🔗 Google Drive berhasil di-mount")
            
            # 1. Pastikan direktori di Drive ada
            self.drive_config_dir.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"📁 Direktori Drive dibuat: {self.drive_config_dir}")
            
            # 2. Copy template configs dari repo ke Drive jika Drive kosong
            if self.repo_config_dir.exists() and not any(self.drive_config_dir.glob('*.yaml')):
                self._copy_config_templates_to_drive()
                self._logger.info(f"📋 Template config disalin dari repo ke Drive")
            
            # 3. Setup symlink /content/configs -> Drive
            self._setup_config_symlink()
            
        except Exception as e:
            self._logger.error(f"❌ Error setup config structure: {str(e)}")
            # Fallback: buat direktori lokal
            self.config_dir.mkdir(parents=True, exist_ok=True)
            self._logger.warning(f"⚠️ Fallback ke direktori lokal: {self.config_dir}")
    
    def _copy_config_templates_to_drive(self) -> None:
        """
        Copy template config dari repo ke Drive (hanya jika Drive kosong)
        """
        try:
            # Copy semua file .yaml dan .yml dari repo ke Drive
            for ext in ['.yaml', '.yml']:
                for template_file in self.repo_config_dir.glob(f'*{ext}'):
                    if template_file.is_file() and not template_file.name.startswith('.'):
                        drive_file = self.drive_config_dir / template_file.name
                        shutil.copy2(template_file, drive_file)
                        self._logger.debug(f"📋 Template {template_file.name} -> Drive")
                        
        except Exception as e:
            self._logger.error(f"❌ Error copy templates: {str(e)}")
    
    def _setup_config_symlink(self) -> None:
        """
        Setup symlink /content/configs -> /content/drive/MyDrive/SmartCash/configs
        """
        try:
            # Cek apakah symlink sudah ada dan valid
            if self.config_dir.is_symlink():
                # Cek apakah symlink menuju ke tempat yang benar
                current_target = self.config_dir.resolve()
                expected_target = self.drive_config_dir.resolve()
                
                if current_target == expected_target and self.config_dir.exists():
                    self._logger.info(f"✅ Symlink config sudah benar: {self.config_dir} -> {current_target}")
                    return
                else:
                    # Symlink salah atau rusak, hapus
                    os.unlink(self.config_dir)
                    self._logger.warning(f"🔧 Symlink config rusak, akan dibuat ulang")
            
            # Jika direktori biasa ada, hapus terlebih dahulu
            elif self.config_dir.exists():
                shutil.rmtree(self.config_dir)
                self._logger.warning(f"🗑️ Direktori config lokal dihapus untuk membuat symlink")
            
            # Buat symlink baru
            os.symlink(self.drive_config_dir, self.config_dir)
            self._logger.info(f"🔗 Symlink config dibuat: {self.config_dir} -> {self.drive_config_dir}")
            
            # Verifikasi symlink
            if self.config_dir.is_symlink() and self.config_dir.exists():
                self._logger.info(f"✅ Symlink config berhasil dan berfungsi")
            else:
                raise Exception("Symlink tidak berfungsi dengan benar")
                
        except Exception as e:
            self._logger.error(f"❌ Error setup symlink: {str(e)}")
            # Fallback: buat direktori lokal
            if not self.config_dir.exists():
                self.config_dir.mkdir(parents=True, exist_ok=True)
                self._logger.warning(f"⚠️ Fallback ke direktori lokal: {self.config_dir}")
    
    def get_config_path(self, config_name: str = None) -> Path:
        """
        Dapatkan path file konfigurasi (melalui symlink, otomatis ke Drive)
        
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
        
        # Return path melalui symlink (otomatis ke Drive jika di Colab)
        return self.config_dir / config_name
    
    def load_config(self, config_name: str = None) -> Dict[str, Any]:
        """
        Load konfigurasi (melalui symlink, otomatis dari Drive)
        
        Args:
            config_name: Nama konfigurasi (opsional)
            
        Returns:
            Dictionary konfigurasi
        """
        config_path = self.get_config_path(config_name)
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                # Cache config
                cache_key = config_name or self.config_file
                self.config_cache[cache_key] = copy.deepcopy(config)
                self._logger.debug(f"📖 Config loaded: {config_path}")
                return config
        except Exception as e:
            self._logger.error(f"❌ Error load config {config_path}: {str(e)}")
        
        return {}
    
    def save_config(self, config: Dict[str, Any], config_name: str = None) -> bool:
        """
        Simpan konfigurasi (melalui symlink, otomatis ke Drive)
        
        Args:
            config: Konfigurasi yang akan disimpan
            config_name: Nama konfigurasi (opsional)
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            config_path = self.get_config_path(config_name)
            
            # Pastikan direktori ada
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Simpan konfigurasi (otomatis ke Drive melalui symlink)
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            # Cache config
            cache_key = config_name or self.config_file
            self.config_cache[cache_key] = copy.deepcopy(config)
            
            # Notifikasi observer
            self._notify_observers(cache_key, config)
            
            # Log dengan info tentang symlink
            target_info = ""
            if config_path.parent.is_symlink():
                actual_target = config_path.parent.resolve()
                target_info = f" -> {actual_target}"
            
            self._logger.info(f"💾 Config saved: {config_path}{target_info}")
            return True
            
        except Exception as e:
            self._logger.error(f"❌ Error save config: {str(e)}")
            return False
    
    def save_module_config(self, module_name: str, config: Dict[str, Any]) -> bool:
        """
        Simpan konfigurasi modul dengan struktur yang benar
        
        Args:
            module_name: Nama modul (e.g., 'augmentation')
            config: Konfigurasi modul
            
        Returns:
            True jika berhasil, False jika gagal
        """
        # Buat struktur konfigurasi dengan nama modul sebagai root
        full_config = {module_name: config}
        
        # Nama file konfigurasi
        config_filename = f"{module_name}_config.yaml"
        
        result = self.save_config(full_config, config_filename)
        
        if result:
            self._logger.info(f"📝 Module config saved: {module_name}")
        
        return result
    
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
    
    def get_module_config(self, module_name: str, reload: bool = False) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi modul tertentu
        
        Args:
            module_name: Nama modul (e.g., 'augmentation')
            reload: Reload dari disk meskipun ada di cache
            
        Returns:
            Dictionary konfigurasi modul
        """
        config_filename = f"{module_name}_config.yaml"
        full_config = self.get_config(config_filename, reload)
        
        # Return bagian modul dari konfigurasi
        return full_config.get(module_name, {})
    
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
        if config_name in self.observers and callback in self.observers[config_name]:
            self.observers[config_name].remove(callback)
    
    def _notify_observers(self, config_name: str, config: Dict[str, Any]) -> None:
        """
        Notifikasi observer tentang perubahan konfigurasi
        
        Args:
            config_name: Nama konfigurasi
            config: Konfigurasi yang berubah
        """
        if config_name in self.observers:
            for callback in self.observers[config_name]:
                try:
                    callback(config)
                except Exception as e:
                    self._logger.error(f"❌ Error notify observer: {str(e)}")
    
    def get_available_configs(self, ignored_configs: List[str] = None) -> List[str]:
        """
        Dapatkan daftar konfigurasi yang tersedia
        
        Args:
            ignored_configs: List nama konfigurasi yang diabaikan
            
        Returns:
            List nama konfigurasi
        """
        config_files = []
        
        # Pastikan direktori konfigurasi ada
        if not self.config_dir.exists():
            self._logger.warning(f"📁 Direktori konfigurasi tidak ditemukan: {self.config_dir}")
            return []
        
        try:
            # Cek semua file .yaml dan .yml
            for ext in ['.yaml', '.yml']:
                config_files.extend([f.name for f in self.config_dir.glob(f'*{ext}')])
        except Exception as e:
            self._logger.error(f"❌ Error list configs: {str(e)}")
            return []
        
        # Hilangkan ekstensi dan suffix _config
        configs = [f.replace('_config.yaml', '').replace('_config.yml', '').replace('.yaml', '').replace('.yml', '') 
                for f in config_files]
        
        return configs
    
    def is_symlink_active(self) -> bool:
        """
        Cek apakah symlink config aktif dan berfungsi
        
        Returns:
            True jika symlink aktif dan valid
        """
        return self.config_dir.is_symlink() and self.config_dir.exists()
    
    def get_actual_config_location(self) -> str:
        """
        Dapatkan lokasi aktual penyimpanan config
        
        Returns:
            String path lokasi aktual
        """
        if self.is_symlink_active():
            return f"Google Drive via symlink: {self.config_dir.resolve()}"
        else:
            return f"Local directory: {self.config_dir}"


# Dictionary untuk menyimpan instance singleton
_INSTANCE = None

def get_config_manager(base_dir=None, config_file=None, env_prefix='SMARTCASH_'):
    """
    Dapatkan instance SimpleConfigManager (singleton)
    
    Args:
        base_dir: Direktori dasar (default: /content untuk Colab)
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
            logger.error(f"❌ Error create SimpleConfigManager: {str(e)}")
            raise
    
    return _INSTANCE

# Tambahkan method get_instance sebagai static method untuk kompatibilitas
SimpleConfigManager.get_instance = staticmethod(get_config_manager)