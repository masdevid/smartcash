"""
File: smartcash/common/config/module_manager.py
Deskripsi: Pengelolaan konfigurasi modul dengan dukungan persistensi
"""

import copy
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Tuple

from smartcash.common.io import (
    load_config,
    save_config,
    ensure_dir
)

from smartcash.common.config.base_manager import BaseConfigManager

class ModuleConfigManager(BaseConfigManager):
    """
    Pengelolaan konfigurasi modul dengan dukungan persistensi dan observer pattern
    """
    
    def __init__(self, *args, **kwargs):
        """
        Inisialisasi module config manager
        """
        super().__init__(*args, **kwargs)
        
        # Dictionary untuk menyimpan observer
        self.observers = {}
        
        # Dictionary untuk menyimpan referensi UI components
        self.ui_components = {}
    
    def get_module_config(self, module_name: str, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi untuk modul tertentu.
        
        Args:
            module_name: Nama modul
            default_config: Konfigurasi default jika tidak ada yang tersimpan
            
        Returns:
            Dictionary konfigurasi modul
        """
        # Jika sudah ada di cache, gunakan cache
        if module_name in self.module_configs:
            return copy.deepcopy(self.module_configs[module_name])
        
        # Jika belum ada di cache, coba load dari file
        config_path = self._get_module_config_path(module_name)
        
        try:
            if config_path.exists():
                # Load konfigurasi dari file
                config = load_config(config_path, {})
                
                # Simpan ke cache
                self.module_configs[module_name] = config
                
                return copy.deepcopy(config)
        except Exception as e:
            if self._logger:
                self._logger.critical(f"Error saat memuat konfigurasi {module_name}: {str(e)}")
        
        # Jika gagal load atau file tidak ada, gunakan default
        if default_config:
            # Simpan default ke cache
            self.module_configs[module_name] = copy.deepcopy(default_config)
            
            return copy.deepcopy(default_config)
        
        # Jika tidak ada default, kembalikan empty dict
        return {}
    
    def save_module_config(self, module_name: str, config: Dict[str, Any], create_dirs: bool = True) -> bool:
        """
        Simpan konfigurasi untuk modul tertentu.
        
        Args:
            module_name: Nama modul
            config: Dictionary konfigurasi
            create_dirs: Buat direktori jika belum ada
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            # Buat deep copy untuk menghindari perubahan pada config asli
            config_copy = copy.deepcopy(config)
            
            # Simpan ke cache
            self.module_configs[module_name] = config_copy
            
            # Simpan ke file
            config_path = self._get_module_config_path(module_name)
            
            # Buat direktori jika belum ada
            if create_dirs and not config_path.parent.exists():
                ensure_dir(config_path.parent)
            
            # Simpan konfigurasi
            save_config(config_path, config_copy)
            
            # Notifikasi observer
            self.notify_observers(module_name, config_copy)
            
            return True
        except Exception as e:
            if self._logger:
                self._logger.critical(f"Error saat menyimpan konfigurasi {module_name}: {str(e)}")
            return False
    
    def _get_module_config_path(self, module_name: str) -> Path:
        """
        Dapatkan path file konfigurasi untuk modul tertentu.
        
        Args:
            module_name: Nama modul
            
        Returns:
            Path file konfigurasi
        """
        # Gunakan nama modul sebagai nama file dengan ekstensi yaml
        filename = f"{module_name}_config.yaml"
        
        # Gunakan direktori configs/ di direktori saat ini
        return self.config_dir / filename
    
    # ========== Metode untuk UI components persistence ==========
    
    def register_ui_components(self, module_name: str, ui_components: Dict[str, Any]) -> None:
        """
        Register UI components untuk persistensi.
        
        Args:
            module_name: Nama modul
            ui_components: Dictionary komponen UI
        """
        self.ui_components[module_name] = ui_components
    
    def get_ui_components(self, module_name: str) -> Dict[str, Any]:
        """
        Dapatkan UI components yang tersimpan.
        
        Args:
            module_name: Nama modul
            
        Returns:
            Dictionary komponen UI atau empty dict jika tidak ditemukan
        """
        return self.ui_components.get(module_name, {})
    
    # ========== Metode untuk observer pattern ==========
    
    def register_observer(self, module_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register observer untuk notifikasi perubahan konfigurasi.
        
        Args:
            module_name: Nama modul
            callback: Fungsi callback yang akan dipanggil saat konfigurasi berubah
        """
        if module_name not in self.observers:
            self.observers[module_name] = []
        
        self.observers[module_name].append(callback)
    
    def unregister_observer(self, module_name: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Unregister observer.
        
        Args:
            module_name: Nama modul
            callback: Fungsi callback yang akan dihapus
        """
        if module_name in self.observers and callback in self.observers[module_name]:
            self.observers[module_name].remove(callback)
    
    def notify_observers(self, module_name: str, config: Dict[str, Any]) -> None:
        """
        Notifikasi semua observer tentang perubahan konfigurasi.
        
        Args:
            module_name: Nama modul
            config: Konfigurasi terbaru
        """
        if module_name in self.observers:
            for callback in self.observers[module_name]:
                try:
                    callback(copy.deepcopy(config))
                except Exception as e:
                    if self._logger:
                        self._logger.critical(f"Error saat memanggil observer {module_name}: {str(e)}")
    
    def ensure_ui_persistence(self, module_name: str, config: Dict[str, Any]) -> None:
        """
        Memastikan persistensi UI dengan menyimpan konfigurasi dan notifikasi observer.
        
        Args:
            module_name: Nama modul
            config: Konfigurasi terbaru
        """
        # Simpan ke cache
        self.module_configs[module_name] = copy.deepcopy(config)
        
        # Notifikasi observer
        self.notify_observers(module_name, config)
