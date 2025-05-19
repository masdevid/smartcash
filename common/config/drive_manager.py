"""
File: smartcash/common/config/drive_manager.py
Deskripsi: Pengelolaan sinkronisasi konfigurasi dengan Google Drive
"""

import copy
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from smartcash.common.config.module_manager import ModuleConfigManager

class DriveConfigManager(ModuleConfigManager):
    """
    Pengelolaan sinkronisasi konfigurasi dengan Google Drive
    """
    
    def __init__(self, *args, **kwargs):
        """
        Inisialisasi drive config manager
        """
        super().__init__(*args, **kwargs)
    
    def sync_with_drive(self, config_file: str, sync_strategy: str = 'drive_priority') -> Tuple[bool, str, Dict[str, Any]]:
        """
        Sinkronisasi file konfigurasi dengan Google Drive.
        
        Args:
            config_file: Path ke file konfigurasi
            sync_strategy: Strategi sinkronisasi ('merge', 'drive_priority', 'local_priority')
            
        Returns:
            Tuple (success, message, merged_config)
        """
        try:
            from smartcash.common.config.sync import sync_config_with_drive
            
            # Resolve path konfigurasi
            config_path = self._resolve_config_path(config_file)
            
            # Sinkronisasi dengan Google Drive
            success, message, merged_config = sync_config_with_drive(
                config_path, 
                strategy=sync_strategy
            )
            
            # Jika berhasil, update konfigurasi
            if success and merged_config:
                if config_file == self.config_file:
                    self.config = merged_config
                
            return success, message, merged_config
        except ImportError:
            if self._logger:
                self._logger.critical("Module sync tidak tersedia")
            return False, "Module sync tidak tersedia", {}
        except Exception as e:
            if self._logger:
                self._logger.critical(f"Error saat sinkronisasi dengan Drive: {str(e)}")
            return False, str(e), {}
    
    def sync_to_drive(self, module_name: str) -> Tuple[bool, str]:
        """
        Sinkronisasi konfigurasi modul dengan Google Drive.
        
        Args:
            module_name: Nama modul yang akan disinkronkan
            
        Returns:
            Tuple (success, message)
        """
        try:
            # Import fungsi upload_config_to_drive
            from smartcash.common.config.sync import upload_config_to_drive
            
            # Dapatkan path file konfigurasi
            config_path = self._get_module_config_path(module_name)
            
            # Jika file tidak ada, simpan konfigurasi terlebih dahulu
            if not config_path.exists() and module_name in self.module_configs:
                self.save_module_config(module_name, self.module_configs[module_name])
            
            # Jika file masih tidak ada, kembalikan error
            if not config_path.exists():
                return False, f"File konfigurasi {module_name} tidak ditemukan"
            
            # Upload ke Google Drive
            success, message = upload_config_to_drive(config_path)
            
            return success, message
        except ImportError:
            return False, "Module sync tidak tersedia"
        except Exception as e:
            if self._logger:
                self._logger.critical(f"Error saat sinkronisasi {module_name} ke Drive: {str(e)}")
            return False, str(e)
    
    def use_drive_as_source_of_truth(self) -> None:
        """
        Sinkronisasi semua konfigurasi dengan Drive sebagai sumber kebenaran.
        """
        try:
            from smartcash.common.config.sync import sync_all_configs
            
            # Sinkronisasi semua konfigurasi
            sync_all_configs(strategy='drive_priority')
            
            # Reload konfigurasi dari file
            for module_name in list(self.module_configs.keys()):
                config_path = self._get_module_config_path(module_name)
                
                if config_path.exists():
                    try:
                        # Load konfigurasi dari file
                        config = self._load_config_file(config_path)
                        
                        # Update cache
                        self.module_configs[module_name] = config
                        
                        # Notifikasi observer
                        self.notify_observers(module_name, config)
                    except Exception as e:
                        if self._logger:
                            self._logger.critical(f"Error saat reload konfigurasi {module_name}: {str(e)}")
        except ImportError:
            if self._logger:
                self._logger.critical("Module sync tidak tersedia")
        except Exception as e:
            if self._logger:
                self._logger.critical(f"Error saat sinkronisasi dengan Drive: {str(e)}")
    
    def get_drive_config_path(self, config_file: str = None) -> str:
        """
        Dapatkan path konfigurasi di Google Drive.
        
        Args:
            config_file: Path ke file konfigurasi lokal
            
        Returns:
            Path konfigurasi di Google Drive
        """
        try:
            from smartcash.common.config.sync import get_drive_config_path
            
            # Jika config_file tidak disediakan, gunakan config_file dari instance
            if not config_file and hasattr(self, 'config_file'):
                config_file = self.config_file
            
            # Jika masih tidak ada, kembalikan None
            if not config_file:
                return None
            
            # Resolve path konfigurasi
            config_path = self._resolve_config_path(config_file)
            
            # Dapatkan path di Google Drive
            return get_drive_config_path(config_path)
        except ImportError:
            return None
        except Exception:
            return None
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """
        Load konfigurasi dari file.
        
        Args:
            config_path: Path ke file konfigurasi
            
        Returns:
            Dictionary konfigurasi
            
        Raises:
            FileNotFoundError: Jika file tidak ditemukan
            Exception: Jika terjadi error saat loading
        """
        from smartcash.common.io import load_config
        
        if not config_path.exists():
            raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
        
        return load_config(config_path, {})
