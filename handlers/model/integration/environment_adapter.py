# File: smartcash/handlers/model/integration/environment_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan EnvironmentManager

from typing import Dict, Optional, Any, Union, Tuple
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.utils.environment_manager import EnvironmentManager

class EnvironmentAdapter:
    """
    Adapter untuk integrasi dengan EnvironmentManager.
    Menyediakan antarmuka yang konsisten untuk manajemen environment.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi environment adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("environment_adapter")
        
        # Inisialisasi environment manager
        self._env_manager = EnvironmentManager(logger=self.logger)
        
        self.logger.info(f"🔄 EnvironmentAdapter diinisialisasi (Colab: {self._env_manager.is_colab})")
    
    @property
    def is_colab(self) -> bool:
        """Cek apakah running di Google Colab."""
        return self._env_manager.is_colab
    
    @property
    def drive_path(self) -> Optional[Path]:
        """Dapatkan path Google Drive jika di-mount."""
        return self._env_manager.drive_path
    
    def mount_drive(
        self, 
        mount_path: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Mount Google Drive jika di Colab.
        
        Args:
            mount_path: Path untuk mount Drive (opsional)
            
        Returns:
            Tuple (success, message)
        """
        # Gunakan path dari config jika tidak ada
        if mount_path is None:
            mount_path = self.config.get('environment', {}).get('mount_path', '/content/drive/MyDrive/SmartCash')
            
        return self._env_manager.mount_drive(mount_path)
    
    def setup_project_environment(
        self, 
        use_drive: bool = True,
        create_symlinks: bool = True
    ) -> Dict[str, Any]:
        """
        Setup environment project lengkap.
        
        Args:
            use_drive: Gunakan Google Drive untuk storage jika di Colab
            create_symlinks: Buat symlinks ke Drive jika di Colab
            
        Returns:
            Dict dengan status hasil setup
        """
        result = {
            'is_colab': self._env_manager.is_colab,
            'drive_mounted': self._env_manager.is_drive_mounted,
            'directories': {},
            'symlinks': {}
        }
        
        # Setup directories
        result['directories'] = self._env_manager.setup_directories(use_drive=use_drive)
        
        # Buat symlinks jika di Colab
        if self._env_manager.is_colab and create_symlinks:
            result['symlinks'] = self._env_manager.create_symlinks()
        
        return result
    
    def get_path(self, relative_path: str) -> Path:
        """
        Dapatkan absolute path berdasarkan environment.
        
        Args:
            relative_path: Path relatif
            
        Returns:
            Path object untuk path yang diminta
        """
        return self._env_manager.get_path(relative_path)
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi sistem.
        
        Returns:
            Dict informasi sistem
        """
        return self._env_manager.get_system_info()
    
    def adjust_path_for_environment(self, path: Union[str, Path]) -> Path:
        """
        Sesuaikan path berdasarkan environment (Colab/local).
        
        Args:
            path: Path original
            
        Returns:
            Path yang disesuaikan
        """
        path_obj = Path(path)
        
        # Jika di Colab dan Google Drive di-mount, sesuaikan path
        if self._env_manager.is_colab and self._env_manager.is_drive_mounted:
            # Jika path tidak dimulai dengan drive_path, tambahkan
            drive_path = self._env_manager.drive_path
            if drive_path and not str(path_obj).startswith(str(drive_path)):
                # Periksa apakah path relatif atau absolut
                if path_obj.is_absolute():
                    # Dapatkan path relatif dari root
                    rel_path = path_obj.relative_to('/')
                    return drive_path / rel_path
                else:
                    return drive_path / path_obj
                    
        return path_obj