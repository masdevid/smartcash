"""
File: smartcash/common/environment.py
Deskripsi: Environment manager yang diperkecil, fokus hanya pada core functionality tanpa duplikasi
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def get_default_base_dir():
    """Dapatkan direktori dasar default untuk aplikasi."""
    return "/content" if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ else str(Path.home() / "SmartCash")

class EnvironmentManager:
    """Environment manager singleton untuk deteksi dan setup environment dasar."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None: 
            cls._instance = super(EnvironmentManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        if getattr(self, '_initialized', False): return
            
        self.logger = logger
        self._in_colab = self._detect_colab()
        self._drive_mounted = False
        self._drive_path = None
        self._base_dir = Path(base_dir) if base_dir else Path('/content') if self._in_colab else Path(os.getcwd())
        
        # Auto-detect drive tanpa verbose logging
        if self._in_colab: 
            self._detect_drive_silently()
            
        self._initialized = True
    
    @property
    def is_colab(self) -> bool: 
        return self._in_colab
    
    @property
    def base_dir(self) -> Path: 
        return self._base_dir
    
    @property
    def drive_path(self) -> Optional[Path]: 
        return self._drive_path
    
    @property
    def is_drive_mounted(self) -> bool: 
        return self._drive_mounted
    
    def _detect_colab(self) -> bool:
        """Deteksi lingkungan Google Colab tanpa import logging."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _detect_drive_silently(self) -> bool:
        """Deteksi drive tanpa verbose logging untuk menghindari spam."""
        drive_mount_point = Path('/content/drive/MyDrive')
        if drive_mount_point.exists():
            self._drive_mounted = True  
            self._drive_path = Path('/content')  # Gunakan /content sebagai base di Colab
            return True
        return False
    
    def mount_drive(self) -> Tuple[bool, str]:
        """Mount Google Drive dengan minimal logging."""
        if not self._in_colab:
            return False, "Bukan environment Colab"
        
        if self._drive_mounted: 
            return True, "Drive sudah terhubung"
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            self._drive_path = Path('/content')
            self._drive_mounted = True
            
            return True, "Drive berhasil terhubung"
        except Exception as e:
            return False, f"Error mounting drive: {str(e)}"
    
    def get_path(self, relative_path: str) -> Path:
        """Dapatkan path absolut berdasarkan environment."""
        return self._base_dir / relative_path
    
    def get_system_info(self) -> Dict[str, Any]:
        """Info sistem minimal tanpa verbose logging."""
        info = {
            'environment': 'Google Colab' if self._in_colab else 'Local',
            'base_directory': str(self._base_dir),
            'drive_mounted': self._drive_mounted,
            'python_version': sys.version.split()[0]
        }
        
        # GPU info minimal
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
        except ImportError:
            info['cuda_available'] = False
            
        return info

# Singleton instance
_environment_manager = None

def get_environment_manager(base_dir: Optional[str] = None, logger = None) -> EnvironmentManager:
    """Dapatkan instance singleton EnvironmentManager."""
    global _environment_manager
    if _environment_manager is None: 
        _environment_manager = EnvironmentManager(base_dir, logger)
    return _environment_manager