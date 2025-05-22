"""
File: smartcash/common/environment.py
Deskripsi: Environment manager dengan logging minimal dan tanggung jawab yang jelas
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def get_default_base_dir():
    """Dapatkan direktori dasar default untuk aplikasi"""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

class EnvironmentManager:
    """Environment manager dengan logging minimal - fokus pada detection dan basic setup"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EnvironmentManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger=None):
        if getattr(self, '_initialized', False):
            return
            
        self._in_colab = self._detect_colab()
        self._drive_mounted = False
        self._drive_path = None
        self.logger = logger  # Optional logger, tidak wajib
        
        # Set base directory
        self._base_dir = (
            Path(base_dir) if base_dir 
            else Path('/content') if self._in_colab 
            else Path(os.getcwd())
        )
        
        # Auto-detect drive tanpa logging berlebihan
        if self._in_colab:
            self._detect_drive_silent()
            
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
        """Deteksi Colab tanpa import berlebihan"""
        return bool(os.environ.get("COLAB_GPU")) or bool(os.environ.get("COLAB_TPU_ADDR"))
    
    def _detect_drive_silent(self) -> bool:
        """Deteksi drive tanpa logging"""
        drive_path = Path('/content/drive/MyDrive')
        if drive_path.exists():
            self._drive_mounted = True
            self._drive_path = drive_path / 'SmartCash'
            return True
        return False
    
    def mount_drive(self) -> Tuple[bool, str]:
        """Mount Google Drive dengan minimal logging"""
        if not self._in_colab:
            return False, "Bukan environment Colab"
        
        if self._drive_mounted:
            return True, "Drive sudah terhubung"
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Set drive path
            self._drive_path = Path('/content/drive/MyDrive/SmartCash')
            self._drive_path.mkdir(parents=True, exist_ok=True)
            self._drive_mounted = True
            
            return True, "Drive berhasil terhubung"
        except Exception as e:
            return False, f"Gagal mount drive: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system info minimal"""
        info = {
            'environment': 'Google Colab' if self._in_colab else 'Local',
            'base_directory': str(self._base_dir),
            'drive_mounted': self._drive_mounted,
            'python_version': sys.version.split()[0]
        }
        
        # Deteksi GPU sederhana
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
        except ImportError:
            info['cuda_available'] = False
            
        return info

# Singleton instance
_environment_manager = None

def get_environment_manager(base_dir: Optional[str] = None, logger=None) -> EnvironmentManager:
    """Dapatkan singleton EnvironmentManager"""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager(base_dir, logger)
    return _environment_manager