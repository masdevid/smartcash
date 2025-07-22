"""
File: smartcash/common/environment.py
Deskripsi: Unified environment manager with consolidated detection logic
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from functools import lru_cache

def get_default_base_dir():
    """Dapatkan direktori dasar default untuk aplikasi."""
    return "/content" if is_colab_environment() else str(Path.home() / "SmartCash")

@lru_cache(maxsize=1)
def is_colab_environment() -> bool:
    """
    Unified Colab environment detection with multiple indicators.
    
    This is the single source of truth for Colab detection across the entire codebase.
    All other modules should import and use this function instead of implementing their own.
    
    Returns:
        bool: True if running in Google Colab, False otherwise.
    """
    # Check environment variables (fastest check)
    env_indicators = ["COLAB_GPU", "COLAB_TPU_ADDR", "COLAB_RELEASE_TAG"]
    if any(os.environ.get(indicator) for indicator in env_indicators):
        return True
    
    # Check for google.colab module in sys.modules (already imported)
    if 'google.colab' in sys.modules:
        return True
    
    # Try import test (slower but most reliable)
    try:
        import google.colab
        return True
    except ImportError:
        pass
    
    # Check filesystem signatures (fallback)
    colab_filesystem_indicators = [
        Path('/content').exists(),
        Path('/usr/local/lib/python3.10/dist-packages/google').exists() or
        Path('/usr/local/lib/python3.11/dist-packages/google').exists() or
        Path('/usr/local/lib/python3.12/dist-packages/google').exists()
    ]
    
    return all(colab_filesystem_indicators)

# Legacy alias for backward compatibility
_is_colab_environment = is_colab_environment

def _is_drive_mounted() -> bool:
    """Deteksi apakah Google Drive sudah mounted dengan multiple checks."""
    if not _is_colab_environment():
        return False
        
    drive_paths = ['/content/drive/MyDrive', '/content/gdrive/MyDrive', '/content/drive/My Drive']
    return any(Path(p).exists() and Path(p).is_dir() and list(Path(p).iterdir()) for p in drive_paths)

def _get_drive_path() -> Optional[Path]:
    """Dapatkan path ke Google Drive yang benar."""
    if not _is_drive_mounted():
        return None
        
    for base in ['/content/drive/MyDrive', '/content/gdrive/MyDrive', '/content/drive/My Drive']:
        if Path(base).exists():
            smartcash_path = Path(base) / 'SmartCash'
            smartcash_path.mkdir(parents=True, exist_ok=True)
            return smartcash_path
    return None

class EnvironmentManager:
    """Fixed environment manager dengan path resolution yang benar"""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger=None):
        if getattr(self, '_initialized', False):
            return
            
        self.logger = logger
        self._in_colab = _is_colab_environment()
        # Don't check for drive mount during initialization
        self._drive_mounted = None
        self._drive_path = None
        self._base_dir = self._resolve_base_dir(base_dir)
        self._data_path = self._resolve_data_path()
        self._initialized = True
    
    def _resolve_base_dir(self, base_dir: Optional[str] = None) -> Path:
        """Resolve base directory dengan prioritas yang benar"""
        if base_dir:
            return Path(base_dir)
        
        # Only check drive path if we're in Colab and haven't checked yet
        if self._in_colab and self._drive_mounted is None:
            self.is_drive_mounted  # This will update _drive_path if needed
            
        if self._drive_path and self._drive_path.exists():
            return self._drive_path
        elif self._in_colab:
            return Path('/content')
        else:
            return Path(os.getcwd())
    
    def _resolve_data_path(self) -> Path:
        """Resolve data path dengan prioritas yang benar"""
        if self._in_colab and self._drive_mounted is None:
            self.is_drive_mounted  # This will update _drive_path if needed
            
        if self._drive_path and self._drive_path.exists():
            return self._drive_path / 'data'
        elif self._in_colab:
            return Path('/content/data')
        else:
            return self._base_dir / 'data'
    
    @property
    def is_colab(self) -> bool:
        """Check if running in Google Colab."""
        # Always check the current environment state
        self._in_colab = _is_colab_environment()
        return self._in_colab
        
    @property
    def is_drive_mounted(self) -> bool:
        """Check if Google Drive is mounted."""
        # Always check the current drive mount state
        self._drive_mounted = _is_drive_mounted()
        if self._drive_mounted and self._drive_path is None:
            self._drive_path = _get_drive_path()
        return self._drive_mounted
    
    @property
    def base_dir(self) -> Path:
        return self._base_dir
    
    @property 
    def drive_path(self) -> Optional[Path]:
        return self._drive_path
    
    @property
    def is_drive_mounted(self) -> bool:
        return self._drive_mounted
    
    def get_dataset_path(self) -> Path:
        """FIXED: Get dataset path yang benar"""
        return self._data_path
    
    def mount_drive(self) -> Tuple[bool, str]:
        """Mount Google Drive dengan enhanced error handling."""
        if not self._in_colab:
            return False, "Bukan environment Colab"
            
        if self._drive_mounted:
            return True, f"Drive sudah terhubung di: {self._drive_path}"
        
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            self._drive_mounted = _is_drive_mounted()
            self._drive_path = _get_drive_path() if self._drive_mounted else None
            self._data_path = self._resolve_data_path()
            
            if self._drive_mounted:
                return True, f"Drive berhasil terhubung di: {self._drive_path}"
            else:
                return False, "Drive mount command completed tapi tidak dapat diakses"
                
        except Exception as e:
            return False, f"Gagal mount drive: {str(e)}"
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get enhanced system info."""
        info = {
            'environment': 'Google Colab' if self._in_colab else 'Local',
            'base_directory': str(self._base_dir),
            'data_directory': str(self._data_path),
            'drive_mounted': self._drive_mounted,
            'drive_path': str(self._drive_path) if self._drive_path else None,
            'python_version': sys.version.split()[0]
        }
        
        # Add CUDA info
        try:
            import torch
            info.update({
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            })
        except ImportError:
            info['cuda_available'] = False
        
        # Add memory info
        try:
            import psutil
            memory = psutil.virtual_memory()
            info.update({
                'total_memory_gb': round(memory.total / (1024**3), 2),
                'available_memory_gb': round(memory.available / (1024**3), 2)
            })
        except ImportError:
            pass
            
        return info
    
    def refresh_drive_status(self) -> bool:
        """Refresh Drive mount status."""
        if not self._in_colab:
            return False
            
        old_status = self._drive_mounted
        self._drive_mounted = _is_drive_mounted()
        self._drive_path = _get_drive_path() if self._drive_mounted else None
        self._data_path = self._resolve_data_path()
        
        if old_status != self._drive_mounted and self.logger:
            self.logger.info(f"🔄 Drive status changed: {old_status} → {self._drive_mounted}")
            
        return self._drive_mounted

# Singleton instance
_environment_manager = None

def get_environment_manager(base_dir: Optional[str] = None, logger=None) -> EnvironmentManager:
    """Dapatkan singleton EnvironmentManager."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager(base_dir, logger)
    return _environment_manager