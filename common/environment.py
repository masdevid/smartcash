"""
File: smartcash/common/environment.py
Deskripsi: Enhanced environment manager dengan deteksi Colab dan Drive yang lebih akurat
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

def get_default_base_dir():
    """Dapatkan direktori dasar default untuk aplikasi."""
    if _is_colab_environment():
        return "/content"
    return str(Path.home() / "SmartCash")

def _is_colab_environment() -> bool:
    """Deteksi lingkungan Colab dengan multiple indicators."""
    # Method 1: Environment variables
    colab_indicators = [
        "COLAB_GPU", "COLAB_TPU_ADDR", "COLAB_RELEASE_TAG", 
        "COLAB_JUPYTER_IP", "TF_FORCE_GPU_ALLOW_GROWTH"
    ]
    
    for indicator in colab_indicators:
        if os.environ.get(indicator):
            print(f"✅ Colab detected via {indicator}")
            return True
    
    # Method 2: Check Google Colab modules
    try:
        import google.colab
        print("✅ Colab detected via google.colab import")
        return True
    except ImportError:
        pass
    
    # Method 3: Check Jupyter environment in Colab
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython and hasattr(ipython, 'kernel'):
            # Check for Colab-specific kernel attributes
            if hasattr(ipython.kernel, 'do_shutdown') and '/usr/local' in sys.path[0]:
                print("✅ Colab detected via Jupyter kernel signature")
                return True
    except Exception:
        pass
    
    # Method 4: File system indicators
    colab_paths = ['/content', '/usr/local/lib/python3.10/dist-packages/google/colab']
    for path in colab_paths:
        if Path(path).exists():
            print(f"✅ Colab detected via filesystem path: {path}")
            return True
    
    print("❌ Colab environment not detected")
    return False

def _is_drive_mounted() -> bool:
    """Deteksi apakah Google Drive sudah mounted dengan multiple checks."""
    if not _is_colab_environment():
        print("ℹ️ Bukan environment Colab, Drive tidak applicable")
        return False
    
    # Method 1: Check standard mount point
    drive_path = Path('/content/drive/MyDrive')
    if drive_path.exists() and drive_path.is_dir():
        try:
            # Verify it's actually accessible
            list(drive_path.iterdir())
            print(f"✅ Drive mounted dan accessible di: {drive_path}")
            return True
        except (PermissionError, OSError):
            print(f"⚠️ Drive path exists tapi tidak accessible: {drive_path}")
            return False
    
    # Method 2: Check alternative mount points
    alternative_paths = [
        '/content/gdrive/MyDrive',
        '/content/drive/My Drive'
    ]
    
    for alt_path in alternative_paths:
        alt_path_obj = Path(alt_path)
        if alt_path_obj.exists() and alt_path_obj.is_dir():
            try:
                list(alt_path_obj.iterdir())
                print(f"✅ Drive mounted di alternative path: {alt_path}")
                return True
            except (PermissionError, OSError):
                continue
    
    print("❌ Google Drive tidak mounted atau tidak accessible")
    return False

def _get_drive_path() -> Optional[Path]:
    """Dapatkan path ke Google Drive yang benar."""
    if not _is_drive_mounted():
        return None
    
    # Check standard path first
    standard_path = Path('/content/drive/MyDrive/SmartCash')
    if Path('/content/drive/MyDrive').exists():
        standard_path.mkdir(parents=True, exist_ok=True)
        print(f"📁 Drive SmartCash path: {standard_path}")
        return standard_path
    
    # Check alternative paths
    alternative_bases = [
        '/content/gdrive/MyDrive',
        '/content/drive/My Drive'
    ]
    
    for base in alternative_bases:
        if Path(base).exists():
            drive_path = Path(base) / 'SmartCash'
            drive_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Drive SmartCash path (alternative): {drive_path}")
            return drive_path
    
    return None

class EnvironmentManager:
    """Enhanced environment manager dengan deteksi akurat."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EnvironmentManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger=None):
        if getattr(self, '_initialized', False):
            return
            
        print("🚀 Initializing EnvironmentManager...")
        
        self.logger = logger
        self._in_colab = _is_colab_environment()
        self._drive_mounted = _is_drive_mounted() if self._in_colab else False
        self._drive_path = _get_drive_path() if self._drive_mounted else None
        
        # Set base directory
        self._base_dir = (
            Path(base_dir) if base_dir 
            else Path('/content') if self._in_colab 
            else Path(os.getcwd())
        )
        
        # Log summary
        print(f"🌍 Environment Summary:")
        print(f"   • Colab: {self._in_colab}")
        print(f"   • Drive Mounted: {self._drive_mounted}")
        print(f"   • Drive Path: {self._drive_path}")
        print(f"   • Base Dir: {self._base_dir}")
        
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
    
    def mount_drive(self) -> Tuple[bool, str]:
        """Mount Google Drive dengan enhanced error handling."""
        if not self._in_colab:
            return False, "Bukan environment Colab"
        
        if self._drive_mounted:
            return True, f"Drive sudah terhubung di: {self._drive_path}"
        
        try:
            print("🔗 Attempting to mount Google Drive...")
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Re-check mount status
            self._drive_mounted = _is_drive_mounted()
            self._drive_path = _get_drive_path() if self._drive_mounted else None
            
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
            'drive_mounted': self._drive_mounted,
            'drive_path': str(self._drive_path) if self._drive_path else None,
            'python_version': sys.version.split()[0]
        }
        
        # Enhanced GPU detection
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_device_name'] = torch.cuda.get_device_name(0)
        except ImportError:
            info['cuda_available'] = False
        
        # Memory info
        try:
            import psutil
            info['total_memory_gb'] = round(psutil.virtual_memory().total / (1024**3), 2)
            info['available_memory_gb'] = round(psutil.virtual_memory().available / (1024**3), 2)
        except ImportError:
            pass
            
        return info
    
    def refresh_drive_status(self) -> bool:
        """Refresh Drive mount status."""
        if self._in_colab:
            old_status = self._drive_mounted
            self._drive_mounted = _is_drive_mounted()
            self._drive_path = _get_drive_path() if self._drive_mounted else None
            
            if old_status != self._drive_mounted:
                print(f"🔄 Drive status changed: {old_status} → {self._drive_mounted}")
            
            return self._drive_mounted
        return False

# Singleton instance
_environment_manager = None

def get_environment_manager(base_dir: Optional[str] = None, logger=None) -> EnvironmentManager:
    """Dapatkan singleton EnvironmentManager."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager(base_dir, logger)
    return _environment_manager