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
    return "/content" if _is_colab_environment() else str(Path.home() / "SmartCash")

def _is_colab_environment() -> bool:
    """Deteksi lingkungan Colab dengan multiple indicators."""
    return (any(os.environ.get(indicator) and print(f"✅ Colab detected via {indicator}") for indicator in ["COLAB_GPU", "COLAB_TPU_ADDR", "COLAB_RELEASE_TAG", "COLAB_JUPYTER_IP", "TF_FORCE_GPU_ALLOW_GROWTH"])) or \
           (lambda: ((__import__('google.colab'), print("✅ Colab detected via google.colab import"), True)[2])() if True else False)() if True else False or \
           (lambda: (print("✅ Colab detected via Jupyter kernel signature"), True)[1] if (lambda ipy: ipy and hasattr(ipy, 'kernel') and hasattr(ipy.kernel, 'do_shutdown') and '/usr/local' in sys.path[0])(__import__('IPython').get_ipython()) else False)() if True else False or \
           any(Path(path).exists() and print(f"✅ Colab detected via filesystem path: {path}") for path in ['/content', '/usr/local/lib/python3.10/dist-packages/google/colab']) or \
           (print("❌ Colab environment not detected"), False)[1]

def _is_drive_mounted() -> bool:
    """Deteksi apakah Google Drive sudah mounted dengan multiple checks."""
    return False if not _is_colab_environment() and print("ℹ️ Bukan environment Colab, Drive tidak applicable") else \
           (lambda p: (list(p.iterdir()), print(f"✅ Drive mounted dan accessible di: {p}"), True)[2] if p.exists() and p.is_dir() else False)(Path('/content/drive/MyDrive')) if True else False or \
           any((lambda ap: (list(ap.iterdir()), print(f"✅ Drive mounted di alternative path: {ap}"), True)[2] if ap.exists() and ap.is_dir() else False)(Path(alt_path)) for alt_path in ['/content/gdrive/MyDrive', '/content/drive/My Drive']) or \
           (print("❌ Google Drive tidak mounted atau tidak accessible"), False)[1]

def _get_drive_path() -> Optional[Path]:
    """Dapatkan path ke Google Drive yang benar."""
    return None if not _is_drive_mounted() else \
           (lambda sp: (sp.mkdir(parents=True, exist_ok=True), print(f"📁 Drive SmartCash path: {sp}"), sp)[2])(Path('/content/drive/MyDrive/SmartCash')) if Path('/content/drive/MyDrive').exists() else \
           next(((lambda dp: (dp.mkdir(parents=True, exist_ok=True), print(f"📁 Drive SmartCash path (alternative): {dp}"), dp)[2])(Path(base) / 'SmartCash') for base in ['/content/gdrive/MyDrive', '/content/drive/My Drive'] if Path(base).exists()), None)

class EnvironmentManager:
    """Enhanced environment manager dengan deteksi akurat."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        return cls._instance if cls._instance else (setattr(cls, '_instance', super(EnvironmentManager, cls).__new__(cls)), setattr(cls._instance, '_initialized', False), cls._instance)[2]
    
    def __init__(self, base_dir: Optional[str] = None, logger=None):
        (lambda: None)() if getattr(self, '_initialized', False) else \
        (print("🚀 Initializing EnvironmentManager..."), setattr(self, 'logger', logger), setattr(self, '_in_colab', _is_colab_environment()), 
         setattr(self, '_drive_mounted', _is_drive_mounted() if self._in_colab else False), 
         setattr(self, '_drive_path', _get_drive_path() if self._drive_mounted else None),
         setattr(self, '_base_dir', Path(base_dir) if base_dir else Path('/content') if self._in_colab else Path(os.getcwd())),
         print(f"🌍 Environment Summary:\n   • Colab: {self._in_colab}\n   • Drive Mounted: {self._drive_mounted}\n   • Drive Path: {self._drive_path}\n   • Base Dir: {self._base_dir}"),
         setattr(self, '_initialized', True))
    
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
        return (False, "Bukan environment Colab") if not self._in_colab else \
               (True, f"Drive sudah terhubung di: {self._drive_path}") if self._drive_mounted else \
               (lambda: (print("🔗 Attempting to mount Google Drive..."), __import__('google.colab').drive.mount('/content/drive'), 
                        setattr(self, '_drive_mounted', _is_drive_mounted()), setattr(self, '_drive_path', _get_drive_path() if self._drive_mounted else None),
                        (True, f"Drive berhasil terhubung di: {self._drive_path}") if self._drive_mounted else (False, "Drive mount command completed tapi tidak dapat diakses"))[4])() if True else (False, f"Gagal mount drive: {str(__import__('sys').exc_info()[1])}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get enhanced system info."""
        return {**{'environment': 'Google Colab' if self._in_colab else 'Local', 'base_directory': str(self._base_dir), 'drive_mounted': self._drive_mounted, 'drive_path': str(self._drive_path) if self._drive_path else None, 'python_version': sys.version.split()[0]}, 
                **(lambda: {'cuda_available': __import__('torch').cuda.is_available(), **({'cuda_device_count': __import__('torch').cuda.device_count(), 'cuda_device_name': __import__('torch').cuda.get_device_name(0)} if __import__('torch').cuda.is_available() else {})} if True else {'cuda_available': False})(),
                **(lambda: {'total_memory_gb': round(__import__('psutil').virtual_memory().total / (1024**3), 2), 'available_memory_gb': round(__import__('psutil').virtual_memory().available / (1024**3), 2)} if True else {})()}
    
    def refresh_drive_status(self) -> bool:
        """Refresh Drive mount status."""
        return (lambda old: (setattr(self, '_drive_mounted', _is_drive_mounted()), setattr(self, '_drive_path', _get_drive_path() if self._drive_mounted else None), 
                            print(f"🔄 Drive status changed: {old} → {self._drive_mounted}") if old != self._drive_mounted else None, self._drive_mounted)[3])(self._drive_mounted) if self._in_colab else False

# Singleton instance
_environment_manager = None

def get_environment_manager(base_dir: Optional[str] = None, logger=None) -> EnvironmentManager:
    """Dapatkan singleton EnvironmentManager."""
    global _environment_manager
    return _environment_manager if _environment_manager else (globals().update({'_environment_manager': EnvironmentManager(base_dir, logger)}), _environment_manager)[1]