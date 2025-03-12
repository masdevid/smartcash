"""
File: smartcash/utils/environment_manager.py
Refactored environment manager with improved singleton and drive integration
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

class EnvironmentManager:
    """
    Centralized singleton manager for environment detection and setup.
    
    Handles:
    - Google Colab detection
    - Google Drive mounting
    - Path resolution for different environments
    - Project directory setup
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(EnvironmentManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        """
        Initialize EnvironmentManager.
        
        Args:
            base_dir: Base project directory
            logger: Logging instance
        """
        # Prevent re-initialization
        if self._initialized:
            return
            
        self._logger = logger
        self._in_colab = self._detect_colab()
        self._drive_mounted = False
        self._drive_path = None
        
        # Set base directory
        self._base_dir = (
            Path(base_dir) if base_dir 
            else Path('/content') if self._in_colab 
            else Path(os.getcwd())
        )
        
        # Auto-mount drive if in Colab
        if self._in_colab:
            self._check_drive_mounted()
            
        self._initialized = True
    
    @property
    def is_colab(self) -> bool:
        """Check if running in Google Colab."""
        return self._in_colab
    
    @property
    def base_dir(self) -> Path:
        """Get base directory."""
        return self._base_dir
    
    @property
    def drive_path(self) -> Optional[Path]:
        """Get Google Drive path if mounted."""
        return self._drive_path
    
    @property
    def is_drive_mounted(self) -> bool:
        """Check if Google Drive is mounted."""
        return self._drive_mounted
    
    def _check_drive_mounted(self) -> bool:
        """Check and mount Google Drive if possible."""
        drive_path = Path('/content/drive/MyDrive/SmartCash')
        if os.path.exists('/content/drive/MyDrive'):
            os.makedirs(drive_path, exist_ok=True)
            self._drive_mounted = True
            self._drive_path = drive_path
            return True
        return False
    
    def mount_drive(self, mount_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Mount Google Drive if in Colab.
        
        Args:
            mount_path: Custom mount path (optional)
            
        Returns:
            Tuple of (success, message)
        """
        if not self._in_colab:
            msg = "⚠️ Google Drive can only be mounted in Google Colab"
            if self._logger:
                self._logger.warning(msg)
            return False, msg
        
        try:
            # Already mounted
            if self._drive_mounted:
                return True, "✅ Google Drive already mounted"
            
            # Import and mount
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Set drive path
            mount_path = mount_path or '/content/drive/MyDrive/SmartCash'
            self._drive_path = Path(mount_path)
            os.makedirs(self._drive_path, exist_ok=True)
            
            self._drive_mounted = True
            
            msg = f"✅ Google Drive mounted at {self._drive_path}"
            if self._logger:
                self._logger.info(msg)
            
            return True, msg
        
        except Exception as e:
            msg = f"❌ Drive mount error: {str(e)}"
            if self._logger:
                self._logger.error(msg)
            return False, msg
    
    def get_path(self, relative_path: str) -> Path:
        """
        Get absolute path based on environment.
        
        Args:
            relative_path: Relative path from base directory
            
        Returns:
            Absolute path
        """
        return self._base_dir / relative_path
    
    def setup_directories(self, use_drive: bool = False) -> Dict[str, int]:
        """
        Create project directory structure.
        
        Args:
            use_drive: Use Google Drive for storage if available
            
        Returns:
            Directory creation statistics
        """
        # Standard project directories
        directories = [
            "data/train/images", "data/train/labels",
            "data/valid/images", "data/valid/labels", 
            "data/test/images", "data/test/labels",
            "configs", "runs/train/weights", 
            "logs", "exports"
        ]
        
        # Determine base directory
        base = (self._drive_path if use_drive and self._drive_mounted 
                else self._base_dir)
        
        # Create directories
        stats = {
            'created': 0,
            'existing': 0,
            'error': 0
        }
        
        for dir_path in directories:
            full_path = base / dir_path
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                stats['created' if full_path.exists() else 'error'] += 1
            except Exception as e:
                stats['error'] += 1
                if self._logger:
                    self._logger.warning(f"⚠️ Directory creation error: {dir_path} - {str(e)}")
        
        if self._logger:
            self._logger.info(f"📁 Directory setup: {stats['created']} created, {stats['existing']} existing")
        
        return stats
    
    def create_symlinks(self) -> Dict[str, int]:
        """
        Create symlinks from local to Google Drive directories.
        
        Returns:
            Symlink creation statistics
        """
        if not self._drive_mounted:
            msg = "⚠️ Google Drive not mounted, cannot create symlinks"
            if self._logger:
                self._logger.warning(msg)
            return {'created': 0, 'existing': 0, 'error': 0}
        
        # Symlink mappings
        symlinks = {
            'data': self._drive_path / 'data',
            'configs': self._drive_path / 'configs', 
            'runs': self._drive_path / 'runs',
            'logs': self._drive_path / 'logs'
        }
        
        stats = {
            'created': 0,
            'existing': 0,
            'error': 0
        }
        
        for local_name, target_path in symlinks.items():
            try:
                local_path = self._base_dir / local_name
                
                # Ensure target directory exists
                target_path.mkdir(parents=True, exist_ok=True)
                
                # Create symlink if not exists
                if not local_path.exists():
                    local_path.symlink_to(target_path)
                    stats['created'] += 1
                    if self._logger:
                        self._logger.info(f"🔗 Symlink created: {local_name} -> {target_path}")
                else:
                    stats['existing'] += 1
            except Exception as e:
                stats['error'] += 1
                if self._logger:
                    self._logger.warning(f"⚠️ Symlink creation error: {local_name} - {str(e)}")
        
        return stats
    def get_directory_tree(self, root_dir=None, max_depth=3, indent=0, _current_depth=0):
        """
        Dapatkan struktur direktori dalam format HTML.
        
        Args:
            root_dir: Direktori awal untuk ditampilkan (default: base_dir)
            max_depth: Kedalaman maksimum direktori yang ditampilkan
            indent: Indentasi awal (untuk rekursi)
            _current_depth: Kedalaman saat ini (untuk rekursi)
            
        Returns:
            String HTML yang menampilkan struktur direktori
        """
        if root_dir is None:
            root_dir = self._base_dir
        
        # Konversi ke Path jika string
        root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        
        if not root_dir.exists():
            return f"<span style='color:red'>❌ Directory not found: {root_dir}</span>"
        
        if _current_depth > max_depth:
            return "<span style='color:gray'>...</span>"
        
        # Mulai dengan ul atau pre tag
        if indent == 0:
            result = "<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n"
        else:
            result = ""
        
        # Tampilkan direktori current
        if indent == 0:
            result += f"<span style='color:#0366d6; font-weight:bold;'>{root_dir.name}/</span>\n"
        
        # Buat space indentasi
        spaces = "│  " * indent
        
        # Dapatkan isi direktori, sortir direktori dulu
        items = sorted(root_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            
            # Pilih garis/karakter penghubung
            prefix = "└─ " if is_last else "├─ "
            
            if item.is_dir():
                # Gunakan warna biru untuk direktori
                result += f"{spaces}{prefix}<span style='color:#0366d6; font-weight:bold;'>{item.name}/</span>\n"
                
                # Tambah garis penghubung untuk level berikutnya
                next_spaces = spaces + ("   " if is_last else "│  ")
                if _current_depth < max_depth:
                    # Rekursi untuk subdirektori
                    subdirs = self.get_directory_tree(
                        root_dir=item,
                        max_depth=max_depth,
                        indent=indent + 1,
                        _current_depth=_current_depth + 1
                    )
                    # Hapus tag pre dari hasil rekursi
                    subdirs = subdirs.replace("<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n", "")
                    subdirs = subdirs.replace("</pre>", "")
                    result += subdirs
            else:
                # Gunakan warna default untuk file
                result += f"{spaces}{prefix}{item.name}\n"
        
        # Tutup tag jika pada level teratas
        if indent == 0:
            result += "</pre>"
        
        return result
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get comprehensive system information.
        
        Returns:
            System details dictionary
        """
        info = {
            'environment': 'Colab' if self._in_colab else 'Local',
            'base_directory': str(self._base_dir),
            'drive_mounted': self._drive_mounted,
            'python_version': sys.version
        }
        
        # GPU detection
        try:
            import torch
            info['cuda'] = {
                'available': torch.cuda.is_available(),
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else None
            }
        except ImportError:
            info['cuda'] = {'available': False}
        
        return info
    
    def _detect_colab(self) -> bool:
        """
        Detect Google Colab environment.
        
        Returns:
            Whether running in Colab
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False

def get_environment_manager(base_dir: Optional[str] = None, logger = None) -> EnvironmentManager:
    """
    Get singleton EnvironmentManager instance.
    
    Args:
        base_dir: Base project directory
        logger: Logging instance
    
    Returns:
        EnvironmentManager singleton
    """
    return EnvironmentManager(base_dir, logger)