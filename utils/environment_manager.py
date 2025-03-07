"""
File: smartcash/utils/environment_manager.py
Author: Alfrida Sabar
Deskripsi: Centralized environment manager yang mendeteksi dan menghandle environment runtime (Colab/local)
          serta integrasi dengan Google Drive.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, List, Union

class EnvironmentManager:
    """
    Singleton manager untuk environment detection dan setup.
    
    Menangani:
    - Deteksi Google Colab
    - Mount Google Drive
    - Path resolution untuk different environments
    - Setup directories untuk project
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(EnvironmentManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        """
        Initialize EnvironmentManager.
        
        Args:
            base_dir: Base directory for the project (optional)
            logger: Logger to use for logging
        """
        # Only initialize once
        if self._initialized:
            return
            
        self._logger = logger
        self._in_colab = self._detect_colab()
        self._drive_mounted = False
        self._drive_path = None
        
        # Set base directory
        if base_dir:
            self._base_dir = Path(base_dir)
        elif self._in_colab:
            self._base_dir = Path('/content')
        else:
            self._base_dir = Path(os.getcwd())
            
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
    
    def get_path(self, relative_path: str) -> Path:
        """
        Get absolute path based on environment.
        
        Args:
            relative_path: Relative path from base directory
            
        Returns:
            Path object for the requested path
        """
        return self._base_dir / relative_path
    
    def mount_drive(self, mount_path: str = '/content/drive/MyDrive/SmartCash') -> Tuple[bool, str]:
        """
        Mount Google Drive if in Colab.
        
        Args:
            mount_path: Path to mount drive to
            
        Returns:
            Tuple (success, message)
        """
        if not self._in_colab:
            msg = "⚠️ Google Drive hanya dapat di-mount di Google Colab"
            if self._logger:
                self._logger.warning(msg)
            return False, msg
        
        try:
            # Check if drive already mounted
            if os.path.exists('/content/drive/MyDrive'):
                self._drive_mounted = True
                self._drive_path = Path(mount_path)
                os.makedirs(self._drive_path, exist_ok=True)
                msg = "✅ Google Drive sudah di-mount"
                if self._logger:
                    self._logger.info(msg)
                return True, msg
            
            # Mount drive
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Create project directory
            self._drive_path = Path(mount_path)
            os.makedirs(self._drive_path, exist_ok=True)
            
            self._drive_mounted = True
            msg = f"✅ Google Drive berhasil di-mount di {mount_path}"
            if self._logger:
                self._logger.success(msg)
            return True, msg
            
        except Exception as e:
            msg = f"❌ Error saat mount Google Drive: {str(e)}"
            if self._logger:
                self._logger.error(msg)
            return False, msg
    
    def setup_directories(self, use_drive: bool = False) -> Dict[str, int]:
        """
        Create project directory structure.
        
        Args:
            use_drive: Whether to use Google Drive for storage
            
        Returns:
            Dictionary with statistics about created directories
        """
        # Standard directories to create
        directories = [
            "data/train/images",
            "data/train/labels",
            "data/valid/images",
            "data/valid/labels",
            "data/test/images",
            "data/test/labels",
            "configs",
            "runs/train/weights",
            "logs",
            "exports"
        ]
        
        # Determine base directory for creating structure
        if use_drive and self._drive_mounted:
            base = self._drive_path
        else:
            base = self._base_dir
            
        # Create directories
        stats = {
            'created': 0,
            'existing': 0,
            'error': 0
        }
        
        for d in directories:
            try:
                dir_path = base / d
                if not dir_path.exists():
                    os.makedirs(dir_path, exist_ok=True)
                    stats['created'] += 1
                else:
                    stats['existing'] += 1
            except Exception as e:
                stats['error'] += 1
                if self._logger:
                    self._logger.warning(f"⚠️ Error creating directory {d}: {str(e)}")
        
        # Log results
        if self._logger:
            self._logger.info(f"📁 Directory setup statistics: {stats['created']} created, {stats['existing']} existing")
            
        return stats
    
    def create_symlinks(self) -> Dict[str, int]:
        """
        Create symlinks from local to Google Drive directories.
        
        Returns:
            Dictionary with statistics about created symlinks
        """
        if not self._drive_mounted:
            msg = "⚠️ Google Drive tidak di-mount, tidak dapat membuat symlinks"
            if self._logger:
                self._logger.warning(msg)
            return {'created': 0, 'existing': 0, 'error': 0}
        
        # Define symlinks to create (local -> drive)
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
        
        for local_path, target_path in symlinks.items():
            try:
                # Ensure target directory exists
                os.makedirs(target_path, exist_ok=True)
                
                # Create symlink if it doesn't exist
                local_path_obj = self._base_dir / local_path
                if not local_path_obj.exists():
                    os.symlink(target_path, local_path_obj)
                    stats['created'] += 1
                    if self._logger:
                        self._logger.info(f"🔗 Created symlink: {local_path} -> {target_path}")
                else:
                    stats['existing'] += 1
            except Exception as e:
                stats['error'] += 1
                if self._logger:
                    self._logger.warning(f"⚠️ Error creating symlink {local_path}: {str(e)}")
        
        return stats
    
    def get_directory_tree(self, max_depth: int = 3) -> str:
        """
        Generate directory tree structure as HTML.
        
        Args:
            max_depth: Maximum depth to display
            
        Returns:
            HTML string representing directory tree
        """
        def _get_tree(directory, prefix='', is_last=True, depth=0):
            if depth > max_depth:
                return ""
                
            base_name = os.path.basename(directory)
            result = ""
            
            if depth > 0:
                # Add connector line and directory name
                result += f"{prefix}{'└── ' if is_last else '├── '}<span style='color: #3498db;'>{base_name}</span><br/>"
            else:
                # Root directory
                result += f"<span style='color: #2980b9; font-weight: bold;'>{base_name}</span><br/>"
                
            # Update prefix for children
            if depth > 0:
                prefix += '    ' if is_last else '│   '
                
            # List directory contents
            try:
                items = list(sorted([x for x in Path(directory).iterdir()]))
                
                # Only show a subset of items if there are too many
                if len(items) > 10 and depth > 0:
                    items = items[:10]
                    show_ellipsis = True
                else:
                    show_ellipsis = False
                    
                for i, item in enumerate(items):
                    if item.is_dir():
                        # Recursively process subdirectories
                        result += _get_tree(str(item), prefix, i == len(items) - 1 and not show_ellipsis, depth + 1)
                    elif depth < max_depth:
                        # Add file name
                        result += f"{prefix}{'└── ' if i == len(items) - 1 and not show_ellipsis else '├── '}{item.name}<br/>"
                        
                if show_ellipsis:
                    result += f"{prefix}└── <i>... dan item lainnya</i><br/>"
            except Exception:
                result += f"{prefix}└── <i>Error saat membaca direktori</i><br/>"
                
            return result
        
        base_dir = self._drive_path if self._drive_mounted else self._base_dir
        html = "<div style='font-family: monospace;'>"
        html += _get_tree(base_dir)
        html += "</div>"
        
        return html
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'in_colab': self._in_colab,
            'drive_mounted': self._drive_mounted,
            'base_dir': str(self._base_dir),
            'python_version': sys.version,
        }
        
        # Check if GPU is available (using torch if available)
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if info['cuda_available']:
                info['cuda_device'] = torch.cuda.get_device_name(0)
                info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except ImportError:
            info['cuda_available'] = False
        
        return info
    
    def _detect_colab(self) -> bool:
        """
        Detect if running in Google Colab.
        
        Returns:
            Boolean indicating if running in Colab
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False