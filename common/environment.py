"""
File: smartcash/common/environment.py
Deskripsi: Manajer lingkungan terpusat untuk deteksi dan konfigurasi environment aplikasi SmartCash
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

class EnvironmentManager:
    """
    Manajer lingkungan singleton terpusat untuk deteksi dan setup environment.
    
    Menangani:
    - Deteksi Google Colab
    - Mounting Google Drive
    - Resolusi path untuk berbagai lingkungan
    - Setup direktori proyek
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implementasi pola singleton."""
        if cls._instance is None: cls._instance = super(EnvironmentManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        """
        Inisialisasi EnvironmentManager.
        
        Args:
            base_dir: Direktori dasar proyek
            logger: Instance logger
        """
        # Mencegah re-inisialisasi
        if getattr(self, '_initialized', False): return
            
        # Import komponen yang sudah ada
        try:
            from smartcash.common.utils import is_colab, is_notebook
            from smartcash.common.logger import get_logger
            from smartcash.common.io import file_utils
            
            self._in_colab = is_colab()
            self._in_notebook = is_notebook()
            self.logger = logger or get_logger("environment_manager")
            self._file_utils = file_utils
        except ImportError:
            self._in_colab = self._detect_colab()
            self._in_notebook = self._detect_notebook()
            self._file_utils = None
            self.logger = logger
        
        self._drive_mounted = False
        self._drive_path = None
        
        # Set direktori dasar
        self._base_dir = (
            Path(base_dir) if base_dir 
            else Path('/content/SmartCash') if self._in_colab 
            else Path(os.getcwd())
        )
        
        # Auto-detect drive jika di Colab
        if self._in_colab: self.detect_drive()
            
        self._initialized = True
        
        if self.logger:
            env_type = "Google Colab" if self._in_colab else "Notebook" if self._in_notebook else "Lokal"
            self.logger.info(f"🔍 Environment terdeteksi: {env_type}")
            self.logger.info(f"📂 Direktori dasar: {self._base_dir}")
    
    @property
    def is_colab(self) -> bool: return self._in_colab
    
    @property
    def is_notebook(self) -> bool: return self._in_notebook
    
    @property
    def base_dir(self) -> Path: return self._base_dir
    
    @property
    def drive_path(self) -> Optional[Path]: return self._drive_path
    
    @property
    def is_drive_mounted(self) -> bool: return self._drive_mounted
    
    def detect_drive(self) -> bool:
        """Deteksi dan set status Google Drive menggunakan drive_utils jika tersedia."""
        try:
            # Menggunakan drive_utils.detect_drive_mount jika tersedia
            from smartcash.ui.utils.drive_utils import detect_drive_mount
            is_mounted, drive_path = detect_drive_mount()
            
            if is_mounted:
                # Set drive path ke SmartCash
                self._drive_mounted = True
                self._drive_path = Path(drive_path) / 'SmartCash'
                # Pastikan directory SmartCash ada di Drive
                os.makedirs(self._drive_path, exist_ok=True)
                if self.logger: self.logger.info(f"✅ Google Drive terdeteksi pada: {self._drive_path}")
                return True
        except ImportError:
            # Fallback jika drive_utils tidak tersedia
            drive_path = Path('/content/drive/MyDrive/SmartCash')
            drive_mount_point = Path('/content/drive/MyDrive')
            
            if drive_mount_point.exists():
                # Pastikan directory SmartCash ada di Drive
                os.makedirs(drive_path, exist_ok=True)
                self._drive_mounted = True
                self._drive_path = drive_path
                if self.logger: self.logger.info(f"✅ Google Drive terdeteksi pada: {drive_path}")
                return True
        return False
    
    def mount_drive(self, mount_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Mount Google Drive jika di Colab.
        
        Args:
            mount_path: Custom mount path (opsional)
            
        Returns:
            Tuple dari (sukses, pesan)
        """
        if not self._in_colab:
            msg = "⚠️ Google Drive hanya dapat di-mount di Google Colab"
            if self.logger: self.logger.warning(msg)
            return False, msg
        
        try:
            # Sudah ter-mount
            if self._drive_mounted: return True, "✅ Google Drive sudah ter-mount"
            
            # Import dan mount
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Set drive path
            mount_path = mount_path or '/content/drive/MyDrive/SmartCash'
            self._drive_path = Path(mount_path)
            os.makedirs(self._drive_path, exist_ok=True)
            
            self._drive_mounted = True
            
            msg = f"✅ Google Drive berhasil di-mount pada {self._drive_path}"
            if self.logger: self.logger.info(msg)
            
            return True, msg
        
        except Exception as e:
            msg = f"❌ Error mounting Google Drive: {str(e)}"
            if self.logger: self.logger.error(msg)
            return False, msg
    
    def get_path(self, relative_path: str) -> Path:
        """Dapatkan path absolut berdasarkan environment."""
        return self._base_dir / relative_path
    
    def get_project_root(self) -> Path:
        """Dapatkan direktori root proyek SmartCash."""
        from smartcash.common.utils import get_project_root
        try: return get_project_root()
        except: return self._base_dir  # Fallback
    
    def setup_project_structure(self, use_drive: bool = False, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, int]:
        """
        Buat struktur direktori proyek SmartCash.
        
        Args:
            use_drive: Gunakan Google Drive untuk penyimpanan jika tersedia
            progress_callback: Callback untuk menampilkan progres (current, total, message)
            
        Returns:
            Statistik pembuatan direktori
        """
        # Struktur direktori standar proyek SmartCash
        directories: List[str] = [
            "data/train/images", "data/train/labels",
            "data/valid/images", "data/valid/labels", 
            "data/test/images", "data/test/labels",
            "data/preprocessed/train", "data/preprocessed/valid", "data/preprocessed/test",
            "configs", "runs/train/weights", 
            "logs", "exports",
            "checkpoints"
        ]
        
        # Tentukan direktori dasar
        base = (self._drive_path if use_drive and self._drive_mounted else self._base_dir)
        
        # Buat direktori dengan fallback ke method langsung
        stats = {'created': 0, 'existing': 0, 'error': 0}
        total_dirs = len(directories)
        
        # Fungsi sederhana untuk membuat direktori
        def ensure_dir(path):
            try:
                os.makedirs(path, exist_ok=True)
                return path
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Error pembuatan direktori: {path} - {str(e)}")
                raise e
        
        # Buat direktori
        for idx, dir_path in enumerate(directories):
            if progress_callback: 
                progress_callback(idx + 1, total_dirs, f"Membuat direktori: {dir_path}")
                
            full_path = base / dir_path
            try:
                if not full_path.exists():
                    ensure_dir(full_path)
                    stats['created'] += 1
                else:
                    stats['existing'] += 1
            except Exception:
                stats['error'] += 1
        
        if self.logger:
            self.logger.info(f"📁 Setup direktori: {stats['created']} dibuat, {stats['existing']} sudah ada")
        
        return stats
    
    def create_symlinks(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, int]:
        """
        Buat symlink dari direktori lokal ke direktori Google Drive.
        
        Args:
            progress_callback: Callback untuk menampilkan progres (current, total, message)
            
        Returns:
            Statistik pembuatan symlink
        """
        if not self._drive_mounted:
            msg = "⚠️ Google Drive tidak ter-mount, tidak dapat membuat symlink"
            if self.logger: self.logger.warning(msg)
            return {'created': 0, 'existing': 0, 'error': 0}
        
        # Mapping symlink (koreksi: gunakan path /content, bukan /content/SmartCash)
        symlinks = {
            'data': self._drive_path / 'data',
            'configs': self._drive_path / 'configs', 
            'runs': self._drive_path / 'runs',
            'logs': self._drive_path / 'logs',
            'checkpoints': self._drive_path / 'checkpoints'
        }
        
        stats = {'created': 0, 'existing': 0, 'error': 0}
        total_symlinks = len(symlinks)
        
        import shutil, os
        
        for idx, (local_name, target_path) in enumerate(symlinks.items()):
            if progress_callback: progress_callback(idx + 1, total_symlinks, f"Membuat symlink: {local_name} -> {target_path}")
                
            try:
                # Pastikan direktori target ada
                os.makedirs(target_path, exist_ok=True)
                
                # Gunakan path /content, bukan /content/SmartCash
                local_path = Path('/content') / local_name
                
                # Cek jika path lokal ada dan bukan symlink
                if local_path.exists() and not local_path.is_symlink():
                    backup_path = local_path.with_name(f"{local_name}_backup")
                    if self.logger:
                        self.logger.info(f"🔄 Memindahkan direktori lokal ke backup: {local_path} -> {backup_path}")
                    
                    # Hapus backup yang sudah ada
                    if backup_path.exists(): shutil.rmtree(backup_path)
                    
                    # Pindahkan direktori lokal ke backup
                    local_path.rename(backup_path)
                
                # Buat symlink jika belum ada
                if not local_path.exists():
                    local_path.symlink_to(target_path)
                    stats['created'] += 1
                    if self.logger: self.logger.info(f"🔗 Symlink berhasil dibuat: {local_name} -> {target_path}")
                else:
                    stats['existing'] += 1
            except Exception as e:
                stats['error'] += 1
                if self.logger: self.logger.warning(f"⚠️ Error pembuatan symlink: {local_name} - {str(e)}")
        
        return stats
    
    def get_directory_tree(self, root_dir: Optional[Union[str, Path]] = None, max_depth: int = 3, 
                         indent: int = 0, _current_depth: int = 0) -> str:
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
        # Implementasi default jika komponen tidak tersedia
        root_dir = Path(root_dir or self._base_dir)
        
        if not root_dir.exists(): return f"<span style='color:red'>❌ Direktori tidak ditemukan: {root_dir}</span>"
        if _current_depth > max_depth: return "<span style='color:gray'>...</span>"
        
        result = "<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n" if indent == 0 else ""
        
        if indent == 0: result += f"<span style='color:#0366d6; font-weight:bold;'>{root_dir.name}/</span>\n"
        
        spaces = "│  " * indent
        
        try:
            items = sorted(root_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return f"{spaces}<span style='color:red'>❌ Akses ditolak: {root_dir}</span>\n"
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            prefix = "└─ " if is_last else "├─ "
            
            if item.is_dir():
                result += f"{spaces}{prefix}<span style='color:#0366d6; font-weight:bold;'>{item.name}/</span>\n"
                
                next_spaces = spaces + ("   " if is_last else "│  ")
                if _current_depth < max_depth:
                    subdirs = self.get_directory_tree(item, max_depth, indent + 1, _current_depth + 1)
                    subdirs = subdirs.replace("<pre style='margin:0; padding:5px; background:#f8f9fa; font-family:monospace; color:#333;'>\n", "")
                    subdirs = subdirs.replace("</pre>", "")
                    result += subdirs
            else:
                ext = item.suffix.lower()
                color = self._get_color_for_extension(ext)
                result += f"{spaces}{prefix}<span style='color:{color};'>{item.name}</span>\n"
        
        if indent == 0: result += "</pre>"
        return result
    
    def _get_color_for_extension(self, ext: str) -> str:
        """Mendapatkan warna berdasarkan ekstensi file."""
        if ext in ['.py']: return "#3572A5"  # Python files
        elif ext in ['.md', '.txt']: return "#6A737D"  # Documentation files
        elif ext in ['.jpg', '.jpeg', '.png', '.bmp']: return "#E34F26"  # Image files
        elif ext in ['.json', '.yaml', '.yml']: return "#F1E05A"  # Config files
        elif ext in ['.pt', '.pth']: return "#9B4DCA"  # PyTorch model files
        return "#333"  # Default color
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi sistem komprehensif.
        
        Returns:
            Dictionary detail sistem
        """
        # Gunakan utils jika tersedia
        try:
            from smartcash.common.utils import get_system_info
            return get_system_info()
        except ImportError:
            pass
            
        # Implementasi default
        import platform, sys
        
        info = {
            'environment': 'Google Colab' if self._in_colab else 'Jupyter/IPython' if self._in_notebook else 'Local',
            'base_directory': str(self._base_dir),
            'drive_mounted': self._drive_mounted,
            'python_version': sys.version
        }
        
        # Deteksi GPU
        try:
            import torch
            info['cuda'] = {
                'available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'memory_gb': round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else None
            }
        except ImportError:
            info['cuda'] = {'available': False}
            
        # Informasi sistem
        try:
            import psutil
            info['system'] = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
        except ImportError:
            info['system'] = {'platform': platform.platform() if 'platform' in sys.modules else 'Unknown'}
            
        return info
    
    def _detect_colab(self) -> bool:
        """
        Deteksi lingkungan Google Colab.
        
        Returns:
            Apakah berjalan di Colab
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False
            
    def _detect_notebook(self) -> bool:
        """
        Deteksi lingkungan Jupyter/IPython notebook.
        
        Returns:
            Apakah berjalan di notebook
        """
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except ImportError:
            return False

# Singleton instance
_environment_manager = None

def get_environment_manager(base_dir: Optional[str] = None, logger = None) -> EnvironmentManager:
    """
    Dapatkan instance singleton EnvironmentManager.
    
    Args:
        base_dir: Direktori dasar proyek
        logger: Instance logger
    
    Returns:
        Singleton EnvironmentManager
    """
    global _environment_manager
    if _environment_manager is None: _environment_manager = EnvironmentManager(base_dir, logger)
    return _environment_manager