"""
File: smartcash/common/environment.py
Deskripsi: Manajer lingkungan terpusat untuk deteksi dan konfigurasi environment aplikasi SmartCash dengan perhitungan progres yang dioptimalkan
"""

import os
import sys
import shutil
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
        if cls._instance is None:
            cls._instance = super(EnvironmentManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        """
        Inisialisasi EnvironmentManager.
        
        Args:
            base_dir: Direktori dasar proyek
            logger: Instance logger
        """
        # Mencegah re-inisialisasi
        if getattr(self, '_initialized', False):
            return
            
        self._logger = logger
        self._in_colab = self._detect_colab()
        self._in_notebook = self._detect_notebook()
        self._drive_mounted = False
        self._drive_path = None
        
        # Set direktori dasar
        self._base_dir = (
            Path(base_dir) if base_dir 
            else Path('/content/SmartCash') if self._in_colab 
            else Path(os.getcwd())
        )
        
        # Auto-mount drive jika di Colab
        if self._in_colab:
            self._check_drive_mounted()
            
        self._initialized = True
        
        if self._logger:
            env_type = "Google Colab" if self._in_colab else "Notebook" if self._in_notebook else "Lokal"
            self._logger.info(f"🔍 Environment terdeteksi: {env_type}")
            self._logger.info(f"📂 Direktori dasar: {self._base_dir}")
    
    @property
    def is_colab(self) -> bool:
        """Cek apakah berjalan di Google Colab."""
        return self._in_colab
    
    @property
    def is_notebook(self) -> bool:
        """Cek apakah berjalan di notebook (Jupyter, IPython)."""
        return self._in_notebook
    
    @property
    def base_dir(self) -> Path:
        """Dapatkan direktori dasar."""
        return self._base_dir
    
    @property
    def drive_path(self) -> Optional[Path]:
        """Dapatkan path Google Drive jika ter-mount."""
        return self._drive_path
    
    @property
    def is_drive_mounted(self) -> bool:
        """Cek apakah Google Drive ter-mount."""
        return self._drive_mounted
    
    def _check_drive_mounted(self) -> bool:
        """Cek dan mount Google Drive jika memungkinkan."""
        drive_path = Path('/content/drive/MyDrive/SmartCash')
        if os.path.exists('/content/drive/MyDrive'):
            os.makedirs(drive_path, exist_ok=True)
            self._drive_mounted = True
            self._drive_path = drive_path
            if self._logger:
                self._logger.info(f"✅ Google Drive terdeteksi pada: {drive_path}")
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
            if self._logger:
                self._logger.warning(msg)
            return False, msg
        
        try:
            # Sudah ter-mount
            if self._drive_mounted:
                return True, "✅ Google Drive sudah ter-mount"
            
            # Import dan mount
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Set drive path
            mount_path = mount_path or '/content/drive/MyDrive/SmartCash'
            self._drive_path = Path(mount_path)
            os.makedirs(self._drive_path, exist_ok=True)
            
            self._drive_mounted = True
            
            msg = f"✅ Google Drive berhasil di-mount pada {self._drive_path}"
            if self._logger:
                self._logger.info(msg)
            
            return True, msg
        
        except Exception as e:
            msg = f"❌ Error mounting Google Drive: {str(e)}"
            if self._logger:
                self._logger.error(msg)
            return False, msg
    
    def get_path(self, relative_path: str) -> Path:
        """
        Dapatkan path absolut berdasarkan environment.
        
        Args:
            relative_path: Path relatif dari direktori dasar
            
        Returns:
            Path absolut
        """
        return self._base_dir / relative_path
    
    def get_project_root(self) -> Path:
        """
        Dapatkan direktori root proyek SmartCash.
        
        Returns:
            Path direktori root proyek
        """
        # Cek beberapa kemungkinan lokasi root project
        current = Path.cwd()
        
        # Cek jika direktori saat ini atau parentnya mengandung file setup.py atau struktur proyek smartcash
        for path in [current] + list(current.parents):
            if (path / "setup.py").exists() and (path / "smartcash").exists():
                return path
                
        # Fallback ke base_dir jika tidak ditemukan
        return self._base_dir
    
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
        base = (self._drive_path if use_drive and self._drive_mounted 
                else self._base_dir)
        
        # Buat direktori
        stats = {
            'created': 0,
            'existing': 0,
            'error': 0
        }
        
        total_dirs = len(directories)
        
        for idx, dir_path in enumerate(directories):
            if progress_callback:
                # Perhitungan progress yang benar: idx dimulai dari 0, jadi tambahkan 1 untuk menghindari progress 0%
                progress_callback(idx + 1, total_dirs, f"Membuat direktori: {dir_path}")
                
            full_path = base / dir_path
            try:
                if not full_path.exists():
                    full_path.mkdir(parents=True, exist_ok=True)
                    stats['created'] += 1
                else:
                    stats['existing'] += 1
            except Exception as e:
                stats['error'] += 1
                if self._logger:
                    self._logger.warning(f"⚠️ Error pembuatan direktori: {dir_path} - {str(e)}")
        
        if self._logger:
            self._logger.info(f"📁 Setup direktori: {stats['created']} dibuat, {stats['existing']} sudah ada")
        
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
            if self._logger:
                self._logger.warning(msg)
            return {'created': 0, 'existing': 0, 'error': 0}
        
        # Mapping symlink
        symlinks = {
            'data': self._drive_path / 'data',
            'configs': self._drive_path / 'configs', 
            'runs': self._drive_path / 'runs',
            'logs': self._drive_path / 'logs',
            'checkpoints': self._drive_path / 'checkpoints'
        }
        
        stats = {
            'created': 0,
            'existing': 0,
            'error': 0
        }
        
        total_symlinks = len(symlinks)
        
        for idx, (local_name, target_path) in enumerate(symlinks.items()):
            if progress_callback:
                # Perhitungan progress yang benar: idx dimulai dari 0, jadi tambahkan 1
                progress_callback(idx + 1, total_symlinks, f"Membuat symlink: {local_name} -> {target_path}")
                
            try:
                local_path = self._base_dir / local_name
                
                # Pastikan direktori target ada
                target_path.mkdir(parents=True, exist_ok=True)
                
                # Cek jika path lokal ada dan bukan symlink
                if local_path.exists() and not local_path.is_symlink():
                    # Rename direktori yang ada untuk backup
                    backup_path = local_path.with_name(f"{local_name}_backup")
                    if self._logger:
                        self._logger.info(f"🔄 Memindahkan direktori lokal ke backup: {local_path} -> {backup_path}")
                    local_path.rename(backup_path)
                
                # Buat symlink jika belum ada
                if not local_path.exists():
                    # Pada Windows gunakan metode alternatif jika symlink tidak tersedia
                    if os.name == 'nt':
                        try:
                            os.symlink(target_path, local_path, target_is_directory=True)
                        except OSError:
                            # Jika symlink gagal (misalnya hak admin), gunakan junction
                            if self._logger:
                                self._logger.warning(f"⚠️ Symlink gagal, mencoba junction pada Windows untuk: {local_name}")
                            import subprocess
                            subprocess.run(f'mklink /J "{local_path}" "{target_path}"', shell=True)
                    else:
                        local_path.symlink_to(target_path)
                    
                    stats['created'] += 1
                    if self._logger:
                        self._logger.info(f"🔗 Symlink berhasil dibuat: {local_name} -> {target_path}")
                else:
                    stats['existing'] += 1
            except Exception as e:
                stats['error'] += 1
                if self._logger:
                    self._logger.warning(f"⚠️ Error pembuatan symlink: {local_name} - {str(e)}")
        
        return stats
    
    def get_directory_tree(self, root_dir: Optional[Union[str, Path]] = None, 
                          max_depth: int = 3, 
                          indent: int = 0, 
                          _current_depth: int = 0) -> str:
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
            return f"<span style='color:red'>❌ Direktori tidak ditemukan: {root_dir}</span>"
        
        if _current_depth > max_depth:
            return "<span style='color:gray'>...</span>"
        
        # Mulai dengan pre tag
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
        try:
            items = sorted(root_dir.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        except PermissionError:
            return f"{spaces}<span style='color:red'>❌ Akses ditolak: {root_dir}</span>\n"
        
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
                # File ekstensi untuk warna
                ext = item.suffix.lower()
                color = "#333"  # Default color
                
                # Set warna berdasarkan tipe file
                if ext in ['.py']:
                    color = "#3572A5"  # Python files
                elif ext in ['.md', '.txt']:
                    color = "#6A737D"  # Documentation files
                elif ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    color = "#E34F26"  # Image files
                elif ext in ['.json', '.yaml', '.yml']:
                    color = "#F1E05A"  # Config files
                elif ext in ['.pt', '.pth']:
                    color = "#9B4DCA"  # PyTorch model files
                
                # Tambahkan file ke hasil
                result += f"{spaces}{prefix}<span style='color:{color};'>{item.name}</span>\n"
        
        # Tutup tag jika pada level teratas
        if indent == 0:
            result += "</pre>"
        
        return result
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi sistem komprehensif.
        
        Returns:
            Dictionary detail sistem
        """
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
            import platform
            import psutil
            
            info['system'] = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'ram_gb': round(psutil.virtual_memory().total / (1024**3), 2)
            }
        except ImportError:
            info['system'] = {'platform': platform.platform() if 'platform' in sys.modules else 'Unknown'}
            
        return info
    
    def install_requirements(self, requirements_file: Optional[Union[str, Path]] = None,
                           additional_packages: Optional[List[str]] = None,
                           progress_callback: Optional[Callable[[int, int, str], None]] = None) -> bool:
        """
        Install requirement packages menggunakan pip.
        
        Args:
            requirements_file: Path ke file requirements.txt
            additional_packages: List package tambahan untuk diinstall
            progress_callback: Callback untuk menampilkan progres (current, total, message)
            
        Returns:
            Status keberhasilan instalasi
        """
        import subprocess
        
        success = True
        steps_completed = 0
        total_steps = (1 if requirements_file else 0) + (1 if additional_packages else 0)
        
        # Install dari file requirements
        if requirements_file:
            req_path = Path(requirements_file)
            if req_path.exists():
                if self._logger:
                    self._logger.info(f"📦 Menginstall packages dari {req_path}...")
                
                if progress_callback:
                    progress_callback(1, total_steps, f"Menginstall packages dari {req_path}")
                
                cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    if self._logger:
                        self._logger.error(f"❌ Error instalasi requirements: {process.stderr}")
                    success = False
                else:
                    if self._logger:
                        self._logger.info(f"✅ Requirements berhasil diinstall")
                    steps_completed += 1
            else:
                if self._logger:
                    self._logger.warning(f"⚠️ File requirements tidak ditemukan: {req_path}")
                success = False
        
        # Install package tambahan
        if additional_packages:
            if self._logger:
                self._logger.info(f"📦 Menginstall package tambahan: {', '.join(additional_packages)}")
            
            if progress_callback:
                progress_callback(steps_completed + 1, total_steps, f"Menginstall package tambahan")
                
            cmd = [sys.executable, "-m", "pip", "install"] + additional_packages
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                if self._logger:
                    self._logger.error(f"❌ Error instalasi package tambahan: {process.stderr}")
                success = False
            else:
                if self._logger:
                    self._logger.info(f"✅ Package tambahan berhasil diinstall")
                    
        return success
    
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
            if get_ipython() is not None:
                return True
            return False
        except ImportError:
            return False

def get_environment_manager(base_dir: Optional[str] = None, logger = None) -> EnvironmentManager:
    """
    Dapatkan instance singleton EnvironmentManager.
    
    Args:
        base_dir: Direktori dasar proyek
        logger: Instance logger
    
    Returns:
        Singleton EnvironmentManager
    """
    return EnvironmentManager(base_dir, logger)