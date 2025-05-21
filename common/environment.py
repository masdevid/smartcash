"""
File: smartcash/common/environment.py
Deskripsi: Manajer lingkungan terpusat untuk deteksi dan konfigurasi environment aplikasi SmartCash
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

def get_default_base_dir():
    """
    Dapatkan direktori dasar default untuk aplikasi.
    
    Returns:
        str: Path direktori dasar
    """
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        return "/content"
    return str(Path.home() / "SmartCash")

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
            cls._instance._is_syncing = False  # Flag untuk mencegah rekursi sinkronisasi
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
            self._file_utils = file_utils
            
            # Initialize logger properly
            if logger is None:
                self.logger = get_logger()
            else:
                self.logger = logger
                
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
            else Path('/content') if self._in_colab 
            else Path(os.getcwd())
        )
        
        # Auto-detect drive jika di Colab
        if self._in_colab: self.detect_drive()
            
        self._initialized = True
        
        if self.logger:
            env_type = "Google Colab" if self._in_colab else "Notebook" if self._in_notebook else "Lokal"
            # Ubah ke debug untuk mengurangi verbose log
            self.logger.debug(f"🔍 Environment terdeteksi: {env_type}")
            self.logger.debug(f"📂 Direktori dasar: {self._base_dir}")
    
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
                # Set drive path ke direktori root Colab
                self._drive_mounted = True
                
                # PERBAIKAN: Gunakan /content sebagai direktori utama di Colab
                # Ini menghindari redundansi direktori /content/SmartCash/configs dan /content/configs
                if Path('/content').exists():  # Deteksi lingkungan Colab
                    self._drive_path = Path('/content')
                    if self.logger: self.logger.debug(f"✅ Menggunakan direktori Colab: {self._drive_path}")
                else:
                    # Fallback ke SmartCash di drive jika bukan di Colab
                    self._drive_path = Path(drive_path) / 'SmartCash'
                    # Pastikan directory SmartCash ada di Drive
                    os.makedirs(self._drive_path, exist_ok=True)
                    if self.logger: self.logger.debug(f"✅ Google Drive terdeteksi pada: {self._drive_path}")
                
                return True
        except ImportError:
            # Fallback jika drive_utils tidak tersedia
            drive_path = Path('/content/drive/MyDrive/SmartCash')
            drive_mount_point = Path('/content/drive/MyDrive')
            
            if drive_mount_point.exists():
                # Pastikan directory SmartCash ada di Drive
                self._drive_mounted = True
                self._drive_path = drive_path
                os.makedirs(self._drive_path, exist_ok=True)
                if self.logger: self.logger.debug(f"✅ Google Drive terdeteksi pada: {self._drive_path}")
                
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
            if self.logger: self.logger.debug(msg)
            
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
        stats = {'created': 0, 'existing': 0, 'error': 0, 'preserved': 0}
        total_dirs = len(directories)
        
        # Fungsi sederhana untuk membuat direktori dengan mempertahankan file yang sudah ada
        def ensure_dir(path):
            try:
                # Cek apakah direktori sudah ada
                if path.exists() and path.is_dir():
                    return path
                    
                # Jika direktori belum ada, buat direktori baru
                os.makedirs(path, exist_ok=True)
                return path
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Error pembuatan direktori: {path} - {str(e)}")
                raise e
        
        # Iterasi dan buat setiap direktori
        for i, dir_path in enumerate(directories):
            try:
                # Report progress
                if progress_callback:
                    progress_callback(i, total_dirs, f"Creating directory: {dir_path}")
                
                # Buat direktori
                path = base / dir_path
                if path.exists():
                    stats['existing'] += 1
                else:
                    ensure_dir(path)
                    stats['created'] += 1
            except Exception as e:
                stats['error'] += 1
                if self.logger:
                    self.logger.error(f"❌ Error saat membuat direktori {dir_path}: {str(e)}")
        
        # Report final progress
        if progress_callback:
            progress_callback(total_dirs, total_dirs, "Struktur direktori selesai dibuat")
        
        return stats
    
    def create_symlinks(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict[str, int]:
        """
        Buat symlinks dari Drive ke direktori lokal di Colab.
        
        Args:
            progress_callback: Callback untuk menampilkan progres
            
        Returns:
            Statistik pembuatan symlink
        """
        if not self._in_colab or not self._drive_mounted:
            return {'created': 0, 'existing': 0, 'error': 0}
        
        # Direktori yang akan di-symlink
        dirs_to_link = ['data', 'models', 'logs', 'exports', 'configs']
        stats = {'created': 0, 'existing': 0, 'error': 0}
        
        for i, dirname in enumerate(dirs_to_link):
            try:
                # Report progress
                if progress_callback:
                    progress_callback(i, len(dirs_to_link), f"Creating symlink for: {dirname}")
                
                # Path sumber dan target
                src_path = self._drive_path / dirname
                target_path = self._base_dir / dirname
                
                # Pastikan direktori sumber ada
                if not src_path.exists():
                    src_path.mkdir(parents=True, exist_ok=True)
                
                # Jika target sudah ada dan bukan symlink, backup dan hapus
                if target_path.exists() and not target_path.is_symlink():
                    backup_path = self._base_dir / f"{dirname}_backup"
                    if backup_path.exists():
                        import shutil
                        shutil.rmtree(backup_path)
                    target_path.rename(backup_path)
                    if self.logger:
                        self.logger.info(f"🔄 Direktori {dirname} di-backup ke {dirname}_backup")
                
                # Buat symlink jika belum ada
                if not target_path.exists():
                    target_path.symlink_to(src_path, target_is_directory=True)
                    stats['created'] += 1
                    if self.logger:
                        self.logger.info(f"🔗 Symlink dibuat: {target_path} -> {src_path}")
                else:
                    stats['existing'] += 1
            except Exception as e:
                stats['error'] += 1
                if self.logger:
                    self.logger.error(f"❌ Error saat membuat symlink {dirname}: {str(e)}")
        
        # Report final progress
        if progress_callback:
            progress_callback(len(dirs_to_link), len(dirs_to_link), "Symlinks selesai dibuat")
        
        return stats
    
    def get_directory_tree(self, root_dir: Optional[Union[str, Path]] = None, max_depth: int = 3, 
                         indent: int = 0, _current_depth: int = 0) -> str:
        """
        Dapatkan representasi tree dari direktori.
        
        Args:
            root_dir: Direktori root (default: base_dir)
            max_depth: Kedalaman maksimum tree
            indent: Indentasi awal
            _current_depth: Kedalaman saat ini (untuk rekursi internal)
            
        Returns:
            String representasi tree
        """
        if root_dir is None:
            root_dir = self._base_dir
        else:
            root_dir = Path(root_dir)
            
        if not root_dir.exists():
            return f"Directory not found: {root_dir}"
            
        if _current_depth > max_depth:
            return "..."
            
        result = []
        
        try:
            # Get all entries in the directory
            entries = list(root_dir.iterdir())
            entries.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for i, entry in enumerate(entries):
                is_last = i == len(entries) - 1
                prefix = "└── " if is_last else "├── "
                full_prefix = "    " * indent + prefix
                
                # Add color based on file type
                if entry.is_dir():
                    color = self._get_color_for_extension("dir")
                    entry_str = f"{full_prefix}<span style='color:{color}'>{entry.name}/</span>"
                else:
                    ext = entry.suffix.lower()
                    color = self._get_color_for_extension(ext)
                    entry_str = f"{full_prefix}<span style='color:{color}'>{entry.name}</span>"
                
                result.append(entry_str)
                
                if entry.is_dir():
                    next_indent = indent + 1
                    next_prefix = "    " if is_last else "│   "
                    subtree = self.get_directory_tree(
                        entry, max_depth, next_indent, _current_depth + 1
                    )
                    if subtree != "...":
                        result.append(subtree)
                        
            return "\n".join(result)
        except Exception as e:
            return f"Error reading directory: {str(e)}"
    
    def _get_color_for_extension(self, ext: str) -> str:
        """Get color for file extension."""
        colors = {
            "dir": "#4285F4",  # Blue for directories
            ".py": "#0F9D58",  # Green for Python
            ".ipynb": "#F4B400",  # Yellow for notebooks
            ".md": "#DB4437",  # Red for markdown
            ".txt": "#4285F4",  # Blue for text
            ".yaml": "#AA46BB",  # Purple for yaml
            ".yml": "#AA46BB",  # Purple for yml
            ".json": "#F26522",  # Orange for json
        }
        return colors.get(ext, "#000000")  # Default black
    
    def sync_config(self) -> Tuple[bool, str]:
        """
        Sinkronisasi konfigurasi antara lokal dan drive.
        
        Returns:
            Tuple dari (sukses, pesan)
        """
        # Cek apakah di Colab dan drive terhubung
        if not self._in_colab or not self._drive_mounted:
            return False, "Sinkronisasi hanya tersedia di Google Colab dengan Drive terhubung"
        
        try:
            # Import komponen UI jika tersedia
            try:
                from smartcash.ui.utils.alert_utils import create_info_alert
                from IPython.display import display
                
                # Tampilkan pesan informasi sinkronisasi
                display(create_info_alert(
                    "Sinkronisasi Config",
                    "info",
                    "🔄"
                ))
            except ImportError:
                pass
                
            # Dapatkan direktori konfigurasi
            local_config_dir = self._base_dir / "configs"
            drive_config_dir = self._drive_path / "configs"
            
            # Pastikan kedua direktori ada
            os.makedirs(local_config_dir, exist_ok=True)
            os.makedirs(drive_config_dir, exist_ok=True)
            
            # Buat symlink jika belum ada
            if not local_config_dir.is_symlink():
                # Backup direktori lokal jika ada
                if local_config_dir.exists():
                    backup_dir = self._base_dir / "configs_backup"
                    if backup_dir.exists():
                        import shutil
                        shutil.rmtree(backup_dir)
                    import shutil
                    shutil.move(local_config_dir, backup_dir)
                    if self.logger:
                        self.logger.info(f"🔄 Direktori konfigurasi di-backup ke configs_backup")
                
                # Buat symlink
                os.symlink(drive_config_dir, local_config_dir)
                if self.logger:
                    self.logger.info(f"🔗 Symlink konfigurasi dibuat: {local_config_dir} -> {drive_config_dir}")
            
            # Tampilkan ringkasan
            try:
                from smartcash.ui.utils.metric_utils import create_metric_display
                from IPython.display import display
                
                # Hitung jumlah file
                local_files = list(local_config_dir.glob("*.yaml"))
                drive_files = list(drive_config_dir.glob("*.yaml"))
                
                # Tampilkan metrik
                display(create_metric_display("Config files", len(local_files)))
                display(create_metric_display("Status", "Synced", is_good=True))
            except ImportError:
                pass
                
            return True, "Konfigurasi berhasil disinkronkan"
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat sinkronisasi konfigurasi: {str(e)}")
            return False, f"Error saat sinkronisasi konfigurasi: {str(e)}"
    
    def save_environment_config(self) -> Tuple[bool, str]:
        """
        Simpan konfigurasi environment ke file.
        
        Returns:
            Tuple dari (sukses, pesan)
        """
        try:
            # Dapatkan config manager
            from smartcash.common.config import get_config_manager
            config_manager = get_config_manager()
            
            # Dapatkan informasi sistem
            env_info = self.get_system_info()
            
            # Tambahkan informasi drive
            if self._drive_mounted:
                config_manager.set('drive_mounted', True)
                config_manager.set('drive_path', str(self._drive_path))
            
            # Simpan konfigurasi ke file menggunakan metode yang tersedia
            try:
                # Simpan sebagai konfigurasi modul 'environment'
                config_manager.save_module_config('environment', {
                    'environment': env_info,
                    'base_dir': str(self._base_dir),
                    'dataset_path': str(self._base_dir / 'data'),
                    'model_path': str(self._base_dir / 'models'),
                    'config_path': str(self._base_dir / 'configs'),
                    'drive_mounted': self._drive_mounted,
                    'drive_path': str(self._drive_path) if self._drive_mounted else None
                })
            except Exception as save_error:
                if self.logger:
                    self.logger.warning(f"⚠️ Error saat menyimpan konfigurasi modul: {str(save_error)}")
            
            # Tampilkan ringkasan dengan format yang sama dengan dependency installer
            try:
                from smartcash.ui.utils.alert_utils import create_info_alert
                from smartcash.ui.utils.metric_utils import create_metric_display
                from IPython.display import display
                
                # Header ringkasan
                display(create_info_alert(
                    "Konfigurasi Environment",
                    "success",
                    "✅"
                ))
                
                # Metrik
                display(create_metric_display("Environment", "Google Colab" if self._in_colab else "Local"))
                display(create_metric_display("Base Directory", str(self._base_dir)))
                display(create_metric_display("Drive Mounted", "Yes" if self._drive_mounted else "No", is_good=self._drive_mounted))
                if self._drive_mounted:
                    display(create_metric_display("Drive Path", str(self._drive_path)))
            except ImportError:
                # Fallback jika komponen UI tidak tersedia
                pass
            
            if self.logger:
                self.logger.info("✅ Konfigurasi environment berhasil disimpan")
            
            return True, "Konfigurasi environment berhasil disimpan"
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saat menyimpan konfigurasi environment: {str(e)}")
            return False, f"Error saat menyimpan konfigurasi environment: {str(e)}"
    
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
            
    def _sync_config_files_on_drive_connect(self) -> None:
        """
        Sinkronisasi otomatis file **_config.yaml saat drive terhubung.
        """
        # Cek apakah drive terhubung
        if not self._drive_mounted:
            return
            
        try:
            # Tampilkan informasi sinkronisasi dengan format yang sama dengan dependency installer
            try:
                from smartcash.ui.utils.alert_utils import create_info_alert
                from IPython.display import display
                
                # Tampilkan pesan informasi sinkronisasi
                display(create_info_alert(
                    "Sinkronisasi Config",
                    "info",
                    "🔄"
                ))
            except ImportError:
                pass
            
            # Lakukan sinkronisasi sederhana dengan symlink
            self.sync_config()
            
            if self.logger:
                self.logger.info("✅ Sinkronisasi config berhasil")
                
        except Exception as sync_error:
            if self.logger:
                self.logger.warning(f"⚠️ Error saat sinkronisasi otomatis: {str(sync_error)}")
    
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