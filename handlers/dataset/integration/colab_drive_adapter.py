# File: smartcash/handlers/dataset/integration/colab_drive_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan Google Drive di Colab

import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, List, Union, Tuple, Callable

from smartcash.utils.logger import SmartCashLogger


class ColabDriveAdapter:
    """
    Adapter untuk mengintegrasikan SmartCash dengan Google Drive di Google Colab.
    
    Menyediakan fitur:
    - Deteksi otomatis penggunaan di Google Colab
    - Mount/unmount Google Drive
    - Konversi path lokal ke path Google Drive
    - Dukungan untuk symlink di Google Drive
    """
    
    def __init__(
        self,
        mount_path: str = "/content/drive/MyDrive/SmartCash",
        local_path: str = "/content/SmartCash",
        auto_mount: bool = True,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi adapter Google Drive.
        
        Args:
            mount_path: Path Google Drive setelah di-mount
            local_path: Path lokal untuk symlink
            auto_mount: Jika True, otomatis mount drive saat inisialisasi
            logger: Logger kustom (opsional)
        """
        self.logger = logger or SmartCashLogger("colab_drive_adapter")
        self.mount_path = Path(mount_path)
        self.local_path = Path(local_path)
        
        # Cek apakah berjalan di Google Colab
        self.is_colab = self._check_colab()
        
        if self.is_colab:
            self.logger.info(f"🔍 Terdeteksi berjalan di Google Colab")
            if auto_mount:
                self.mount_drive()
        else:
            self.logger.info(f"🔍 Tidak berjalan di Google Colab, adapter hanya simulasi")
    
    def _check_colab(self) -> bool:
        """
        Cek apakah kode berjalan di Google Colab.
        
        Returns:
            True jika berjalan di Colab
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def mount_drive(self) -> bool:
        """
        Mount Google Drive di Colab.
        
        Returns:
            True jika berhasil
        """
        if not self.is_colab:
            self.logger.warning("⚠️ Tidak berjalan di Google Colab, tidak dapat mount drive")
            return False
            
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Cek apakah mount berhasil
            if Path("/content/drive/MyDrive").exists():
                self.logger.success("✅ Google Drive berhasil di-mount")
                
                # Buat direktori SmartCash jika belum ada
                if not self.mount_path.exists():
                    self.mount_path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"📁 Direktori {self.mount_path} berhasil dibuat")
                
                return True
            else:
                self.logger.error("❌ Gagal memverifikasi mount point Google Drive")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Gagal mount Google Drive: {str(e)}")
            return False
    
    def setup_symlinks(self, directories: List[str] = ['data', 'models', 'configs']) -> Dict[str, bool]:
        """
        Setup symlink dari direktori lokal ke Google Drive.
        
        Args:
            directories: List nama direktori yang akan di-symlink
            
        Returns:
            Dict status symlink untuk setiap direktori
        """
        if not self.is_colab:
            self.logger.warning("⚠️ Tidak berjalan di Google Colab, tidak dapat setup symlinks")
            return {dir: False for dir in directories}
        
        # Pastikan local_path ada
        self.local_path.mkdir(parents=True, exist_ok=True)
        
        # Buat symlink untuk setiap direktori
        results = {}
        for dir_name in directories:
            try:
                # Path di Google Drive
                drive_dir = self.mount_path / dir_name
                
                # Path lokal
                local_dir = self.local_path / dir_name
                
                # Buat direktori di Drive jika belum ada
                if not drive_dir.exists():
                    drive_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"📁 Direktori {drive_dir} berhasil dibuat di Drive")
                
                # Hapus symlink lokal jika sudah ada
                if local_dir.exists():
                    if local_dir.is_symlink():
                        os.unlink(local_dir)
                    else:
                        self.logger.warning(f"⚠️ {local_dir} sudah ada dan bukan symlink, mencadangkan...")
                        backup_path = Path(f"{local_dir}_backup")
                        if backup_path.exists():
                            shutil.rmtree(backup_path)
                        shutil.move(local_dir, backup_path)
                
                # Buat symlink
                os.symlink(drive_dir, local_dir, target_is_directory=True)
                self.logger.success(f"🔗 Symlink berhasil dibuat: {local_dir} -> {drive_dir}")
                results[dir_name] = True
                
            except Exception as e:
                self.logger.error(f"❌ Gagal membuat symlink untuk {dir_name}: {str(e)}")
                results[dir_name] = False
        
        return results
    
    def convert_path(self, path: Union[str, Path], to_drive: bool = True) -> Path:
        """
        Konversi path antara lokal dan Google Drive.
        
        Args:
            path: Path yang akan dikonversi
            to_drive: Jika True, konversi dari lokal ke Drive; jika False, sebaliknya
            
        Returns:
            Path yang dikonversi
        """
        path = Path(path)
        
        if to_drive:
            # Konversi path lokal ke path Drive
            if str(path).startswith(str(self.local_path)):
                relative_path = path.relative_to(self.local_path)
                return self.mount_path / relative_path
            return path
        else:
            # Konversi path Drive ke path lokal
            if str(path).startswith(str(self.mount_path)):
                relative_path = path.relative_to(self.mount_path)
                return self.local_path / relative_path
            return path
    
    def save_to_drive(self, source_path: Union[str, Path], target_dir: str) -> Optional[Path]:
        """
        Simpan file ke Google Drive.
        
        Args:
            source_path: Path file sumber
            target_dir: Direktori tujuan di Drive (relatif terhadap mount_path)
            
        Returns:
            Path hasil penyimpanan di Drive jika berhasil, None jika gagal
        """
        if not self.is_colab:
            self.logger.warning("⚠️ Tidak berjalan di Google Colab, tidak dapat menyimpan ke Drive")
            return None
        
        try:
            source_path = Path(source_path)
            if not source_path.exists():
                raise FileNotFoundError(f"File sumber tidak ditemukan: {source_path}")
            
            # Path target di Drive
            drive_target_dir = self.mount_path / target_dir
            drive_target_dir.mkdir(parents=True, exist_ok=True)
            
            drive_target_path = drive_target_dir / source_path.name
            
            # Salin file
            shutil.copy2(source_path, drive_target_path)
            self.logger.success(f"✅ File berhasil disimpan ke Drive: {drive_target_path}")
            
            return drive_target_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan file ke Drive: {str(e)}")
            return None
    
    def with_drive_paths(self, func: Callable, *args, **kwargs) -> Any:
        """
        Decorator-like function untuk menjalankan fungsi dengan path yang dikonversi ke Drive.
        
        Args:
            func: Fungsi yang akan dijalankan
            *args, **kwargs: Argumen untuk fungsi
            
        Returns:
            Hasil dari func
        """
        if not self.is_colab:
            return func(*args, **kwargs)
        
        # Konversi semua path dalam args dan kwargs
        new_args = [self.convert_path(arg) if isinstance(arg, (str, Path)) else arg for arg in args]
        new_kwargs = {k: self.convert_path(v) if isinstance(v, (str, Path)) else v for k, v in kwargs.items()}
        
        return func(*new_args, **new_kwargs)