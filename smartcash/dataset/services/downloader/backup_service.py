"""
File: smartcash/dataset/services/downloader/backup_service.py
Deskripsi: Layanan backup dataset ringkas dengan fitur kompres ke ZIP
"""

import os
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union, Any
from datetime import datetime
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetError
from smartcash.components.observer.manager_observer import ObserverManager
from smartcash.components.observer import notify, EventTopics


class BackupService:
    """Layanan untuk membackup dataset ke file ZIP."""
    
    def __init__(self, backup_dir: Optional[str] = None, logger=None):
        """
        Inisialisasi BackupService.
        
        Args:
            backup_dir: Direktori untuk menyimpan backup (opsional)
            logger: Logger kustom (opsional)
        """
        self.logger = logger or get_logger("backup_service")
        self.backup_dir = Path(backup_dir or "data/backups")
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Setup observer manager untuk tracking - PERBAIKAN: Ubah dari get_instance() ke inisialisasi langsung
        try:
            self.observer_manager = ObserverManager()
        except (ImportError, AttributeError):
            self.observer_manager = None
        
        self.logger.info(f"🗄️ BackupService diinisialisasi, backup dir: {self.backup_dir}")
    
    def backup_dataset(
        self, 
        src_dir: Union[str, Path], 
        backup_name: Optional[str] = None,
        show_progress: bool = True,
        compression: int = zipfile.ZIP_DEFLATED
    ) -> Dict[str, Any]:
        """
        Buat backup dataset dalam format ZIP.
        
        Args:
            src_dir: Direktori sumber untuk backup
            backup_name: Nama custom untuk backup (opsional)
            show_progress: Tampilkan progress backup
            compression: Level kompresi ZIP (0-9, default: ZIP_DEFLATED)
            
        Returns:
            Dictionary berisi informasi backup
        """
        start_time = time.time()
        src_path = Path(src_dir)
        
        if not src_path.exists():
            raise DatasetError(f"❌ Direktori sumber tidak ditemukan: {src_path}")
        
        # Generate nama backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or src_path.name
        zip_path = self.backup_dir / f"{backup_name}_backup_{timestamp}.zip"
        
        # Siapkan untuk backup
        self.logger.info(f"📦 Membuat backup dataset: {src_path} → {zip_path}")
        
        # Notifikasi backup dimulai
        self._notify(
            EventTopics.BACKUP_START,
            message=f"Membuat backup dataset ke {zip_path.name}",
            source=str(src_path),
            destination=str(zip_path),
            status="info"
        )
        
        try:
            # Hitung file untuk progress bar
            files_to_backup = list(src_path.glob('**/*'))
            files_to_backup = [f for f in files_to_backup if f.is_file()]
            total_files = len(files_to_backup)
            total_size = sum(f.stat().st_size for f in files_to_backup)
            
            if total_files == 0:
                raise DatasetError(f"❌ Tidak ada file ditemukan untuk dibackup di {src_path}")
            
            # Buat ZIP file dengan progress tracking
            with zipfile.ZipFile(zip_path, 'w', compression) as zipf:
                with tqdm(total=total_files, desc=f"📦 Backup ke ZIP", unit="file", disable=not show_progress) as pbar:
                    for i, file in enumerate(files_to_backup):
                        # Tambahkan ke ZIP dengan path relatif
                        arcname = file.relative_to(src_path)
                        zipf.write(file, arcname)
                        pbar.update(1)
                        
                        # Notifikasi progress setiap 5% atau 20 file
                        if i % max(1, min(20, total_files // 20)) == 0:
                            self._notify(
                                EventTopics.BACKUP_PROGRESS,
                                progress=i,
                                total=total_files,
                                percentage=int((i / total_files) * 100),
                                status="info"
                            )
            
            # Dapatkan statistik hasil
            elapsed_time = time.time() - start_time
            backup_size = zip_path.stat().st_size
            backup_size_mb = backup_size / (1024 * 1024)
            compression_ratio = (1 - (backup_size / total_size)) * 100 if total_size > 0 else 0
            
            # Notifikasi selesai
            self._notify(
                EventTopics.BACKUP_COMPLETE,
                message=f"Backup selesai: {backup_size_mb:.2f} MB (rasio kompresi: {compression_ratio:.1f}%)",
                size_mb=backup_size_mb,
                file_count=total_files,
                duration=elapsed_time,
                status="success"
            )
            
            self.logger.success(
                f"✅ Backup selesai ({elapsed_time:.1f}s)\n"
                f"   • Backup ZIP: {zip_path}\n"
                f"   • File: {total_files} file\n"
                f"   • Ukuran: {backup_size_mb:.2f} MB (kompresi: {compression_ratio:.1f}%)"
            )
            
            return {
                "status": "success",
                "source_dir": str(src_path),
                "backup_path": str(zip_path),
                "file_count": total_files,
                "original_size_mb": total_size / (1024 * 1024),
                "backup_size_mb": backup_size_mb,
                "compression_ratio": compression_ratio,
                "duration": elapsed_time
            }
            
        except Exception as e:
            # Notifikasi error dan hapus backup tidak lengkap
            self._notify(
                EventTopics.BACKUP_ERROR,
                message=f"Error saat backup dataset: {str(e)}",
                status="error"
            )
                
            self.logger.error(f"❌ Error saat backup dataset: {str(e)}")
            
            # Hapus file ZIP yang mungkin rusak
            if zip_path.exists():
                try:
                    zip_path.unlink()
                except:
                    pass
                
            raise DatasetError(f"Error saat backup dataset: {str(e)}")
    
    def _notify(self, event_type, **kwargs):
        """Helper untuk mengirimkan notifikasi observer dengan one-liner."""
        if self.observer_manager: notify(event_type, self, **kwargs)
            
    def list_backups(self, filter_pattern: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Daftar semua backup ZIP yang tersedia.
        
        Args:
            filter_pattern: Pattern untuk memfilter nama backup (opsional)
            
        Returns:
            Dictionary berisi informasi backup
        """
        backups = {}
        
        try:
            # List semua file ZIP di backup_dir
            zip_files = list(self.backup_dir.glob("*.zip"))
            
            # Filter berdasarkan pattern jika ada
            if filter_pattern:
                import fnmatch
                zip_files = [z for z in zip_files if fnmatch.fnmatch(z.name, filter_pattern)]
            
            # Buat informasi untuk setiap backup
            for idx, backup_path in enumerate(sorted(zip_files, key=lambda x: x.stat().st_mtime, reverse=True)):
                try:
                    # Dapatkan metadata file
                    backup_size = backup_path.stat().st_size
                    backup_size_mb = backup_size / (1024 * 1024)
                    backup_time = datetime.fromtimestamp(backup_path.stat().st_mtime)
                    
                    # Hitung jumlah file dalam ZIP
                    with zipfile.ZipFile(backup_path, 'r') as zipf:
                        file_count = len(zipf.namelist())
                    
                    backups[backup_path.name] = {
                        "path": str(backup_path),
                        "size_mb": backup_size_mb,
                        "file_count": file_count,
                        "created_at": backup_time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                except Exception as e:
                    self.logger.debug(f"⚠️ Error saat mendapatkan info backup {backup_path.name}: {str(e)}")
                    backups[backup_path.name] = {
                        "path": str(backup_path),
                        "error": str(e)
                    }
            
            return backups
            
        except Exception as e:
            self.logger.error(f"❌ Error saat listing backup: {str(e)}")
            return {}