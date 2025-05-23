"""
File: smartcash/dataset/services/downloader/backup_service.py
Deskripsi: Updated backup service dengan Drive storage support
"""

import os, time, zipfile, shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.io import ensure_dir

class BackupService:
    """Backup service dengan Drive storage priority."""
    
    def __init__(self, config: Dict[str, Any], logger=None, observer_manager=None):
        self.config = config
        self.logger = logger or get_logger()
        self.observer_manager = observer_manager
        self._progress_callback: Optional[Callable] = None
        
        # Environment untuk Drive detection
        self.env_manager = get_environment_manager()
        
        # Setup backup directory dengan Drive priority
        self.backup_dir = self._setup_backup_dir()
        
        self.logger.info(f"💾 BackupService initialized: {self.backup_dir}")
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set progress callback."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def _setup_backup_dir(self) -> Path:
        """Setup backup directory dengan Drive priority."""
        if self.env_manager.is_colab and self.env_manager.is_drive_mounted:
            # Drive backup directory
            backup_dir = self.env_manager.drive_path / 'backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup symlink ke Colab
            colab_backup = Path('/content/backups')
            if colab_backup.exists():
                if colab_backup.is_symlink():
                    if colab_backup.resolve() != backup_dir.resolve():
                        colab_backup.unlink()
                        colab_backup.symlink_to(backup_dir)
                else:
                    shutil.rmtree(colab_backup, ignore_errors=True)
                    colab_backup.symlink_to(backup_dir)
            else:
                colab_backup.symlink_to(backup_dir)
            
            self.logger.info(f"💾 Backup akan disimpan di Drive: {backup_dir}")
            return backup_dir
        else:
            # Local backup
            backup_dir = Path(self.config.get('data', {}).get('backup_dir', 'data/backups'))
            ensure_dir(backup_dir)
            return backup_dir
    
    def backup_dataset(self, src_dir: Union[str, Path], backup_name: Optional[str] = None, 
                      show_progress: bool = True, compression: int = zipfile.ZIP_DEFLATED) -> Dict[str, Any]:
        """Backup dataset ke Drive dengan progress callback."""
        start_time = time.time()
        src_path = Path(src_dir)
        
        if not src_path.exists():
            msg = f"❌ Source tidak ditemukan: {src_path}"
            self.logger.warning(msg)
            return {"status": "error", "message": msg}
        
        # Count files
        files_to_backup = [f for f in src_path.glob('**/*') if f.is_file()]
        total_files = len(files_to_backup)
        
        if total_files == 0:
            msg = f"❌ Tidak ada file untuk backup: {src_path}"
            self.logger.warning(msg)
            return {"status": "empty", "message": msg}
        
        # Generate backup name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = backup_name or src_path.name
        zip_path = self.backup_dir / f"{backup_name}_backup_{timestamp}.zip"
        
        self.logger.info(f"💾 Membuat backup: {src_path} → {zip_path}")
        self._notify_progress("backup", 0, 100, f"Memulai backup ke {zip_path.name}")
        
        try:
            total_size = sum(f.stat().st_size for f in files_to_backup)
            
            # Create ZIP dengan progress
            with zipfile.ZipFile(zip_path, 'w', compression) as zipf:
                for i, file in enumerate(files_to_backup):
                    arcname = file.relative_to(src_path)
                    zipf.write(file, arcname)
                    
                    # Progress callback setiap 5%
                    if i % max(1, total_files // 20) == 0:
                        progress = int((i / total_files) * 100)
                        self._notify_progress("backup", progress, 100, f"Backup: {progress}%")
            
            # Final stats
            backup_size = zip_path.stat().st_size
            backup_size_mb = backup_size / (1024 * 1024)
            compression_ratio = (1 - (backup_size / total_size)) * 100 if total_size > 0 else 0
            elapsed_time = time.time() - start_time
            
            self._notify_progress("backup", 100, 100, f"Backup selesai: {backup_size_mb:.2f} MB")
            
            self.logger.success(
                f"✅ Backup selesai ({elapsed_time:.1f}s)\n"
                f"   • File: {total_files} file\n"
                f"   • Ukuran: {backup_size_mb:.2f} MB (kompresi: {compression_ratio:.1f}%)\n"
                f"   • Lokasi: {zip_path}"
            )
            
            return {
                "status": "success", "source_dir": str(src_path), "backup_path": str(zip_path),
                "file_count": total_files, "backup_size_mb": backup_size_mb,
                "compression_ratio": compression_ratio, "duration": elapsed_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Backup gagal: {str(e)}")
            self._notify_progress("backup", 0, 100, f"Error: {str(e)}")
            
            # Cleanup failed backup
            if zip_path.exists():
                zip_path.unlink()
            
            return {"status": "error", "message": f"Backup gagal: {str(e)}"}
    
    def cleanup_old_backups(self, max_backups: int = 3) -> None:
        """Cleanup old backups di Drive."""
        try:
            backup_files = list(self.backup_dir.glob("*_backup_*.zip"))
            if len(backup_files) <= max_backups:
                return
            
            # Sort by modification time
            backup_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Remove oldest backups
            for backup_file in backup_files[:-max_backups]:
                backup_file.unlink()
                self.logger.info(f"🗑️ Backup lama dihapus: {backup_file.name}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error cleanup backup: {str(e)}")
