"""
File: smartcash/dataset/services/downloader/cleanup_service.py
Deskripsi: Updated cleanup service dengan Drive integration dan progress callback
"""

import os, shutil, time
from pathlib import Path
from typing import Dict, Any, Optional, Union, Callable

from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager

class CleanupService:
    """Cleanup service dengan Drive storage awareness."""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None, observer_manager=None):
        self.config = config or {}
        self.logger = logger or get_logger()
        self.observer_manager = observer_manager
        self._progress_callback: Optional[Callable] = None
        
        # Environment manager untuk Drive detection
        self.env_manager = get_environment_manager()
    
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
    
    def cleanup_dataset(self, dataset_path: Union[str, Path], backup_before_delete: bool = True, 
                       show_progress: bool = True) -> Dict[str, Any]:
        """Cleanup dataset dengan Drive awareness dan progress callback."""
        start_time = time.time()
        dataset_path = Path(dataset_path)
        
        self.logger.info(f"🧹 Memulai cleanup: {dataset_path}")
        self._notify_progress("cleanup", 0, 100, f"Memulai cleanup: {dataset_path.name}")
        
        if not dataset_path.exists():
            msg = f"Dataset tidak ditemukan: {dataset_path}"
            self.logger.warning(f"⚠️ {msg}")
            return {"status": "error", "message": msg}
        
        try:
            # Check if it's Drive storage
            is_drive_storage = self._is_drive_path(dataset_path)
            storage_type = "Drive" if is_drive_storage else "Local"
            
            self.logger.info(f"📍 Storage type: {storage_type}")
            
            # Backup jika diminta
            if backup_before_delete:
                self._notify_progress("cleanup", 10, 100, "Membuat backup sebelum cleanup")
                backup_result = self._create_backup_before_cleanup(dataset_path)
                
                if backup_result.get("status") != "success":
                    self.logger.warning(f"⚠️ Backup gagal: {backup_result.get('message')}")
                    # Lanjutkan cleanup meskipun backup gagal
            
            # Count files untuk progress
            self._notify_progress("cleanup", 30, 100, "Menghitung file yang akan dihapus")
            file_count, total_size = self._count_files(dataset_path)
            
            if file_count == 0:
                msg = f"Tidak ada file untuk dihapus di {dataset_path}"
                self.logger.info(f"ℹ️ {msg}")
                return {"status": "success", "message": msg, "file_count": 0}
            
            # Delete files dengan progress
            self._notify_progress("cleanup", 50, 100, f"Menghapus {file_count} file")
            deleted_count = self._delete_files_with_progress(dataset_path, file_count)
            
            # Handle symlinks untuk Drive storage
            if is_drive_storage:
                self._cleanup_colab_symlinks(dataset_path)
            
            # Verify deletion
            self._notify_progress("cleanup", 90, 100, "Memverifikasi penghapusan")
            if dataset_path.exists():
                # Force removal jika masih ada
                try:
                    shutil.rmtree(dataset_path, ignore_errors=True)
                    if dataset_path.exists():
                        raise Exception("Direktori masih ada setelah penghapusan")
                except Exception as e:
                    return {"status": "error", "message": f"Gagal menghapus direktori: {str(e)}"}
            
            duration = time.time() - start_time
            self._notify_progress("cleanup", 100, 100, f"Cleanup selesai: {deleted_count} file dihapus")
            
            self.logger.success(
                f"✅ Cleanup selesai ({duration:.1f}s)\n"
                f"   • Storage: {storage_type}\n"
                f"   • File dihapus: {deleted_count}\n"
                f"   • Total size: {total_size / (1024*1024):.2f} MB"
            )
            
            return {
                "status": "success", "message": f"Dataset berhasil dihapus: {dataset_path}",
                "duration": duration, "file_count": deleted_count, "total_size": total_size,
                "storage_type": storage_type
            }
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {str(e)}")
            self._notify_progress("cleanup", 0, 100, f"Error: {str(e)}")
            return {"status": "error", "message": f"Cleanup gagal: {str(e)}"}
    
    def _is_drive_path(self, path: Path) -> bool:
        """Check apakah path berada di Drive."""
        if not self.env_manager.is_colab or not self.env_manager.drive_path:
            return False
        
        try:
            # Resolve symlinks
            resolved_path = path.resolve()
            drive_path = self.env_manager.drive_path.resolve()
            
            # Check jika path berada dalam Drive
            return str(resolved_path).startswith(str(drive_path))
        except Exception:
            return False
    
    def _create_backup_before_cleanup(self, dataset_path: Path) -> Dict[str, Any]:
        """Create backup sebelum cleanup."""
        try:
            from smartcash.dataset.services.downloader.backup_service import BackupService
            
            backup_service = BackupService(self.config, self.logger)
            if self._progress_callback:
                backup_service.set_progress_callback(self._progress_callback)
            
            return backup_service.backup_dataset(dataset_path, show_progress=True)
        except Exception as e:
            return {"status": "error", "message": f"Backup error: {str(e)}"}
    
    def _count_files(self, dataset_path: Path) -> tuple[int, int]:
        """Count files dan total size."""
        file_count = 0
        total_size = 0
        
        try:
            for root, dirs, files in os.walk(str(dataset_path)):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_stat = os.stat(file_path)
                        file_count += 1
                        total_size += file_stat.st_size
                    except (OSError, FileNotFoundError):
                        pass
        except Exception:
            pass
        
        return file_count, total_size
    
    def _delete_files_with_progress(self, dataset_path: Path, total_files: int) -> int:
        """Delete files dengan progress callback."""
        deleted_count = 0
        
        try:
            for root, dirs, files in os.walk(str(dataset_path), topdown=False):
                # Delete files
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        
                        # Progress update setiap 10%
                        if deleted_count % max(1, total_files // 10) == 0:
                            progress = 50 + int((deleted_count / total_files) * 30)
                            self._notify_progress("cleanup", progress, 100, 
                                                f"Dihapus: {deleted_count}/{total_files}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal hapus {file_path}: {str(e)}")
                
                # Delete empty directories
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    try:
                        if not os.listdir(dir_path):
                            os.rmdir(dir_path)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat delete files: {str(e)}")
        
        return deleted_count
    
    def _cleanup_colab_symlinks(self, dataset_path: Path) -> None:
        """Cleanup symlinks di Colab yang mengarah ke Drive path."""
        if not self.env_manager.is_colab:
            return
        
        try:
            # Find potential symlink paths
            dataset_name = dataset_path.name
            potential_symlinks = [
                Path('/content') / dataset_name,
                Path('/content/data'),
                Path('/content/downloads') / dataset_name
            ]
            
            for symlink_path in potential_symlinks:
                if symlink_path.is_symlink():
                    try:
                        if symlink_path.resolve() == dataset_path.resolve():
                            symlink_path.unlink()
                            self.logger.info(f"🔗 Symlink dihapus: {symlink_path}")
                    except Exception:
                        pass
                        
        except Exception as e:
            self.logger.warning(f"⚠️ Error cleanup symlinks: {str(e)}")
    
    def cleanup_multiple_datasets(self, dataset_paths: list, backup_before_delete: bool = True,
                                 show_progress: bool = True) -> Dict[str, Any]:
        """Cleanup multiple datasets dengan progress tracking."""
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        self._notify_progress("cleanup_multi", 0, 100, f"Cleanup {len(dataset_paths)} dataset")
        
        for i, dataset_path in enumerate(dataset_paths):
            progress = int((i / len(dataset_paths)) * 100)
            self._notify_progress("cleanup_multi", progress, 100, 
                                f"Cleanup {i+1}/{len(dataset_paths)}: {Path(dataset_path).name}")
            
            result = self.cleanup_dataset(dataset_path, backup_before_delete, show_progress)
            results.append(result)
            
            if result["status"] == "success":
                success_count += 1
            else:
                error_count += 1
        
        duration = time.time() - start_time
        self._notify_progress("cleanup_multi", 100, 100, 
                            f"Multi-cleanup selesai: {success_count}/{len(dataset_paths)} berhasil")
        
        self.logger.info(f"✅ Multi-cleanup: {success_count}/{len(dataset_paths)} berhasil ({duration:.1f}s)")
        
        return {
            "status": "success" if error_count == 0 else "partial",
            "message": f"{success_count}/{len(dataset_paths)} dataset berhasil dihapus",
            "duration": duration, "success_count": success_count, "error_count": error_count,
            "results": results
        }