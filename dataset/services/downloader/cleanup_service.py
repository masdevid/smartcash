"""
File: smartcash/dataset/services/downloader/cleanup_service.py
Deskripsi: Layanan untuk membersihkan dan menghapus dataset
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import time

from smartcash.common.logger import get_logger
from smartcash.dataset.services.downloader.notification_utils import notify_service_event

class CleanupService:
    """Layanan untuk membersihkan dan menghapus dataset."""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None, observer_manager=None):
        """
        Inisialisasi CleanupService.
        
        Args:
            config: Konfigurasi service
            logger: Logger untuk logging
            observer_manager: Observer manager untuk notifikasi
        """
        self.config = config or {}
        self.logger = logger or get_logger()
        self.observer_manager = observer_manager
        
    def set_observer_manager(self, observer_manager):
        """Set observer manager untuk notifikasi UI."""
        self.observer_manager = observer_manager
    
    def cleanup_dataset(self, dataset_path: Union[str, Path], backup_before_delete: bool = True, 
                       show_progress: bool = True) -> Dict[str, Any]:
        """
        Hapus dataset dari direktori yang ditentukan.
        
        Args:
            dataset_path: Path ke dataset yang akan dihapus
            backup_before_delete: Buat backup sebelum menghapus
            show_progress: Tampilkan progress
            
        Returns:
            Dict dengan status operasi
        """
        start_time = time.time()
        dataset_path = Path(dataset_path)
        
        # Notifikasi start
        notify_service_event(
            "cleanup", 
            "start", 
            self, 
            self.observer_manager,
            message=f"Memulai penghapusan dataset: {dataset_path.name}"
        )
        
        # Validasi path
        if not dataset_path.exists():
            self.logger.warning(f"⚠️ Dataset tidak ditemukan: {dataset_path}")
            notify_service_event(
                "cleanup", 
                "error", 
                self, 
                self.observer_manager,
                message=f"Dataset tidak ditemukan: {dataset_path.name}"
            )
            return {
                "status": "error",
                "message": f"Dataset tidak ditemukan: {dataset_path}"
            }
        
        try:
            # Buat backup jika diminta
            if backup_before_delete:
                self.logger.info(f"💾 Membuat backup sebelum menghapus: {dataset_path}")
                notify_service_event(
                    "cleanup", 
                    "progress", 
                    self, 
                    self.observer_manager,
                    message=f"Membuat backup dataset: {dataset_path.name}",
                    progress=25,
                    total=100,
                    step="backup",
                    current_step=1,
                    total_steps=3
                )
                
                # Import backup service secara lazy untuk menghindari circular import
                from smartcash.dataset.services.downloader.backup_service import BackupService
                backup_service = BackupService(
                    config=self.config,
                    logger=self.logger,
                    observer_manager=self.observer_manager
                )
                
                # Buat backup
                backup_result = backup_service.backup_dataset(
                    dataset_path, 
                    show_progress=show_progress
                )
                
                if backup_result["status"] != "success":
                    self.logger.warning(f"⚠️ Gagal membuat backup: {backup_result['message']}")
                    # Lanjutkan proses penghapusan meskipun backup gagal
            
            # Hitung ukuran dan jumlah file untuk progress
            total_size = 0
            file_count = 0
            
            if show_progress:
                self.logger.info(f"📊 Menghitung ukuran dataset: {dataset_path}")
                notify_service_event(
                    "cleanup", 
                    "progress", 
                    self, 
                    self.observer_manager,
                    message=f"Menghitung ukuran dataset: {dataset_path.name}",
                    progress=50,
                    total=100,
                    step="count",
                    current_step=2,
                    total_steps=3
                )
                
                for root, dirs, files in os.walk(str(dataset_path)):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(file_path)
                            file_count += 1
                        except (OSError, FileNotFoundError):
                            pass
            
            # Hapus dataset
            self.logger.info(f"🗑️ Menghapus dataset: {dataset_path}")
            notify_service_event(
                "cleanup", 
                "progress", 
                self, 
                self.observer_manager,
                message=f"Menghapus dataset: {dataset_path.name}",
                progress=75,
                total=100,
                step="delete",
                current_step=3,
                total_steps=3
            )
            
            # Gunakan shutil.rmtree untuk menghapus direktori dan isinya
            if dataset_path.is_symlink():
                os.unlink(str(dataset_path))
            else:
                shutil.rmtree(dataset_path, ignore_errors=True)
            
            # Verifikasi penghapusan
            if dataset_path.exists():
                self.logger.error(f"❌ Gagal menghapus dataset: {dataset_path}")
                notify_service_event(
                    "cleanup", 
                    "error", 
                    self, 
                    self.observer_manager,
                    message=f"Gagal menghapus dataset: {dataset_path.name}"
                )
                return {
                    "status": "error",
                    "message": f"Gagal menghapus dataset: {dataset_path}"
                }
            
            # Notifikasi complete
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"✅ Dataset berhasil dihapus: {dataset_path} ({duration:.2f}s)")
            notify_service_event(
                "cleanup", 
                "complete", 
                self, 
                self.observer_manager,
                message=f"Dataset berhasil dihapus: {dataset_path.name}",
                duration=duration,
                file_count=file_count,
                total_size=total_size
            )
            
            return {
                "status": "success",
                "message": f"Dataset berhasil dihapus: {dataset_path}",
                "duration": duration,
                "file_count": file_count,
                "total_size": total_size
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error saat menghapus dataset: {str(e)}")
            notify_service_event(
                "cleanup", 
                "error", 
                self, 
                self.observer_manager,
                message=f"Error saat menghapus dataset: {str(e)}"
            )
            return {
                "status": "error",
                "message": f"Error saat menghapus dataset: {str(e)}"
            }
    
    def cleanup_multiple_datasets(self, dataset_paths: List[Union[str, Path]], 
                                backup_before_delete: bool = True,
                                show_progress: bool = True) -> Dict[str, Any]:
        """
        Hapus multiple dataset.
        
        Args:
            dataset_paths: List path ke dataset yang akan dihapus
            backup_before_delete: Buat backup sebelum menghapus
            show_progress: Tampilkan progress
            
        Returns:
            Dict dengan status operasi
        """
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        # Notifikasi start
        notify_service_event(
            "cleanup", 
            "start", 
            self, 
            self.observer_manager,
            message=f"Memulai penghapusan {len(dataset_paths)} dataset"
        )
        
        # Hapus dataset satu per satu
        for i, dataset_path in enumerate(dataset_paths):
            # Update progress
            progress = int((i / len(dataset_paths)) * 100)
            notify_service_event(
                "cleanup", 
                "progress", 
                self, 
                self.observer_manager,
                message=f"Menghapus dataset {i+1}/{len(dataset_paths)}: {Path(dataset_path).name}",
                progress=progress,
                total=100,
                step="delete",
                current_step=i+1,
                total_steps=len(dataset_paths)
            )
            
            # Hapus dataset
            result = self.cleanup_dataset(
                dataset_path, 
                backup_before_delete=backup_before_delete,
                show_progress=show_progress
            )
            
            results.append(result)
            
            if result["status"] == "success":
                success_count += 1
            else:
                error_count += 1
        
        # Notifikasi complete
        end_time = time.time()
        duration = end_time - start_time
        
        self.logger.info(f"✅ {success_count}/{len(dataset_paths)} dataset berhasil dihapus ({duration:.2f}s)")
        
        if error_count > 0:
            self.logger.warning(f"⚠️ {error_count}/{len(dataset_paths)} dataset gagal dihapus")
        
        notify_service_event(
            "cleanup", 
            "complete", 
            self, 
            self.observer_manager,
            message=f"{success_count}/{len(dataset_paths)} dataset berhasil dihapus",
            duration=duration,
            success_count=success_count,
            error_count=error_count
        )
        
        return {
            "status": "success" if error_count == 0 else "partial",
            "message": f"{success_count}/{len(dataset_paths)} dataset berhasil dihapus",
            "duration": duration,
            "success_count": success_count,
            "error_count": error_count,
            "results": results
        } 