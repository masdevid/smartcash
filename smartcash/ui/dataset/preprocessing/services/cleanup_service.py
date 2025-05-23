"""
File: smartcash/ui/dataset/preprocessing/services/cleanup_service.py
Deskripsi: Service untuk cleanup data preprocessing dengan 2-level progress
"""

from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.utils.drive_utils import (
    check_existing_preprocessing_data, safe_cleanup_preprocessing_data
)
from smartcash.ui.dataset.preprocessing.utils.progress_tracker import (
    create_progress_tracker, ProcessingSteps
)

logger = get_logger(__name__)

class CleanupService:
    """Service untuk cleanup data preprocessing dengan progress tracking."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi cleanup service.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', logger)
        
        # Setup progress tracker untuk cleanup
        self.progress_tracker = create_progress_tracker(ui_components)
        
        # State tracking
        self.is_running = False
        self.current_future: Optional[Future] = None
    
    def check_cleanup_requirements(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Cek requirements dan data yang akan dibersihkan.
        
        Returns:
            Tuple[bool, str, Dict]: (dapat_cleanup, message, data_info)
        """
        try:
            # Cek existing data
            existing_data = check_existing_preprocessing_data(self.ui_components)
            
            if not existing_data['exists']:
                return False, "Tidak ada data preprocessing yang perlu dibersihkan", existing_data
            
            # Format informasi data
            total_files = existing_data['total_files']
            total_size = existing_data['size_mb']
            
            message = f"Ditemukan {total_files:,} file ({total_size:.1f} MB) untuk dibersihkan"
            
            if existing_data['symlink_active']:
                message += " (Symlink aktif - data Drive akan aman)"
            
            return True, message, existing_data
            
        except Exception as e:
            self.logger.error(f"❌ Error cek cleanup requirements: {str(e)}")
            return False, f"Error: {str(e)}", {}
    
    def run_cleanup(self) -> Future:
        """
        Jalankan cleanup secara asynchronous.
        
        Returns:
            Future: Future object untuk tracking
        """
        if self.is_running:
            raise RuntimeError("Cleanup sudah berjalan")
        
        # Cek requirements
        can_cleanup, message, data_info = self.check_cleanup_requirements()
        if not can_cleanup:
            raise ValueError(message)
        
        # Set running state
        self.is_running = True
        self.ui_components['cleanup_running'] = True
        self.ui_components['stop_requested'] = False
        
        # Jalankan dengan ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._run_cleanup_task, data_info)
        self.current_future = future
        
        return future
    
    def _run_cleanup_task(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task untuk menjalankan cleanup (dijalankan di thread terpisah).
        
        Args:
            data_info: Informasi data yang akan dibersihkan
            
        Returns:
            Dict: Hasil cleanup
        """
        try:
            # Setup cleanup steps
            cleanup_steps = [
                ProcessingSteps.PREPARATION,
                ProcessingSteps.TRAIN_SPLIT,      # Cleanup train data
                ProcessingSteps.VAL_SPLIT,        # Cleanup val data  
                ProcessingSteps.TEST_SPLIT,       # Cleanup test data
                ProcessingSteps.FINALIZATION      # Cleanup empty dirs
            ]
            
            # Start progress tracking
            self.progress_tracker.start_processing(cleanup_steps)
            
            # Step 1: Preparation
            self.progress_tracker.start_step(ProcessingSteps.PREPARATION)
            self._preparation_step(data_info)
            self.progress_tracker.complete_step(ProcessingSteps.PREPARATION, "Persiapan cleanup selesai")
            
            # Step 2-4: Cleanup per split
            cleanup_stats = self._cleanup_splits(data_info)
            
            # Step 5: Finalization
            self.progress_tracker.start_step(ProcessingSteps.FINALIZATION)
            self._finalization_step(cleanup_stats)
            self.progress_tracker.complete_step(ProcessingSteps.FINALIZATION, "Finalisasi cleanup selesai")
            
            # Complete processing
            total_deleted = cleanup_stats.get('deleted_files', 0)
            self.progress_tracker.complete_processing(f"Cleanup selesai - {total_deleted} file dihapus")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"❌ Error cleanup task: {str(e)}")
            current_step = list(ProcessingSteps)[min(self.progress_tracker.current_step, len(ProcessingSteps)-1)]
            self.progress_tracker.handle_error(current_step, str(e))
            raise
        finally:
            self.is_running = False
            self.ui_components['cleanup_running'] = False
            self.current_future = None
    
    def _preparation_step(self, data_info: Dict[str, Any]) -> None:
        """
        Step persiapan cleanup.
        
        Args:
            data_info: Informasi data yang akan dibersihkan
        """
        self.logger.info("🔧 Mempersiapkan cleanup...")
        
        # Log informasi data
        total_files = data_info.get('total_files', 0)
        total_size = data_info.get('size_mb', 0)
        symlink_active = data_info.get('symlink_active', False)
        
        self.logger.info(f"📊 Data yang akan dibersihkan:")
        self.logger.info(f"  • Total file: {total_files:,}")
        self.logger.info(f"  • Total size: {total_size:.1f} MB")
        self.logger.info(f"  • Symlink aktif: {'Ya' if symlink_active else 'Tidak'}")
        
        # Update progress
        self.progress_tracker.update_step_progress(50, "Menganalisis struktur data...")
        
        # Cek splits yang ada
        splits = data_info.get('splits', {})
        if splits:
            self.logger.info(f"📁 Split yang ditemukan:")
            for split_name, split_data in splits.items():
                files = split_data.get('files', 0)
                size = split_data.get('size_mb', 0)
                self.logger.info(f"  • {split_name}: {files:,} file ({size:.1f} MB)")
        
        self.progress_tracker.update_step_progress(100, "Persiapan selesai")
    
    def _cleanup_splits(self, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleanup data per split dengan progress tracking.
        
        Args:
            data_info: Informasi data yang akan dibersihkan
            
        Returns:
            Dict: Statistik cleanup
        """
        # Buat progress callback untuk safe_cleanup
        def progress_callback(current: int, total: int, message: str) -> None:
            # Update progress berdasarkan step saat ini
            current_step_enum = list(ProcessingSteps)[self.progress_tracker.current_step - 1]
            
            # Map ke step name yang sesuai
            step_progress = (current / total * 100) if total > 0 else 0
            self.progress_tracker.update_step_progress(current, message)
        
        # Step 2: Cleanup Train Split
        self.progress_tracker.start_step(ProcessingSteps.TRAIN_SPLIT)
        self.progress_tracker.update_step_progress(0, "Membersihkan data training...")
        
        # Step 3: Cleanup Val Split  
        self.progress_tracker.complete_step(ProcessingSteps.TRAIN_SPLIT, "Data training dibersihkan")
        self.progress_tracker.start_step(ProcessingSteps.VAL_SPLIT)
        self.progress_tracker.update_step_progress(0, "Membersihkan data validasi...")
        
        # Step 4: Cleanup Test Split
        self.progress_tracker.complete_step(ProcessingSteps.VAL_SPLIT, "Data validasi dibersihkan")
        self.progress_tracker.start_step(ProcessingSteps.TEST_SPLIT) 
        self.progress_tracker.update_step_progress(0, "Membersihkan data testing...")
        
        # Jalankan cleanup sebenarnya dengan progress callback
        cleanup_stats = safe_cleanup_preprocessing_data(self.ui_components, progress_callback)
        
        self.progress_tracker.complete_step(ProcessingSteps.TEST_SPLIT, "Data testing dibersihkan")
        
        return cleanup_stats
    
    def _finalization_step(self, cleanup_stats: Dict[str, Any]) -> None:
        """
        Step finalisasi cleanup.
        
        Args:
            cleanup_stats: Statistik hasil cleanup
        """
        self.logger.info("🔄 Finalisasi cleanup...")
        
        # Update progress
        self.progress_tracker.update_step_progress(50, "Membersihkan direktori kosong...")
        
        # Log statistik cleanup
        deleted_files = cleanup_stats.get('deleted_files', 0)
        deleted_dirs = cleanup_stats.get('deleted_dirs', 0) 
        skipped_symlinks = cleanup_stats.get('skipped_symlinks', 0)
        errors = cleanup_stats.get('errors', [])
        
        self.logger.info(f"📊 Hasil cleanup:")
        self.logger.info(f"  • File dihapus: {deleted_files:,}")
        self.logger.info(f"  • Direktori dibersihkan: {deleted_dirs}")
        self.logger.info(f"  • Symlink dipertahankan: {skipped_symlinks}")
        
        if errors:
            self.logger.warning(f"  • Error: {len(errors)}")
            for error in errors[:5]:  # Tampilkan max 5 error
                self.logger.warning(f"    - {error}")
        
        self.progress_tracker.update_step_progress(100, "Finalisasi selesai")
    
    def stop_cleanup(self) -> bool:
        """
        Hentikan cleanup yang sedang berjalan.
        
        Returns:
            bool: True jika berhasil dihentikan
        """
        if not self.is_running:
            return False
        
        try:
            # Set stop flags
            self.ui_components['stop_requested'] = True
            
            # Cancel future jika ada
            if self.current_future and not self.current_future.done():
                self.current_future.cancel()
            
            # Reset progress tracker
            self.progress_tracker.reset()
            
            self.logger.warning("⏹️ Cleanup dihentikan")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stop cleanup: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Dapatkan status cleanup service.
        
        Returns:
            Dict: Status service
        """
        return {
            'is_running': self.is_running,
            'stop_requested': self.ui_components.get('stop_requested', False),
            'current_step': self.progress_tracker.current_step,
            'total_steps': self.progress_tracker.total_steps,
            'overall_progress': self.progress_tracker.overall_progress,
            'step_progress': self.progress_tracker.step_progress
        }

def create_cleanup_service(ui_components: Dict[str, Any]) -> CleanupService:
    """
    Factory function untuk membuat cleanup service.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        CleanupService: Instance cleanup service
    """
    service = CleanupService(ui_components)
    ui_components['cleanup_service'] = service
    return service