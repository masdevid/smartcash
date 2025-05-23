"""
File: smartcash/ui/dataset/preprocessing/services/service_runner.py
Deskripsi: Service runner untuk menjalankan preprocessing dengan integrasi backend
"""

from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.ui.dataset.preprocessing.utils.service_integration import (
    create_service_integrator, validate_preprocessing_requirements
)
from smartcash.ui.dataset.preprocessing.utils.progress_tracker import (
    create_progress_tracker, create_progress_callback, ProcessingSteps
)
from smartcash.ui.dataset.preprocessing.utils.drive_utils import (
    setup_drive_preprocessing_storage, sync_preprocessing_to_drive
)

logger = get_logger(__name__)

class ServiceRunner:
    """Runner untuk menjalankan preprocessing service dengan UI integration."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi service runner.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger', logger)
        
        # Setup integrator dan tracker
        self.service_integrator = create_service_integrator(ui_components)
        self.progress_tracker = create_progress_tracker(ui_components)
        
        # Setup progress callback
        progress_callback = create_progress_callback(self.progress_tracker)
        self.service_integrator.register_progress_callback(progress_callback)
        
        # State tracking
        self.is_running = False
        self.current_future: Optional[Future] = None
        
    def validate_requirements(self) -> Tuple[bool, str]:
        """
        Validasi requirements sebelum menjalankan preprocessing.
        
        Returns:
            Tuple[bool, str]: (valid, message)
        """
        try:
            # Validasi menggunakan utility function
            is_valid, errors = validate_preprocessing_requirements(self.ui_components)
            
            if not is_valid:
                error_message = "Validasi preprocessing gagal:\n" + "\n".join(f"• {error}" for error in errors)
                return False, error_message
            
            return True, "Validasi preprocessing berhasil"
            
        except Exception as e:
            self.logger.error(f"❌ Error validasi requirements: {str(e)}")
            return False, f"Error validasi: {str(e)}"
    
    def setup_storage(self) -> Tuple[bool, str]:
        """
        Setup penyimpanan untuk preprocessing (Drive atau lokal).
        
        Returns:
            Tuple[bool, str]: (berhasil, message)
        """
        try:
            # Setup Drive storage dengan fallback ke lokal
            success, message = setup_drive_preprocessing_storage(self.ui_components)
            
            if success:
                self.logger.info(f"✅ {message}")
            else:
                self.logger.warning(f"⚠️ {message}")
            
            return success, message
            
        except Exception as e:
            self.logger.error(f"❌ Error setup storage: {str(e)}")
            return False, f"Error setup storage: {str(e)}"
    
    def run_preprocessing(self, config: Dict[str, Any]) -> Future:
        """
        Jalankan preprocessing secara asynchronous.
        
        Args:
            config: Konfigurasi preprocessing
            
        Returns:
            Future: Future object untuk tracking
        """
        if self.is_running:
            raise RuntimeError("Preprocessing sudah berjalan")
        
        # Validasi requirements
        valid, message = self.validate_requirements()
        if not valid:
            raise ValueError(message)
        
        # Setup service integrator
        if not self.service_integrator.setup_service(config):
            raise RuntimeError("Gagal setup preprocessing service")
        
        # Set running state
        self.is_running = True
        self.ui_components['preprocessing_running'] = True
        self.ui_components['stop_requested'] = False
        
        # Jalankan dengan ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._run_preprocessing_task, config)
        self.current_future = future
        
        return future
    
    def _run_preprocessing_task(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Task untuk menjalankan preprocessing (dijalankan di thread terpisah).
        
        Args:
            config: Konfigurasi preprocessing
            
        Returns:
            Dict: Hasil preprocessing
        """
        try:
            # Start progress tracking
            split = config.get('split', 'all')
            steps = self._determine_processing_steps(split)
            self.progress_tracker.start_processing(steps)
            
            # Step 1: Preparation
            self.progress_tracker.start_step(ProcessingSteps.PREPARATION)
            self._preparation_step(config)
            self.progress_tracker.complete_step(ProcessingSteps.PREPARATION)
            
            # Step 2-4: Process splits
            result = self._process_splits(config, steps)
            
            # Step 5: Finalization
            self.progress_tracker.start_step(ProcessingSteps.FINALIZATION)
            self._finalization_step(result)
            self.progress_tracker.complete_step(ProcessingSteps.FINALIZATION)
            
            # Complete processing
            self.progress_tracker.complete_processing("Preprocessing berhasil diselesaikan")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error preprocessing task: {str(e)}")
            current_step = list(ProcessingSteps)[min(self.progress_tracker.current_step, len(ProcessingSteps)-1)]
            self.progress_tracker.handle_error(current_step, str(e))
            raise
        finally:
            self.is_running = False
            self.ui_components['preprocessing_running'] = False
            self.current_future = None
    
    def _determine_processing_steps(self, split: str) -> List[ProcessingSteps]:
        """
        Tentukan step yang akan dijalankan berdasarkan split.
        
        Args:
            split: Split yang akan diproses
            
        Returns:
            List[ProcessingSteps]: List step yang akan dijalankan
        """
        steps = [ProcessingSteps.PREPARATION]
        
        if split == 'all':
            steps.extend([ProcessingSteps.TRAIN_SPLIT, ProcessingSteps.VAL_SPLIT, ProcessingSteps.TEST_SPLIT])
        elif split == 'train':
            steps.append(ProcessingSteps.TRAIN_SPLIT)
        elif split == 'val':
            steps.append(ProcessingSteps.VAL_SPLIT)
        elif split == 'test':
            steps.append(ProcessingSteps.TEST_SPLIT)
        
        steps.append(ProcessingSteps.FINALIZATION)
        return steps
    
    def _preparation_step(self, config: Dict[str, Any]) -> None:
        """
        Step persiapan preprocessing.
        
        Args:
            config: Konfigurasi preprocessing
        """
        self.logger.info("🔧 Mempersiapkan preprocessing...")
        
        # Validasi data directory
        data_dir = Path(config.get('data_dir', 'data'))
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory tidak ditemukan: {data_dir}")
        
        # Setup output directory
        output_dir = Path(config.get('preprocessed_dir', 'data/preprocessed'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validasi konfigurasi
        required_keys = ['resolution', 'normalization']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Konfigurasi {key} diperlukan")
        
        self.logger.info("✅ Persiapan selesai")
    
    def _process_splits(self, config: Dict[str, Any], steps: List[ProcessingSteps]) -> Dict[str, Any]:
        """
        Proses splits sesuai konfigurasi.
        
        Args:
            config: Konfigurasi preprocessing
            steps: List steps yang akan dijalankan
            
        Returns:
            Dict: Hasil preprocessing
        """
        split = config.get('split', 'all')
        
        # Mapping step ke split
        split_mapping = {
            ProcessingSteps.TRAIN_SPLIT: 'train',
            ProcessingSteps.VAL_SPLIT: 'val', 
            ProcessingSteps.TEST_SPLIT: 'test'
        }
        
        results = {
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'splits': {}
        }
        
        # Process each split step
        for step in steps:
            if step in split_mapping:
                split_name = split_mapping[step]
                
                # Skip jika tidak sesuai dengan konfigurasi split
                if split != 'all' and split != split_name:
                    continue
                
                # Start step
                self.progress_tracker.start_step(step)
                
                # Process split menggunakan service integrator
                split_result = self.service_integrator.preprocess_dataset(split_name)
                
                if split_result:
                    results['processed'] += split_result.get('processed', 0)
                    results['skipped'] += split_result.get('skipped', 0)
                    results['failed'] += split_result.get('failed', 0)
                    results['splits'][split_name] = split_result
                
                # Complete step
                self.progress_tracker.complete_step(step, f"Split {split_name} selesai")
                
                # Check stop request
                if self.ui_components.get('stop_requested', False):
                    raise InterruptedError("Preprocessing dihentikan oleh pengguna")
        
        return results
    
    def _finalization_step(self, result: Dict[str, Any]) -> None:
        """
        Step finalisasi preprocessing.
        
        Args:
            result: Hasil preprocessing
        """
        self.logger.info("🔄 Finalisasi preprocessing...")
        
        # Sync ke Drive jika tersedia
        try:
            success, message = sync_preprocessing_to_drive(
                self.ui_components,
                lambda current, total, msg: self.progress_tracker.update_step_progress(current, msg)
            )
            
            if success:
                self.logger.info(f"✅ {message}")
            else:
                self.logger.warning(f"⚠️ {message}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal sync ke Drive: {str(e)}")
        
        # Log hasil
        total_processed = result.get('processed', 0)
        total_skipped = result.get('skipped', 0)
        total_failed = result.get('failed', 0)
        
        self.logger.info(f"📊 Hasil preprocessing:")
        self.logger.info(f"  • Diproses: {total_processed}")
        self.logger.info(f"  • Dilewati: {total_skipped}")
        self.logger.info(f"  • Gagal: {total_failed}")
    
    def stop_processing(self) -> bool:
        """
        Hentikan preprocessing yang sedang berjalan.
        
        Returns:
            bool: True jika berhasil dihentikan
        """
        if not self.is_running:
            return False
        
        try:
            # Set stop flags
            self.ui_components['stop_requested'] = True
            self.service_integrator.stop_processing()
            
            # Cancel future jika ada
            if self.current_future and not self.current_future.done():
                self.current_future.cancel()
            
            # Reset progress tracker
            self.progress_tracker.reset()
            
            self.logger.warning("⏹️ Preprocessing dihentikan")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error stop processing: {str(e)}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Dapatkan status service runner.
        
        Returns:
            Dict: Status runner
        """
        return {
            'is_running': self.is_running,
            'stop_requested': self.ui_components.get('stop_requested', False),
            'current_step': self.progress_tracker.current_step,
            'total_steps': self.progress_tracker.total_steps,
            'overall_progress': self.progress_tracker.overall_progress,
            'step_progress': self.progress_tracker.step_progress,
            'service_status': self.service_integrator.get_processing_status()
        }

def create_service_runner(ui_components: Dict[str, Any]) -> ServiceRunner:
    """
    Factory function untuk membuat service runner.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ServiceRunner: Instance service runner
    """
    runner = ServiceRunner(ui_components)
    ui_components['service_runner'] = runner
    return runner