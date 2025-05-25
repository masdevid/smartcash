"""
File: smartcash/dataset/preprocessor/core/preprocessing_manager.py
Deskripsi: Main orchestrator untuk koordinasi semua operasi preprocessing dengan progress callback
"""

import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetProcessingError
from smartcash.dataset.preprocessor.core.preprocessing_coordinator import SplitCoordinator
from smartcash.dataset.preprocessor.core.preprocessing_validator import PreprocessingValidator
from smartcash.dataset.preprocessor.operations.progress_tracker import ProgressTracker
from smartcash.dataset.preprocessor.utils.preprocessing_config import PreprocessingConfig
from smartcash.dataset.preprocessor.utils.preprocessing_stats import PreprocessingStats


class PreprocessingManager:
    """Main orchestrator untuk semua operasi preprocessing dengan unified service architecture."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize preprocessing manager dengan service dependencies."""
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        
        # Initialize service components dengan dependency injection
        self.config_manager = PreprocessingConfig(config, self.logger)
        self.validator = PreprocessingValidator(config, self.logger)
        self.coordinator = SplitCoordinator(config, self.logger)
        self.progress_tracker = ProgressTracker(self.logger)
        self.stats_collector = PreprocessingStats(self.logger)
        
        self.logger.debug("🎯 PreprocessingManager initialized dengan unified architecture")
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk UI notifications."""
        self._progress_callback = callback
        self.progress_tracker.register_callback(callback)
        self.coordinator.register_progress_callback(callback)
        self.logger.debug("📡 Progress callback registered ke semua service components")
    
    def coordinate_preprocessing(self, split: str = 'all', force_reprocess: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Koordinasi preprocessing dataset dengan unified service approach.
        
        Args:
            split: Target split ('train', 'valid', 'test', 'all')
            force_reprocess: Paksa pemrosesan ulang
            **kwargs: Parameter tambahan preprocessing
            
        Returns:
            Dictionary hasil preprocessing dengan statistik lengkap
        """
        start_time = time.time()
        
        try:
            # Phase 1: Validation & Setup (0-10%)
            self._notify_progress(5, "Memulai validasi preprocessing", step=1)
            
            validation_result = self.validator.validate_preprocessing_request(split, force_reprocess, **kwargs)
            if not validation_result['valid']:
                raise DatasetProcessingError(f"Validasi gagal: {validation_result['message']}")
            
            processing_config = self.config_manager.prepare_processing_config(**kwargs)
            target_splits = validation_result['target_splits']
            
            self._notify_progress(10, f"Validasi selesai: {len(target_splits)} split target", step=1)
            
            # Phase 2: Coordinate Processing (10-90%)
            self._notify_progress(15, "Memulai koordinasi preprocessing", step=2)
            
            coordination_result = self.coordinator.coordinate_parallel_splits(
                target_splits, processing_config, force_reprocess
            )
            
            if not coordination_result['success']:
                raise DatasetProcessingError(f"Koordinasi gagal: {coordination_result.get('message')}")
            
            # Phase 3: Finalization & Stats (90-100%)
            self._notify_progress(95, "Mengumpulkan statistik hasil", step=3)
            
            final_stats = self.stats_collector.aggregate_processing_results(
                coordination_result['split_results'], 
                processing_time=time.time() - start_time
            )
            
            self._notify_progress(100, f"Preprocessing selesai: {final_stats['total_images']} gambar", step=3)
            
            self.logger.success(
                f"✅ Preprocessing berhasil: {final_stats['total_images']} gambar, "
                f"{final_stats['processing_time']:.1f} detik"
            )
            
            return {
                'success': True,
                'message': 'Preprocessing completed successfully',
                **final_stats,
                'config_used': processing_config
            }
            
        except Exception as e:
            error_msg = f"Error koordinasi preprocessing: {str(e)}"
            self._notify_progress(0, error_msg, status='error')
            self.logger.error(f"❌ {error_msg}")
            
            return {
                'success': False,
                'message': error_msg,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Dapatkan summary status preprocessing dari semua service components."""
        return {
            'validator_status': self.validator.get_validation_summary(),
            'coordinator_status': self.coordinator.get_coordination_summary(),
            'progress_status': self.progress_tracker.get_progress_summary(),
            'stats_status': self.stats_collector.get_stats_summary(),
            'manager_ready': True
        }
    
    def cleanup_preprocessing_state(self) -> None:
        """Cleanup state dari semua service components."""
        self.coordinator.cleanup_coordination_state()
        self.progress_tracker.reset_progress_state()
        self.stats_collector.reset_stats_collection()
        self.logger.debug("🧹 Preprocessing manager state cleaned up")
    
    def _notify_progress(self, progress: int, message: str, step: int = 0, status: str = 'info'):
        """Internal progress notification ke registered callbacks."""
        if self._progress_callback:
            try:
                self._progress_callback(
                    progress=progress, total=100, message=message, 
                    status=status, step=step
                )
            except Exception as e:
                self.logger.debug(f"🔧 Progress callback error: {str(e)}")