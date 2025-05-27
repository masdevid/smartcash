"""
File: smartcash/dataset/preprocessor/core/preprocessing_manager.py
Deskripsi: Enhanced preprocessing manager dengan dataset_file_renamer integration untuk eliminasi redundansi
"""

import time
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetProcessingError
from smartcash.dataset.services.dataset_file_renamer import create_dataset_renamer
from smartcash.dataset.preprocessor.core.preprocessing_coordinator import SplitCoordinator
from smartcash.dataset.preprocessor.core.preprocessing_validator import PreprocessingValidator
from smartcash.dataset.preprocessor.operations.progress_tracker import ProgressTracker
from smartcash.dataset.preprocessor.utils.preprocessing_config import PreprocessingConfig
from smartcash.dataset.preprocessor.utils.preprocessing_stats import PreprocessingStats


class PreprocessingManager:
    """Enhanced preprocessing manager dengan dataset_file_renamer integration - eliminasi UUID redundansi."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        
        # Initialize service components
        self.config_manager = PreprocessingConfig(config, self.logger)
        self.validator = PreprocessingValidator(config, self.logger)
        self.coordinator = SplitCoordinator(config, self.logger)
        self.progress_tracker = ProgressTracker(self.logger)
        self.stats_collector = PreprocessingStats(self.logger)
        
        # INTEGRATED: Dataset renamer untuk UUID management
        self.renamer = create_dataset_renamer(config)
        
        self.logger.debug("🎯 PreprocessingManager with integrated dataset renamer")
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback ke semua service components"""
        self._progress_callback = callback
        [service.register_callback(callback) if hasattr(service, 'register_callback') else 
         service.register_progress_callback(callback) for service in 
         [self.progress_tracker, self.coordinator]]
    
    def preprocess_with_uuid_consistency(self, split: str = 'all', force_reprocess: bool = False, 
                                       ensure_uuid_consistency: bool = True, **kwargs) -> Dict[str, Any]:
        """
        ENHANCED: Preprocessing dengan UUID consistency enforcement - eliminasi duplikasi naming logic.
        
        Args:
            split: Target split
            force_reprocess: Paksa reprocess
            ensure_uuid_consistency: Ensure UUID consistency sebelum processing
            **kwargs: Parameter preprocessing
        """
        start_time = time.time()
        
        try:
            # Phase 1: UUID Consistency Check & Rename (0-15%)
            if ensure_uuid_consistency:
                self._notify_progress(5, "Ensuring UUID consistency", step=1)
                consistency_result = self._ensure_dataset_uuid_consistency()
                if not consistency_result['success']:
                    self.logger.warning(f"⚠️ UUID consistency warning: {consistency_result['message']}")
                self._notify_progress(15, f"UUID check: {consistency_result.get('renamed_files', 0)} files renamed", step=1)
            
            # Phase 2: Standard preprocessing dengan UUID consistent files (15-90%)
            return self._coordinate_preprocessing_with_uuid_awareness(split, force_reprocess, start_time, **kwargs)
            
        except Exception as e:
            error_msg = f"Enhanced preprocessing error: {str(e)}"
            self._notify_progress(0, error_msg, status='error')
            return {'success': False, 'message': error_msg, 'processing_time': time.time() - start_time}
    
    def _ensure_dataset_uuid_consistency(self) -> Dict[str, Any]:
        """Ensure UUID consistency menggunakan dataset_file_renamer - eliminasi duplikasi"""
        try:
            data_dir = self.config.get('data', {}).get('dir', 'data')
            
            # Check current consistency status
            preview = self.renamer.get_rename_preview(data_dir, limit=10)
            if preview['status'] == 'success' and preview['total_files'] == 0:
                return {'success': True, 'message': 'Already UUID consistent', 'renamed_files': 0}
            
            # Execute rename untuk UUID consistency
            rename_result = self.renamer.batch_rename_dataset(
                data_dir, backup=False,
                progress_callback=lambda p, m: self._notify_progress(5 + (p // 10), f"UUID rename: {m}")
            )
            
            return {
                'success': rename_result.get('status') == 'success',
                'message': rename_result.get('message', 'UUID consistency completed'),
                'renamed_files': rename_result.get('renamed_files', 0),
                'uuid_registry_size': len(self.renamer.naming_manager.uuid_registry)
            }
            
        except Exception as e:
            return {'success': False, 'message': f'UUID consistency error: {str(e)}'}
    
    def _coordinate_preprocessing_with_uuid_awareness(self, split: str, force_reprocess: bool, 
                                                    start_time: float, **kwargs) -> Dict[str, Any]:
        """Coordinate preprocessing dengan UUID awareness - reuse existing coordinator"""
        try:
            # Validation dengan UUID context
            self._notify_progress(20, "Validating with UUID context", step=1)
            validation_result = self.validator.validate_preprocessing_request(split, force_reprocess, **kwargs)
            
            if not validation_result['valid']:
                raise DatasetProcessingError(f"Validation failed: {validation_result['message']}")
            
            processing_config = self.config_manager.prepare_processing_config(**kwargs)
            target_splits = validation_result['target_splits']
            
            # Coordinate processing dengan UUID consistent files
            self._notify_progress(25, f"Processing {len(target_splits)} splits with UUID consistency", step=2)
            coordination_result = self.coordinator.coordinate_parallel_splits(
                target_splits, processing_config, force_reprocess
            )
            
            if not coordination_result['success']:
                raise DatasetProcessingError(f"Processing failed: {coordination_result.get('message')}")
            
            # Finalize dengan UUID stats
            self._notify_progress(95, "Collecting UUID-aware statistics", step=3)
            final_stats = self.stats_collector.aggregate_processing_results(
                coordination_result['split_results'], processing_time=time.time() - start_time
            )
            
            # Add UUID consistency metrics
            final_stats.update({
                'uuid_registry_size': len(self.renamer.naming_manager.uuid_registry),
                'uuid_consistency_maintained': True,
                'renamer_integrated': True
            })
            
            self._notify_progress(100, f"Processing complete: {final_stats['total_images']} images with UUID consistency", step=3)
            
            self.logger.success(f"✅ Preprocessing with UUID consistency: {final_stats['total_images']} images, {final_stats['processing_time']:.1f}s")
            
            return {'success': True, 'message': 'Preprocessing completed with UUID consistency', **final_stats}
            
        except Exception as e:
            raise DatasetProcessingError(f"UUID-aware preprocessing failed: {str(e)}")
    
    # LEGACY: Backward compatibility method
    def coordinate_preprocessing(self, split: str = 'all', force_reprocess: bool = False, **kwargs) -> Dict[str, Any]:
        """Legacy method - redirects to UUID-aware preprocessing"""
        return self.preprocess_with_uuid_consistency(split, force_reprocess, True, **kwargs)
    
    def get_uuid_consistency_report(self) -> Dict[str, Any]:
        """Get UUID consistency report dari integrated renamer"""
        try:
            data_dir = self.config.get('data', {}).get('dir', 'data')
            return self.renamer.get_rename_preview(data_dir, limit=50)
        except Exception as e:
            return {'status': 'error', 'message': f'UUID report error: {str(e)}'}
    
    def batch_rename_dataset(self, backup: bool = True) -> Dict[str, Any]:
        """Expose dataset renaming functionality"""
        try:
            data_dir = self.config.get('data', {}).get('dir', 'data')
            return self.renamer.batch_rename_dataset(
                data_dir, backup=backup,
                progress_callback=lambda p, m: self._notify_progress(p, f"Renaming: {m}")
            )
        except Exception as e:
            return {'status': 'error', 'message': f'Batch rename error: {str(e)}'}
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Summary dengan UUID integration status"""
        base_summary = {
            'validator_status': self.validator.get_validation_summary(),
            'coordinator_status': self.coordinator.get_coordination_summary(),
            'progress_status': self.progress_tracker.get_progress_summary(),
            'stats_status': self.stats_collector.get_stats_summary(),
            'manager_ready': True
        }
        
        # Add UUID integration info
        base_summary.update({
            'renamer_integrated': self.renamer is not None,
            'uuid_registry_size': len(self.renamer.naming_manager.uuid_registry) if self.renamer else 0,
            'uuid_consistency_available': True
        })
        
        return base_summary
    
    def cleanup_preprocessing_state(self) -> None:
        """Cleanup dengan UUID registry preservation"""
        self.coordinator.cleanup_coordination_state()
        self.progress_tracker.reset_progress_state()
        self.stats_collector.reset_stats_collection()
        # UUID registry intentionally preserved untuk consistency
        self.logger.debug("🧹 Preprocessing state cleaned, UUID registry preserved")
    
    def _notify_progress(self, progress: int, message: str, step: int = 0, status: str = 'info'):
        """Progress notification dengan UUID context"""
        if self._progress_callback:
            try:
                self._progress_callback(progress=progress, total=100, message=message, status=status, step=step)
            except Exception as e:
                self.logger.debug(f"🔧 Progress callback error: {str(e)}")


# REUSE: Factory functions dengan integrated renamer
def create_preprocessing_manager_with_renamer(config: Dict[str, Any]) -> PreprocessingManager:
    """Factory untuk preprocessing manager dengan integrated dataset renamer"""
    return PreprocessingManager(config)

# One-liner utilities
preprocess_with_uuid = lambda config, split='all', force=False: create_preprocessing_manager_with_renamer(config).preprocess_with_uuid_consistency(split, force)
get_uuid_report = lambda config: create_preprocessing_manager_with_renamer(config).get_uuid_consistency_report()
batch_rename_and_preprocess = lambda config, split='all': create_preprocessing_manager_with_renamer(config).preprocess_with_uuid_consistency(split, False, True)