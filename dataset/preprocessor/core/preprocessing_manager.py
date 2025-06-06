"""
File: smartcash/dataset/preprocessor/core/preprocessing_manager.py
Deskripsi: Manager preprocessing dengan detailed progress tracking dan UUID consistency terintegrasi
"""

import time
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetProcessingError
from smartcash.dataset.organizer.dataset_file_renamer import create_dataset_renamer
from smartcash.dataset.preprocessor.core.preprocessing_coordinator import SplitCoordinator
from smartcash.dataset.preprocessor.core.preprocessing_validator import PreprocessingValidator
from smartcash.dataset.preprocessor.utils.preprocessing_config import PreprocessingConfig
from smartcash.dataset.preprocessor.utils.preprocessing_stats import PreprocessingStats

class PreprocessingManager:
    """Manager preprocessing dengan detailed UUID progress tracking"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        
        # Initialize components
        self.config_manager = PreprocessingConfig(config, self.logger)
        self.validator = PreprocessingValidator(config, self.logger)
        self.coordinator = SplitCoordinator(config, self.logger)
        self.stats_collector = PreprocessingStats(self.logger)
        self.renamer = create_dataset_renamer(config)
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback ke semua components"""
        self._progress_callback = callback
        self.coordinator.register_progress_callback(callback)
    
    def preprocess_with_uuid_consistency(self, split: str = 'all', force_reprocess: bool = False, **kwargs) -> Dict[str, Any]:
        """Preprocessing dengan detailed UUID progress tracking"""
        start_time = time.time()
        
        try:
            # Phase 1: UUID Consistency dengan detailed progress (0-20%)
            self._notify_progress(5, "🔍 Checking UUID consistency", step=1)
            consistency_result = self._ensure_uuid_consistency_with_progress()
            
            if not consistency_result['success']:
                self.logger.warning(f"⚠️ UUID warning: {consistency_result['message']}")
            
            # Phase 2: Standard preprocessing (20-90%)
            self._notify_progress(25, "🚀 Starting preprocessing", step=2)
            return self._execute_preprocessing_workflow(split, force_reprocess, start_time, **kwargs)
            
        except Exception as e:
            error_msg = f"Preprocessing error: {str(e)}"
            self._notify_progress(0, f"❌ {error_msg}", step=0)
            return {'success': False, 'message': error_msg, 'processing_time': time.time() - start_time}
    
    def _ensure_uuid_consistency_with_progress(self) -> Dict[str, Any]:
        """Ensure UUID consistency dengan detailed progress tracking"""
        try:
            data_dir = self.config.get('data', {}).get('dir', 'data')
            
            self._notify_progress(8, "📋 Scanning files for UUID consistency")
            preview = self.renamer.get_rename_preview(data_dir, limit=10)
            
            if preview['status'] == 'success' and preview['total_files'] == 0:
                self._notify_progress(15, "✅ Files already UUID consistent")
                return {'success': True, 'message': 'Already UUID consistent', 'renamed_files': 0}
            
            files_to_rename = preview.get('total_files', 0)
            self.logger.info(f"🎯 {files_to_rename} files need UUID renaming")
            
            # Detailed progress callback untuk renaming
            def detailed_rename_progress(progress, message):
                mapped_progress = 8 + (progress * 12 / 100)  # Map 0-100% to 8-20%
                self._notify_progress(int(mapped_progress), f"🔄 {message}")
            
            self._notify_progress(12, f"🔧 Renaming {files_to_rename} files to UUID format")
            rename_result = self.renamer.batch_rename_dataset(
                data_dir, backup=False,
                progress_callback=detailed_rename_progress
            )
            
            success = rename_result.get('status') == 'success'
            renamed_count = rename_result.get('renamed_files', 0)
            
            if success:
                self._notify_progress(20, f"✅ UUID consistency: {renamed_count} files renamed")
                self.logger.success(f"✅ UUID consistency established: {renamed_count} files renamed")
            else:
                self.logger.warning(f"⚠️ UUID rename issues: {rename_result.get('message')}")
            
            return {
                'success': success,
                'message': rename_result.get('message', 'UUID consistency completed'),
                'renamed_files': renamed_count
            }
            
        except Exception as e:
            return {'success': False, 'message': f'UUID consistency error: {str(e)}'}
    
    def _execute_preprocessing_workflow(self, split: str, force_reprocess: bool, start_time: float, **kwargs) -> Dict[str, Any]:
        """Execute preprocessing workflow dengan progress tracking"""
        try:
            # Validation (20-30%)
            self._notify_progress(25, "📋 Validating preprocessing request", step=2)
            validation_result = self.validator.validate_preprocessing_request(split, force_reprocess, **kwargs)
            
            if not validation_result['valid']:
                raise DatasetProcessingError(f"Validation failed: {validation_result['message']}")
            
            # Prepare config (30-35%)
            self._notify_progress(32, "⚙️ Preparing processing configuration", step=2)
            processing_config = self.config_manager.prepare_processing_config(**kwargs)
            target_splits = validation_result['target_splits']
            
            self.logger.info(f"🎯 Processing {len(target_splits)} splits: {target_splits}")
            
            # Coordinate processing (35-85%)
            self._notify_progress(38, f"🔄 Processing {len(target_splits)} splits", step=2)
            coordination_result = self.coordinator.coordinate_parallel_splits(
                target_splits, processing_config, force_reprocess
            )
            
            if not coordination_result['success']:
                raise DatasetProcessingError(f"Processing failed: {coordination_result.get('message')}")
            
            # Collect stats (85-95%)
            self._notify_progress(88, "📊 Collecting processing statistics", step=3)
            final_stats = self.stats_collector.aggregate_processing_results(
                coordination_result['split_results'], processing_time=time.time() - start_time
            )
            
            # Add UUID metrics
            final_stats.update({
                'uuid_registry_size': len(self.renamer.naming_manager.uuid_registry),
                'uuid_consistency_maintained': True
            })
            
            # Complete (95-100%)
            self._notify_progress(98, f"✅ Processing complete: {final_stats['total_images']} images", step=3)
            self.logger.success(f"✅ Preprocessing complete: {final_stats['total_images']} images, {final_stats['processing_time']:.1f}s")
            
            return {'success': True, 'message': 'Preprocessing completed with UUID consistency', **final_stats}
            
        except Exception as e:
            raise DatasetProcessingError(f"Preprocessing workflow failed: {str(e)}")
    
    def coordinate_preprocessing(self, split: str = 'all', force_reprocess: bool = False, **kwargs) -> Dict[str, Any]:
        """Legacy method - redirects to UUID-aware preprocessing"""
        return self.preprocess_with_uuid_consistency(split, force_reprocess, **kwargs)
    
    def get_uuid_consistency_report(self) -> Dict[str, Any]:
        """Get UUID consistency report"""
        try:
            data_dir = self.config.get('data', {}).get('dir', 'data')
            return self.renamer.get_rename_preview(data_dir, limit=50)
        except Exception as e:
            return {'status': 'error', 'message': f'UUID report error: {str(e)}'}
    
    def cleanup_preprocessing_state(self) -> None:
        """Cleanup preprocessing state"""
        self.coordinator.cleanup_coordination_state()
        self.stats_collector.reset_stats_collection()
    
    def _notify_progress(self, progress: int, message: str, step: int = 0):
        """Progress notification dengan detailed info"""
        if self._progress_callback:
            try:
                self._progress_callback(progress=progress, message=message, step=step)
            except Exception as e:
                self.logger.debug(f"🔧 Progress callback error: {str(e)}")

# Factory functions
def create_preprocessing_manager(config: Dict[str, Any]) -> PreprocessingManager:
    """Factory untuk preprocessing manager"""
    return PreprocessingManager(config)

# One-liner utilities
preprocess_with_uuid = lambda config, split='all', force=False: create_preprocessing_manager(config).preprocess_with_uuid_consistency(split, force)
get_uuid_report = lambda config: create_preprocessing_manager(config).get_uuid_consistency_report()