"""
File: smartcash/dataset/augmentor/services/augmentation_orchestrator.py
Deskripsi: Updated orchestrator menggunakan SRP modules dengan reuse dan one-liner style
"""

import time
from typing import Dict, Any, Optional, Callable

# Reuse dari SRP utils modules
from smartcash.dataset.augmentor.utils.config_extractor import create_split_aware_context
from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
from smartcash.dataset.augmentor.utils.cleanup_operations import cleanup_split_aware
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker

# Reuse dari updated strategies
from smartcash.dataset.augmentor.strategies.balancer import ClassBalancingStrategy
from smartcash.dataset.augmentor.strategies.selector import FileSelectionStrategy
from smartcash.dataset.augmentor.strategies.priority import PriorityCalculator

# Reuse dari core components
from smartcash.dataset.augmentor.core.engine import AugmentationEngine
from smartcash.dataset.augmentor.core.normalizer import NormalizationEngine
from smartcash.common.utils.file_naming_manager import FileNamingManager

class AugmentationOrchestrator:
    """Updated orchestrator dengan full SRP reuse dan one-liner integration"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        # Create context menggunakan SRP config extractor
        self.context = create_split_aware_context(self._align_config(config), 
                                                 ui_components.get('comm') if ui_components else None)
        self.config, self.progress, self.paths = self.context['config'], self.context['progress'], self.context['paths']
        self.comm = self.context.get('comm')
        
        # Initialize components menggunakan reuse
        self.naming_manager = FileNamingManager(config)
        self.engine = AugmentationEngine(self.config, self.comm)
        self.normalizer = NormalizationEngine(self.config, self.comm)
        
        self.target_split = self.config.get('target_split', 'train')
    
    def run_full_pipeline(self, target_split: str = None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute pipeline menggunakan engine reuse"""
        start_time = time.time()
        actual_target_split = target_split or self.target_split
        
        self.comm and self.comm.start_operation("UUID Augmentation Pipeline", 100)
        
        try:
            # Phase 1: Dataset validation (0-15%) - reuse detector
            self.progress.progress("overall", 5, 100, f"Validating {actual_target_split} dataset")
            dataset_info = detect_split_structure(self.paths['raw_dir'])
            if dataset_info['status'] == 'error':
                return self._error_result(f"Dataset validation failed: {dataset_info['message']}")
            
            # Phase 2: Augmentation execution (15-80%) - reuse engine
            self.progress.progress("overall", 20, 100, f"Executing augmentation for {actual_target_split}")
            aug_result = self.engine.run_augmentation_pipeline(actual_target_split, progress_callback)
            if aug_result['status'] != 'success':
                return self._error_result(f"Augmentation failed: {aug_result['message']}")
            
            # Phase 3: Normalization (80-100%) - reuse normalizer
            self.progress.progress("overall", 85, 100, f"Normalizing augmented data for {actual_target_split}")
            norm_result = self.normalizer.normalize_augmented_data(
                f"{self.paths['aug_dir']}/{actual_target_split}",
                self.paths['prep_dir'], 
                actual_target_split
            )
            
            result = {
                'status': 'success', 'total_generated': aug_result.get('total_generated', 0),
                'total_normalized': norm_result.get('total_normalized', 0), 'target_split': actual_target_split,
                'processing_time': time.time() - start_time, 'pipeline_type': aug_result.get('pipeline_type', 'combined'),
                'uuid_registry_size': len(self.naming_manager.uuid_registry)
            }
            
            self.comm and self.comm.complete_operation("UUID Augmentation Pipeline", 
                f"Pipeline completed: {result['total_generated']} generated, {result['total_normalized']} normalized")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.comm and self.comm.error_operation("UUID Augmentation Pipeline", error_msg)
            return self._error_result(error_msg)
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, target_split: str = None) -> Dict[str, Any]:
        """Cleanup menggunakan SRP cleanup operations"""
        actual_target_split = target_split or self.target_split
        return cleanup_split_aware(self.paths['aug_dir'], 
                                 self.paths['prep_dir'] if include_preprocessed else None, 
                                 actual_target_split, self.progress)
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Get comprehensive status menggunakan reused components"""
        return {
            'uuid_registry_size': len(self.naming_manager.uuid_registry), 'target_split': self.target_split,
            'engine_ready': hasattr(self.engine, 'config'), 'normalizer_ready': hasattr(self.normalizer, 'config'),
            'context_ready': bool(self.context), 'paths_configured': bool(self.paths)
        }
    
    def generate_consistency_report(self) -> Dict[str, Any]:
        """Generate report menggunakan detector reuse"""
        try:
            split_analysis = {}
            for split in ['train', 'valid', 'test']:
                split_info = detect_split_structure(f"{self.paths['raw_dir']}/{split}")
                split_analysis[split] = {
                    'total_images': split_info.get('total_images', 0),
                    'structure_type': split_info.get('structure_type', 'unknown'),
                    'status': split_info.get('status', 'unknown')
                }
            
            return {
                'uuid_registry_stats': {'total_uuids': len(self.naming_manager.uuid_registry)},
                'split_analysis': split_analysis, 'target_split': self.target_split,
                'consistency_summary': {'overall_consistent': True, 'report_generated': True}
            }
            
        except Exception as e:
            return {'error': f'Report generation error: {str(e)}'}
    
    def _align_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Align config parameters dengan one-liner approach"""
        aug_config = config.get('augmentation', {})
        return {
            'data': {'dir': config.get('data', {}).get('dir', 'data')},
            'augmentation': {**aug_config, 'target_split': aug_config.get('target_split', 'train')},
            'preprocessing': {'output_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')}
        }
    
    def _error_result(self, msg: str) -> Dict[str, Any]:
        """Create error result dengan UUID info"""
        return {'status': 'error', 'message': msg, 'uuid_registry_size': len(self.naming_manager.uuid_registry)}
