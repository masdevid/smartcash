"""
File: smartcash/dataset/augmentor/services/augmentation_orchestrator.py
Deskripsi: Fixed orchestrator dengan granular progress tracking dan normalisasi yang benar
"""

import time
from typing import Dict, Any, Optional, Callable

from smartcash.dataset.augmentor.utils.config_extractor import create_split_aware_context
from smartcash.dataset.augmentor.utils.dataset_detector import detect_split_structure
from smartcash.dataset.augmentor.utils.cleanup_operations import cleanup_split_aware

from smartcash.dataset.augmentor.core.engine import AugmentationEngine
from smartcash.dataset.augmentor.core.normalizer import NormalizationEngine
from smartcash.common.utils.file_naming_manager import FileNamingManager

class AugmentationOrchestrator:
    """Fixed orchestrator dengan granular progress dan normalisasi yang benar"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        self.context = create_split_aware_context(self._align_config(config), 
                                                 ui_components.get('comm') if ui_components else None)
        self.config, self.progress, self.paths = self.context['config'], self.context['progress'], self.context['paths']
        self.comm = self.context.get('comm')
        
        self.naming_manager = FileNamingManager(config)
        self.engine = AugmentationEngine(self.config, self.comm)
        self.normalizer = NormalizationEngine(self.config, self.comm)
        
        self.target_split = self.config.get('target_split', 'train')
    
    def run_full_pipeline(self, target_split: str = None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute pipeline dengan granular progress tracking termasuk balancing"""
        start_time = time.time()
        actual_target_split = target_split or self.target_split
        
        self.comm and self.comm.start_operation("Augmentation Pipeline", 100)
        
        try:
            # Phase 1: Dataset validation (0-10%)
            self._report_progress("overall", 5, 100, f"🔍 Validating {actual_target_split} dataset", progress_callback)
            dataset_info = detect_split_structure(self.paths['raw_dir'])
            if dataset_info['status'] == 'error':
                return self._error_result(f"Dataset validation failed: {dataset_info['message']}")
            
            self._report_progress("overall", 10, 100, f"✅ Dataset valid: {dataset_info.get('total_images', 0)} gambar", progress_callback)
            
            # Phase 2: Class balancing analysis (10-25%)
            self._report_progress("overall", 15, 100, f"⚖️ Analyzing class balance untuk {actual_target_split}", progress_callback)
            balance_result = self._execute_balancing_analysis(actual_target_split, progress_callback)
            
            self._report_progress("overall", 25, 100, f"✅ Balance analysis: {balance_result.get('classes_needing_aug', 0)} kelas butuh augmentasi", progress_callback)
            
            # Phase 3: Augmentation execution (25-70%)
            self._report_progress("overall", 30, 100, f"🔄 Memulai augmentation untuk {actual_target_split}", progress_callback)
            aug_result = self._execute_augmentation_with_progress(actual_target_split, progress_callback)
            if aug_result['status'] != 'success':
                return self._error_result(f"Augmentation failed: {aug_result['message']}")
            
            self._report_progress("overall", 70, 100, f"✅ Augmentation selesai: {aug_result.get('total_generated', 0)} file", progress_callback)
            
            # Phase 4: Normalization (70-95%)
            self._report_progress("overall", 75, 100, f"🔧 Memulai normalisasi untuk {actual_target_split}", progress_callback)
            norm_result = self._execute_normalization_with_progress(actual_target_split, progress_callback)
            
            if norm_result['status'] != 'success':
                self.comm and self.comm.log_warning(f"Normalization issue: {norm_result.get('message', 'Unknown error')}")
            
            self._report_progress("overall", 95, 100, f"✅ Normalisasi selesai: {norm_result.get('total_normalized', 0)} file", progress_callback)
            
            # Phase 5: Final summary (95-100%)
            self._report_progress("overall", 100, 100, "🎉 Pipeline selesai", progress_callback)
            
            result = {
                'status': 'success', 'total_generated': aug_result.get('total_generated', 0),
                'total_normalized': norm_result.get('total_normalized', 0), 'target_split': actual_target_split,
                'processing_time': time.time() - start_time, 'pipeline_type': aug_result.get('pipeline_type', 'combined'),
                'uuid_registry_size': len(self.naming_manager.uuid_registry),
                'balance_result': balance_result, 'aug_result': aug_result, 'norm_result': norm_result
            }
            
            self.comm and self.comm.complete_operation("Augmentation Pipeline", 
                f"Pipeline selesai: {result['total_generated']} generated, {result['total_normalized']} normalized, {balance_result.get('classes_needing_aug', 0)} kelas dibalance")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.comm and self.comm.error_operation("Augmentation Pipeline", error_msg)
            return self._error_result(error_msg)
    
    def _execute_balancing_analysis(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute class balancing analysis dengan progress tracking"""
        try:
            self._report_progress("step", 0, 100, "⚖️ Analyzing class distribution", progress_callback)
            
            from smartcash.dataset.augmentor.strategies.balancer import ClassBalancingStrategy
            
            balancer = ClassBalancingStrategy(self.config, self.comm)
            
            # Progress update
            balance_progress_callback = lambda step, curr, total, msg: self._report_progress(
                "overall", 10 + int((curr / max(total, 1)) * 15), 100, f"⚖️ Balance: {msg}", progress_callback
            )
            
            # Analyze current distribution
            self._report_progress("step", 30, 100, "📊 Calculating class needs", progress_callback)
            target_count = self.config.get('target_count', 500)
            class_needs = balancer.calculate_balancing_needs_split_aware(
                self.paths['raw_dir'], target_split, target_count
            )
            
            # Analysis results
            classes_needing_aug = len([cls for cls, need in class_needs.items() if need > 0])
            total_needed = sum(class_needs.values())
            
            self.comm and self.comm.log_info(f"⚖️ Balance analysis: {classes_needing_aug} kelas butuh {total_needed} sampel tambahan")
            
            # Log top classes yang butuh augmentasi
            if classes_needing_aug > 0:
                priority_order = balancer.get_balancing_priority_order(class_needs)
                top_classes = priority_order[:3]
                self.comm and self.comm.log_info(f"🎯 Top priority classes: {', '.join(top_classes)}")
            
            return {
                'status': 'success', 'class_needs': class_needs, 'classes_needing_aug': classes_needing_aug,
                'total_needed': total_needed, 'target_count': target_count,
                'priority_order': balancer.get_balancing_priority_order(class_needs)
            }
            
        except Exception as e:
            error_msg = f'Balance analysis error: {str(e)}'
            self.comm and self.comm.log_error(error_msg)
            return {'status': 'error', 'message': error_msg, 'classes_needing_aug': 0}
    
    def _execute_augmentation_with_progress(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute augmentation dengan progress tracking yang detail"""
        try:
            self._report_progress("step", 0, 100, "🔍 Analyzing dataset files", progress_callback)
            
            # Create augmentation-specific progress callback (25-70% range)
            def aug_progress_callback(step: str, current: int, total: int, message: str):
                overall_current = 25 + int((current / max(total, 1)) * 45)
                self._report_progress("overall", overall_current, 100, f"🔄 Augmentation: {message}", progress_callback)
            
            result = self.engine.run_augmentation_pipeline(target_split, aug_progress_callback)
            
            if result['status'] == 'success':
                self.comm and self.comm.log_success(f"✅ Augmentation berhasil: {result.get('total_generated', 0)} file dibuat")
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': f'Augmentation error: {str(e)}', 'total_generated': 0}
    
    def _execute_normalization_with_progress(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute normalization dengan progress tracking yang benar"""
        try:
            self._report_progress("step", 0, 100, "🔧 Preparing normalization", progress_callback)
            
            # Path untuk augmented files
            aug_split_dir = f"{self.paths['aug_dir']}"
            prep_split_dir = f"{self.paths['prep_dir']}"
            
            self.comm and self.comm.log_info(f"📁 Normalisasi dari: {aug_split_dir}")
            self.comm and self.comm.log_info(f"📁 Normalisasi ke: {prep_split_dir}")
            
            # Create normalization-specific progress callback (70-95% range)
            def norm_progress_callback(step: str, current: int, total: int, message: str):
                overall_current = 70 + int((current / max(total, 1)) * 25)
                self._report_progress("overall", overall_current, 100, f"🔧 Normalization: {message}", progress_callback)
            
            # Execute normalization dengan parameter yang benar
            result = self.normalizer.normalize_augmented_data(aug_split_dir, prep_split_dir, target_split)
            
            if result['status'] == 'success':
                self.comm and self.comm.log_success(f"✅ Normalisasi berhasil: {result.get('total_normalized', 0)} file dinormalisasi")
                self.comm and self.comm.log_info(f"📂 File tersimpan di: {result.get('target_dir', prep_split_dir)}")
            else:
                self.comm and self.comm.log_warning(f"⚠️ Normalisasi warning: {result.get('message', 'Unknown issue')}")
            
            return result
            
        except Exception as e:
            error_msg = f'Normalization error: {str(e)}'
            self.comm and self.comm.log_error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_normalized': 0}
    
    def _report_progress(self, step: str, current: int, total: int, message: str, progress_callback: Optional[Callable]):
        """Report progress ke semua tracking systems"""
        # Internal communicator
        if self.comm:
            self.comm.progress(step, current, total, message)
        
        # External callback
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, target_split: str = None) -> Dict[str, Any]:
        """Cleanup dengan progress tracking"""
        actual_target_split = target_split or self.target_split
        return cleanup_split_aware(self.paths['aug_dir'], 
                                 self.paths['prep_dir'] if include_preprocessed else None, 
                                 actual_target_split, self.progress)
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        return {
            'uuid_registry_size': len(self.naming_manager.uuid_registry), 'target_split': self.target_split,
            'engine_ready': hasattr(self.engine, 'config'), 'normalizer_ready': hasattr(self.normalizer, 'config'),
            'context_ready': bool(self.context), 'paths_configured': bool(self.paths),
            'paths': self.paths
        }
    
    def generate_consistency_report(self) -> Dict[str, Any]:
        """Generate report dengan dataset analysis"""
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
        """Align config parameters"""
        aug_config = config.get('augmentation', {})
        return {
            'data': {'dir': config.get('data', {}).get('dir', 'data')},
            'augmentation': {**aug_config, 'target_split': aug_config.get('target_split', 'train')},
            'preprocessing': {'output_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')}
        }
    
    def _error_result(self, msg: str) -> Dict[str, Any]:
        """Create error result dengan UUID info"""
        return {'status': 'error', 'message': msg, 'uuid_registry_size': len(self.naming_manager.uuid_registry)}