"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Augmentation service dengan FileNamingManager, preprocessor API, dan live preview
"""

import time
from typing import Dict, Any, Optional, Callable, Tuple
from pathlib import Path

from smartcash.dataset.augmentor.core.engine import AugmentationEngine
from smartcash.dataset.augmentor.core.normalizer import NormalizationEngine
from smartcash.dataset.augmentor.utils.progress_bridge import ProgressBridge
from smartcash.dataset.augmentor.utils.path_resolver import PathResolver
from smartcash.dataset.augmentor.utils.sample_generator import AugmentationSampleGenerator
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class AugmentationService:
    """🎯 Service dengan FileNamingManager, preprocessor API, dan live preview"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_tracker=None):
        self.logger = get_logger(__name__)
        if config is None:
            self.config = get_default_augmentation_config()
            self.logger.info("Using default config")
        else:
            self.config = validate_augmentation_config(config)
            self.logger.info("Config validated")  
        
        self.progress = ProgressBridge(progress_tracker) if progress_tracker else None
        self.engine = AugmentationEngine(self.config, self.progress)
        self.normalizer = NormalizationEngine(self.config, self.progress)
        self.path_resolver = PathResolver(self.config)
        self.sample_generator = AugmentationSampleGenerator(self.config)
        
    def run_augmentation_pipeline(self, target_split: str = "train", 
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🚀 Execute pipeline: augmentation + normalization"""
        start_time = time.time()
        
        try:
            self._report_start(target_split)
            
            # Phase 1: Augmentation
            aug_result = self._execute_augmentation(target_split, progress_callback)
            if aug_result['status'] != 'success':
                return aug_result
                
            # Phase 2: Normalization
            norm_result = self._execute_normalization(target_split, progress_callback)
            if norm_result['status'] != 'success':
                return norm_result
            
            return self._create_final_result(aug_result, norm_result, start_time)
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_generated': 0}
    
    def create_live_preview(self, target_split: str = 'train') -> Dict[str, Any]:
        """🎥 Create live preview augmentation tanpa normalization"""
        return self.engine.create_live_preview(target_split)
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """📊 Enhanced dataset check dengan FileNamingManager support"""
        try:
            paths = self.path_resolver.get_all_paths()
            
            status = {
                'service_ready': True,
                'paths': paths,
                'config': {
                    'types': self.config.get('augmentation', {}).get('types', ['combined']),
                    'num_variations': self.config.get('augmentation', {}).get('num_variations', 2),
                    'target_count': self.config.get('augmentation', {}).get('target_count', 500),
                    'normalization': self.config.get('preprocessing', {}).get('normalization', {}),
                    'file_naming': 'FileNamingManager integrated'
                }
            }
            
            # Enhanced dataset analysis per split dengan pattern detection
            for split in ['train', 'valid', 'test']:
                split_analysis = self._analyze_split_with_patterns(split)
                status.update(split_analysis)
            
            # Log comprehensive summary
            self._log_dataset_analysis(status)
            
            return status
        except Exception as e:
            return {'service_ready': False, 'error': str(e)}
    
    def _analyze_split_with_patterns(self, split: str) -> Dict[str, Any]:
        """🔍 Enhanced analysis dengan FileNamingManager pattern detection"""
        raw_path = Path(self.path_resolver.get_raw_path(split))
        aug_path = Path(self.path_resolver.get_augmented_path(split))
        prep_path = Path(self.path_resolver.get_preprocessed_path(split))
        
        analysis = {}
        
        # Raw data analysis
        if raw_path.exists():
            raw_images = list((raw_path / 'images').glob('rp_*.jpg')) if (raw_path / 'images').exists() else []
            raw_labels = list((raw_path / 'labels').glob('*.txt')) if (raw_path / 'labels').exists() else []
            analysis[f'{split}_raw_images'] = len(raw_images)
            analysis[f'{split}_raw_labels'] = len(raw_labels)
            analysis[f'{split}_raw_status'] = 'available' if raw_images and raw_labels else 'missing'
        else:
            analysis[f'{split}_raw_images'] = 0
            analysis[f'{split}_raw_labels'] = 0
            analysis[f'{split}_raw_status'] = 'not_found'
        
        # Augmented data analysis dengan variance pattern
        if aug_path.exists():
            aug_images = list((aug_path / 'images').glob('aug_*.jpg')) if (aug_path / 'images').exists() else []
            aug_labels = list((aug_path / 'labels').glob('aug_*.txt')) if (aug_path / 'labels').exists() else []
            analysis[f'{split}_augmented'] = len(aug_images)
            analysis[f'{split}_aug_labels'] = len(aug_labels)
            analysis[f'{split}_aug_status'] = 'available' if aug_images else 'empty'
            
            # Variance analysis
            variance_count = self._count_variance_patterns(aug_images)
            analysis[f'{split}_variance_count'] = variance_count
        else:
            analysis[f'{split}_augmented'] = 0
            analysis[f'{split}_aug_labels'] = 0
            analysis[f'{split}_aug_status'] = 'not_found'
            analysis[f'{split}_variance_count'] = 0
        
        # Preprocessed data analysis
        if prep_path.exists():
            prep_images = list((prep_path / 'images').glob('*.npy')) if (prep_path / 'images').exists() else []
            prep_labels = list((prep_path / 'labels').glob('*.txt')) if (prep_path / 'labels').exists() else []
            sample_aug = list((prep_path / 'images').glob('sample_aug_*.jpg')) if (prep_path / 'images').exists() else []
            
            analysis[f'{split}_preprocessed'] = len(prep_images)
            analysis[f'{split}_prep_labels'] = len(prep_labels)
            analysis[f'{split}_sample_aug'] = len(sample_aug)
            analysis[f'{split}_prep_status'] = 'available' if prep_images else 'empty'
        else:
            analysis[f'{split}_preprocessed'] = 0
            analysis[f'{split}_prep_labels'] = 0
            analysis[f'{split}_sample_aug'] = 0
            analysis[f'{split}_prep_status'] = 'not_found'
        
        return analysis
    
    def _count_variance_patterns(self, aug_files: list) -> int:
        """🔢 Count unique variance patterns"""
        unique_variances = set()
        
        for file_path in aug_files:
            try:
                parsed = self.engine.naming_manager.parse_filename(Path(file_path).name)
                if parsed and parsed.get('type') == 'augmented':
                    variance = parsed.get('variance', '001')
                    unique_variances.add(variance)
            except Exception:
                continue
        
        return len(unique_variances)
    
    def _log_dataset_analysis(self, status: Dict[str, Any]):
        """📋 Enhanced dataset analysis dengan pattern info"""
        self.logger.info("📊 Dataset Analysis Report (FileNamingManager):")
        
        for split in ['train', 'valid', 'test']:
            raw_imgs = status.get(f'{split}_raw_images', 0)
            raw_status = status.get(f'{split}_raw_status', 'unknown')
            aug_imgs = status.get(f'{split}_augmented', 0)
            aug_status = status.get(f'{split}_aug_status', 'unknown')
            prep_imgs = status.get(f'{split}_preprocessed', 0)
            prep_status = status.get(f'{split}_prep_status', 'unknown')
            sample_aug = status.get(f'{split}_sample_aug', 0)
            variance_count = status.get(f'{split}_variance_count', 0)
            
            # Status icons
            status_icons = {'available': "✅", 'not_found': "❌", 'unknown': "⚠️", 'empty': "📭", 'missing': "❌"}
            raw_icon = status_icons.get(raw_status, "⚠️")
            aug_icon = status_icons.get(aug_status, "⚠️")
            prep_icon = status_icons.get(prep_status, "⚠️")
            
            # Consolidate split status into single log entry (reduce verbosity)
            status_summary = f"{split.upper()}: Raw={raw_imgs}({raw_status}), Aug={aug_imgs}({aug_status}), Prep={prep_imgs}({prep_status})"
            if sample_aug > 0:
                status_summary += f", Samples={sample_aug}"
            self.logger.info(f"  📂 {status_summary}")
    
    def get_sampling(self, target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
        """📊 Enhanced sampling menggunakan AugmentationSampleGenerator"""
        try:
            result = self.sample_generator.generate_augmentation_samples(
                target_split=target_split,
                max_samples=max_samples,
                max_per_class=2
            )
            
            if result['status'] == 'success':
                self.logger.info(f"📊 Generated {len(result['samples'])} augmentation samples dari {target_split}")
            else:
                self.logger.warning(f"⚠️ Sampling failed: {result.get('message', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error generating samples: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'samples': []}
    
    def cleanup_data(self, target_split: str = None, target: str = 'both') -> Dict[str, Any]:
        """🧹 Configurable cleanup dengan pilihan target"""
        try:
            self.logger.info(f"🧹 Starting cleanup: target={target}, split={target_split or 'all'}")
            
            # Get current status before cleanup
            pre_cleanup_status = self.get_augmentation_status()
            
            result = {
                'status': 'success',
                'total_removed': 0,
                'augmented_removed': 0,
                'samples_removed': 0,
                'target': target,
                'target_split': target_split
            }
            
            # Cleanup augmented files + labels
            if target in ['augmented', 'both']:
                from smartcash.dataset.augmentor.utils.cleanup_manager import CleanupManager
                cleanup_manager = CleanupManager(self.config, self.progress)
                
                aug_result = cleanup_manager.cleanup_augmented_data(target_split=target_split)
                if aug_result.get('status') == 'success':
                    result['augmented_removed'] = aug_result.get('total_removed', 0)
                    result['total_removed'] += result['augmented_removed']
                    self.logger.info(f"✅ Cleaned {result['augmented_removed']} augmented files")
                else:
                    self.logger.warning(f"⚠️ Augmented cleanup failed: {aug_result.get('message', 'Unknown error')}")
            
            # Cleanup sample files
            if target in ['samples', 'both']:
                sample_result = self.sample_generator.cleanup_augmentation_samples(target_split)
                if sample_result.get('status') == 'success':
                    result['samples_removed'] = sample_result.get('total_removed', 0)
                    result['total_removed'] += result['samples_removed']
                    self.logger.info(f"✅ Cleaned {result['samples_removed']} sample files")
                else:
                    self.logger.warning(f"⚠️ Sample cleanup failed: {sample_result.get('message', 'Unknown error')}")
            
            # Enhanced summary with before/after comparison
            if result['total_removed'] > 0:
                post_cleanup_status = self.get_augmentation_status()
                result['cleanup_summary'] = self._create_cleanup_summary(
                    pre_cleanup_status, post_cleanup_status, target_split, target)
                self._log_cleanup_summary(result['cleanup_summary'])
            else:
                self.logger.info("ℹ️ No files were removed")
            
            return result
            
        except Exception as e:
            error_msg = f"Cleanup error: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_removed': 0}
    
    def cleanup_augmented_data(self, target_split: str = None) -> Dict[str, Any]:
        """Cleanup augmented files dan labels saja"""
        return self.cleanup_data(target_split=target_split, target='augmented')
    
    def cleanup_samples(self, target_split: str = None) -> Dict[str, Any]:
        """Cleanup sample files saja"""
        return self.cleanup_data(target_split=target_split, target='samples')
    
    def cleanup_all(self, target_split: str = None) -> Dict[str, Any]:
        """Cleanup semua: augmented + samples"""
        return self.cleanup_data(target_split=target_split, target='both')
    
    def _create_cleanup_summary(self, pre_status: Dict[str, Any], post_status: Dict[str, Any], 
                              target_split: str, target: str = 'both') -> Dict[str, Any]:
        """📊 Enhanced cleanup summary dengan target-specific info"""
        splits = [target_split] if target_split else ['train', 'valid', 'test']
        
        summary = {
            'target_splits': splits,
            'target_type': target,
            'files_removed': {},
            'total_samples_removed': 0,
            'total_augmented_removed': 0
        }
        
        for split in splits:
            pre_aug = pre_status.get(f'{split}_augmented', 0)
            pre_prep = pre_status.get(f'{split}_preprocessed', 0)
            pre_samples = pre_status.get(f'{split}_sample_aug', 0)
            
            post_aug = post_status.get(f'{split}_augmented', 0)
            post_prep = post_status.get(f'{split}_preprocessed', 0)
            post_samples = post_status.get(f'{split}_sample_aug', 0)
            
            aug_removed = pre_aug - post_aug
            prep_removed = pre_prep - post_prep
            samples_removed = pre_samples - post_samples
            
            summary['files_removed'][split] = {
                'augmented': aug_removed,
                'preprocessed': prep_removed,
                'samples': samples_removed,
                'total': aug_removed + prep_removed + samples_removed
            }
            
            summary['total_samples_removed'] += samples_removed
            summary['total_augmented_removed'] += aug_removed + prep_removed
        
        return summary
    
    def _log_cleanup_summary(self, summary: Dict[str, Any]):
        """📋 Enhanced cleanup summary dengan target-specific info"""
        target_type = summary.get('target_type', 'both')
        
        self.logger.success(f"🧹 Cleanup completed (target: {target_type})!")
        
        total_removed = 0
        total_samples = summary.get('total_samples_removed', 0)
        total_augmented = summary.get('total_augmented_removed', 0)
        
        for split, counts in summary['files_removed'].items():
            split_total = counts['total']
            total_removed += split_total
            
            if split_total > 0:
                if target_type == 'augmented':
                    self.logger.info(f"  📂 {split}: {counts['augmented']} aug + {counts['preprocessed']} prep = {counts['augmented'] + counts['preprocessed']} files")
                elif target_type == 'samples':
                    self.logger.info(f"  📂 {split}: {counts['samples']} sample files")
                else:  # both
                    self.logger.info(f"  📂 {split}: {counts['augmented']} aug + {counts['preprocessed']} prep + {counts['samples']} samples = {split_total} files")
        
        if target_type == 'augmented':
            self.logger.info(f"📊 Total augmented files removed: {total_augmented}")
        elif target_type == 'samples':
            self.logger.info(f"📊 Total sample files removed: {total_samples}")
        else:  # both
            self.logger.info(f"📊 Total files removed: {total_removed} ({total_augmented} augmented + {total_samples} samples)")
    
    def _execute_augmentation(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute augmentation dengan progress 1-3/4"""
        self._update_progress("overall", 1, 4, "Memulai augmentasi", progress_callback)
        
        def aug_progress_bridge(level: str, current: int, total: int, message: str):
            if level == "overall":
                mapped_progress = 1 + int((current / total) * 2)
                self._update_progress("overall", mapped_progress, 4, f"Augmentasi: {message}", progress_callback)
            else:
                self._update_progress(level, current, total, message, progress_callback)
        
        result = self.engine.augment_split(target_split, aug_progress_bridge)
        
        if result['status'] == 'success':
            total_gen = result.get('total_generated', 0)
            self._update_progress("overall", 3, 4, f"Augmentasi selesai: {total_gen} file generated", progress_callback)
            
        return result
    
    def _execute_normalization(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute normalization dengan progress 3-4/4"""
        self._update_progress("overall", 3, 4, "Memulai normalisasi", progress_callback)
        
        def norm_progress_bridge(level: str, current: int, total: int, message: str):
            if level == "overall":
                mapped_progress = 3 + int((current / total) * 1)
                self._update_progress("overall", mapped_progress, 4, f"Normalisasi: {message}", progress_callback)
            else:
                self._update_progress(level, current, total, message, progress_callback)
        
        aug_path = self.path_resolver.get_augmented_path(target_split)
        prep_path = self.path_resolver.get_preprocessed_path(target_split)
        
        result = self.normalizer.normalize_augmented_files(aug_path, prep_path, norm_progress_bridge)
        
        if result['status'] == 'success':
            total_norm = result.get('total_normalized', 0)
            self._update_progress("overall", 4, 4, f"Pipeline selesai: {total_norm} files normalized", progress_callback)
            
        return result
    
    def _report_start(self, target_split: str):
        """Report pipeline start dengan FileNamingManager info"""
        self.logger.info(f"🚀 Pipeline augmentasi untuk split: {target_split}")
        self.logger.info(f"📁 Raw: {self.path_resolver.get_raw_path(target_split)}")
        self.logger.info(f"📁 Augmented: {self.path_resolver.get_augmented_path(target_split)}")
        self.logger.info(f"📁 Preprocessed: {self.path_resolver.get_preprocessed_path(target_split)}")
        self.logger.info(f"🔧 FileNamingManager: integrated dengan variance support")
    
    def _update_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Update progress"""
        if self.progress:
            self.progress.update(level, current, total, message)
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass
    
    def _create_final_result(self, aug_result: Dict, norm_result: Dict, start_time: float) -> Dict[str, Any]:
        """Create enhanced final result dengan FileNamingManager info"""
        processing_time = time.time() - start_time
        
        # Extract summaries dari both phases
        aug_summary = aug_result.get('summary', {})
        norm_summary = norm_result.get('summary', {})
        
        result = {
            'status': 'success',
            'total_generated': aug_result.get('total_generated', 0),
            'total_normalized': norm_result.get('total_normalized', 0),
            'processing_time': processing_time,
            'phases': {
                'augmentation': aug_result,
                'normalization': norm_result
            },
            'pipeline_summary': {
                'augmentation': aug_summary,
                'normalization': norm_summary,
                'overall': {
                    'processing_time': f"{processing_time:.1f}s",
                    'pipeline_success': True,
                    'files_flow': f"{aug_result.get('total_generated', 0)} augmented → {norm_result.get('total_normalized', 0)} normalized",
                    'file_naming': 'FileNamingManager with variance support'
                }
            }
        }
        
        # Log final pipeline summary
        self._log_pipeline_summary(result)
        
        return result
    
    def _log_pipeline_summary(self, result: Dict[str, Any]):
        """📋 Enhanced pipeline summary dengan FileNamingManager info"""
        pipeline_summary = result.get('pipeline_summary', {})
        overall = pipeline_summary.get('overall', {})
        
        self.logger.success(f"🎉 Pipeline augmentasi selesai dalam {overall.get('processing_time', 'N/A')}!")
        self.logger.info(f"🔄 Flow: {overall.get('files_flow', 'N/A')}")
        self.logger.info(f"🔧 Naming: {overall.get('file_naming', 'N/A')}")
        
        # Log phase summaries if available
        aug_summary = pipeline_summary.get('augmentation', {})
        if aug_summary:
            aug_output = aug_summary.get('output', {})
            self.logger.info(f"📈 Augmentation: {aug_output.get('success_rate', 'N/A')} success rate @ intensity {aug_output.get('intensity_applied', 'N/A')}")
            self.logger.info(f"🔢 Variance pattern: {aug_output.get('variance_pattern', 'N/A')}")
        
        norm_summary = pipeline_summary.get('normalization', {})
        if norm_summary:
            norm_output = norm_summary.get('output', {})
            self.logger.info(f"🔧 Normalization: {norm_output.get('success_rate', 'N/A')} success rate → {norm_output.get('output_format', 'N/A')}")
            self.logger.info(f"📋 API: {norm_summary.get('configuration', {}).get('api_source', 'N/A')} normalization")


def create_augmentation_service(config: Dict[str, Any], progress_tracker=None) -> AugmentationService:
    """🏭 Fixed factory dengan consistent signature"""
    return AugmentationService(config, progress_tracker)

def run_augmentation_pipeline(config: Dict[str, Any], target_split: str = "train", 
                            progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """One-liner untuk run pipeline"""
    service = create_augmentation_service(config, progress_tracker)
    return service.run_augmentation_pipeline(target_split, progress_callback)

def get_augmentation_samples(config: Dict[str, Any], target_split: str = "train", 
                           max_samples: int = 5, progress_tracker=None) -> Dict[str, Any]:
    """One-liner untuk get sampling data"""
    service = create_augmentation_service(config, progress_tracker)
    return service.get_sampling(target_split, max_samples)