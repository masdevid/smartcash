"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Enhanced service dengan improved dataset check dan summary
"""

import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from smartcash.dataset.augmentor.core.engine import AugmentationEngine
from smartcash.dataset.augmentor.core.normalizer import NormalizationEngine
from smartcash.dataset.augmentor.utils.progress_bridge import ProgressBridge
from smartcash.dataset.augmentor.utils.path_resolver import PathResolver
from smartcash.dataset.augmentor.utils.cleanup_manager import CleanupManager
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class AugmentationService:
    """🎯 Enhanced service dengan comprehensive dataset check dan improved summary"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_tracker=None):
        self.logger = get_logger(__name__)
        if config is None:
            self.config = get_default_augmentation_config()
            self.logger.info("Menggunakan default config")
        else:
            self.config = validate_augmentation_config(config)
            self.logger.info("Config validated dan merged")  
        
        self.progress = ProgressBridge(progress_tracker) if progress_tracker else None
        self.engine = AugmentationEngine(self.config, self.progress)
        self.normalizer = NormalizationEngine(self.config, self.progress)
        self.path_resolver = PathResolver(self.config)
        self.cleanup_manager = CleanupManager(self.config, self.progress)
        
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
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """📊 Enhanced dataset check dengan comprehensive analysis"""
        try:
            paths = self.path_resolver.get_all_paths()
            
            status = {
                'service_ready': True,
                'paths': paths,
                'config': {
                    'types': self.config.get('augmentation', {}).get('types', ['combined']),
                    'num_variations': self.config.get('augmentation', {}).get('num_variations', 2),
                    'target_count': self.config.get('augmentation', {}).get('target_count', 500),
                    'normalization': self.config.get('preprocessing', {}).get('normalization', {})
                }
            }
            
            # Enhanced dataset analysis per split
            for split in ['train', 'valid', 'test']:
                split_analysis = self._analyze_split_comprehensive(split)
                status.update(split_analysis)
            
            # Log comprehensive summary
            self._log_dataset_analysis(status)
            
            return status
        except Exception as e:
            return {'service_ready': False, 'error': str(e)}
            
    def cleanup_processed_data(self, target: str = 'augmented', target_split: str = None) -> Dict[str, Any]:
        """🧹 Membersihkan file hasil augmentasi
        
        Args:
            target: Jenis file yang akan dibersihkan. Default: 'augmented'
            target_split: Nama split yang akan dibersihkan (contoh: 'train', 'valid', 'test'). 
                       Jika None, semua split akan dibersihkan.
                        
        Returns:
            Dict berisi status, total file yang dihapus, dan pesan
        """
        try:
            self.logger.info(f"Memulai cleanup data dengan target={target}, split={target_split}")
            result = self.cleanup_manager.cleanup_data(target=target, target_split=target_split)
            
            if result.get('status') == 'success':
                self.logger.info(f"Cleanup berhasil: {result.get('total_removed', 0)} file dihapus")
            else:
                self.logger.error(f"Cleanup gagal: {result.get('message', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            error_msg = f"Error saat membersihkan data: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_removed': 0}
    
    def _analyze_split_comprehensive(self, split: str) -> Dict[str, Any]:
        """🔍 NEW: Comprehensive analysis per split"""
        raw_path = Path(self.path_resolver.get_raw_path(split))
        aug_path = Path(self.path_resolver.get_augmented_path(split))
        prep_path = Path(self.path_resolver.get_preprocessed_path(split))
        
        analysis = {}
        
        # Raw data analysis
        if raw_path.exists():
            raw_images = list((raw_path / 'images').glob('*.jpg')) if (raw_path / 'images').exists() else []
            raw_labels = list((raw_path / 'labels').glob('*.txt')) if (raw_path / 'labels').exists() else []
            analysis[f'{split}_raw_images'] = len(raw_images)
            analysis[f'{split}_raw_labels'] = len(raw_labels)
            analysis[f'{split}_raw_status'] = 'available' if raw_images and raw_labels else 'missing'
        else:
            analysis[f'{split}_raw_images'] = 0
            analysis[f'{split}_raw_labels'] = 0
            analysis[f'{split}_raw_status'] = 'not_found'
        
        # Augmented data analysis
        if aug_path.exists():
            aug_images = list((aug_path / 'images').glob('aug_*.jpg')) if (aug_path / 'images').exists() else []
            aug_labels = list((aug_path / 'labels').glob('aug_*.txt')) if (aug_path / 'labels').exists() else []
            analysis[f'{split}_augmented'] = len(aug_images)
            analysis[f'{split}_aug_labels'] = len(aug_labels)
            analysis[f'{split}_aug_status'] = 'available' if aug_images else 'empty'
        else:
            analysis[f'{split}_augmented'] = 0
            analysis[f'{split}_aug_labels'] = 0
            analysis[f'{split}_aug_status'] = 'not_found'
        
        # Preprocessed data analysis
        if prep_path.exists():
            prep_images = list((prep_path / 'images').glob('*.npy')) if (prep_path / 'images').exists() else []
            prep_labels = list((prep_path / 'labels').glob('*.txt')) if (prep_path / 'labels').exists() else []
            analysis[f'{split}_preprocessed'] = len(prep_images)
            analysis[f'{split}_prep_labels'] = len(prep_labels)
            analysis[f'{split}_prep_status'] = 'available' if prep_images else 'empty'
        else:
            analysis[f'{split}_preprocessed'] = 0
            analysis[f'{split}_prep_labels'] = 0
            analysis[f'{split}_prep_status'] = 'not_found'
        
        return analysis
    
    def _log_dataset_analysis(self, status: Dict[str, Any]):
        """📋 NEW: Log comprehensive dataset analysis"""
        self.logger.info("📊 Dataset Analysis Report:")
        
        for split in ['train', 'valid', 'test']:
            raw_imgs = status.get(f'{split}_raw_images', 0)
            raw_status = status.get(f'{split}_raw_status', 'unknown')
            aug_imgs = status.get(f'{split}_augmented', 0)
            aug_status = status.get(f'{split}_aug_status', 'unknown')
            prep_imgs = status.get(f'{split}_preprocessed', 0)
            prep_status = status.get(f'{split}_prep_status', 'unknown')
            
            # Status icons
            raw_icon = "✅" if raw_status == 'available' else "❌" if raw_status == 'not_found' else "⚠️"
            aug_icon = "✅" if aug_status == 'available' else "❌" if aug_status == 'not_found' else "⚠️"
            prep_icon = "✅" if prep_status == 'available' else "❌" if prep_status == 'not_found' else "⚠️"
            
            self.logger.info(f"  📂 {split.upper()}:")
            self.logger.info(f"    {raw_icon} Raw: {raw_imgs} images ({raw_status})")
            self.logger.info(f"    {aug_icon} Augmented: {aug_imgs} images ({aug_status})")
            self.logger.info(f"    {prep_icon} Preprocessed: {prep_imgs} .npy files ({prep_status})")
    
    def cleanup_data(self, target: str = 'both', target_split: str = None) -> Dict[str, Any]:
        """🧹 Enhanced cleanup dengan comprehensive reporting
        
        Args:
            target: Jenis file yang akan dibersihkan ('augmented', 'preprocessed', atau 'both')
            target_split: Nama split yang akan dibersihkan (contoh: 'train', 'valid', 'test')
            
        Returns:
            Dict berisi status, total file yang dihapus, dan pesan
        """
        try:
            from smartcash.dataset.augmentor.utils.cleanup_manager import CleanupManager
            cleanup_manager = CleanupManager(self.config, self.progress)
            
            # Get current status before cleanup
            pre_cleanup_status = self.get_augmentation_status()
            
            # Execute cleanup
            result = cleanup_manager.cleanup_data(target=target, target_split=target_split)
            
            # Enhanced result with before/after comparison
            if result.get('status') == 'success':
                post_cleanup_status = self.get_augmentation_status()
                result['cleanup_summary'] = self._create_cleanup_summary(
                    pre_cleanup_status, post_cleanup_status, target_split)
                self._log_cleanup_summary(result['cleanup_summary'])
            
            return result
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def cleanup_augmented_data(self, target_split: str = None) -> Dict[str, Any]:
        """Alias untuk membersihkan data augmented.
        
        Args:
            target_split: Nama split yang akan dibersihkan (contoh: 'train', 'valid', 'test')
            
        Returns:
            Dict berisi status, total file yang dihapus, dan pesan
        """
        return self.cleanup_data(target='augmented', target_split=target_split)
        
    def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
        """Alias untuk membersihkan data preprocessed.
        
        Args:
            target_split: Nama split yang akan dibersihkan (contoh: 'train', 'valid', 'test')
            
        Returns:
            Dict berisi status, total file yang dihapus, dan pesan
        """
        return self.cleanup_data(target='preprocessed', target_split=target_split)
    
    def _create_cleanup_summary(self, pre_status: Dict[str, Any], post_status: Dict[str, Any], target_split: str) -> Dict[str, Any]:
        """📊 NEW: Create cleanup summary"""
        splits = [target_split] if target_split else ['train', 'valid', 'test']
        
        summary = {
            'target_splits': splits,
            'files_removed': {},
            'directories_cleaned': []
        }
        
        for split in splits:
            pre_aug = pre_status.get(f'{split}_augmented', 0)
            pre_prep = pre_status.get(f'{split}_preprocessed', 0)
            post_aug = post_status.get(f'{split}_augmented', 0)
            post_prep = post_status.get(f'{split}_preprocessed', 0)
            
            summary['files_removed'][split] = {
                'augmented': pre_aug - post_aug,
                'preprocessed': pre_prep - post_prep,
                'total': (pre_aug - post_aug) + (pre_prep - post_prep)
            }
        
        return summary
    
    def _log_cleanup_summary(self, summary: Dict[str, Any]):
        """📋 NEW: Log cleanup summary"""
        self.logger.success("🧹 Cleanup completed!")
        
        total_removed = 0
        for split, counts in summary['files_removed'].items():
            split_total = counts['total']
            total_removed += split_total
            if split_total > 0:
                self.logger.info(f"  📂 {split}: {counts['augmented']} augmented + {counts['preprocessed']} preprocessed = {split_total} files")
        
        self.logger.info(f"📊 Total files removed: {total_removed}")
    
    def get_sampling(self, target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
        """📊 Enhanced sampling dengan comprehensive file analysis"""
        try:
            import random
            import cv2
            import numpy as np
            
            raw_path = Path(self.path_resolver.get_raw_path(target_split))
            aug_path = Path(self.path_resolver.get_augmented_path(target_split))
            norm_path = Path(self.path_resolver.get_preprocessed_path(target_split))
            
            samples = []
            raw_images = list((raw_path / 'images').glob('*.jpg')) if raw_path.exists() else []
            
            if not raw_images:
                return {'status': 'error', 'message': 'Tidak ada raw images', 'samples': []}
            
            sampled_files = random.sample(raw_images, min(max_samples, len(raw_images)))
            
            for raw_file in sampled_files:
                try:
                    filename = raw_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        uuid_part = parts[2]
                        
                        # Load raw image
                        raw_image = cv2.imread(str(raw_file))
                        
                        # Find augmented (.jpg)
                        aug_pattern = f"aug_rp_*_{uuid_part}_*_*.jpg"
                        aug_files = list((aug_path / 'images').glob(aug_pattern)) if aug_path.exists() else []
                        aug_image = cv2.imread(str(aug_files[0])) if aug_files else None
                        
                        # Find normalized (.npy)
                        norm_pattern = f"aug_rp_*_{uuid_part}_*_*.npy"
                        norm_files = list((norm_path / 'images').glob(norm_pattern)) if norm_path.exists() else []
                        norm_image = np.load(str(norm_files[0])) if norm_files else None
                        
                        sample_data = {
                            'uuid': uuid_part,
                            'filename': filename,
                            'raw_image': raw_image.tolist() if raw_image is not None else None,
                            'aug_without_norm': aug_image.tolist() if aug_image is not None else None,
                            'aug_norm': norm_image.tolist() if norm_image is not None else None,
                            'raw_path': str(raw_file) if raw_image is not None else None,
                            'aug_path': str(aug_files[0]) if aug_files else None,
                            'norm_path': str(norm_files[0]) if norm_files else None
                        }
                        
                        samples.append(sample_data)
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing sample {raw_file}: {str(e)}")
                    continue
            
            self.logger.info(f"📊 Generated {len(samples)} samples dari {target_split}")
            
            return {
                'status': 'success',
                'samples': samples,
                'total_samples': len(samples),
                'target_split': target_split
            }
            
        except Exception as e:
            error_msg = f"Error generating samples: {str(e)}"
            self.logger.error(error_msg)
    
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
        """Report pipeline start"""
        self.logger.info(f"🚀 Pipeline augmentasi untuk split: {target_split}")
        self.logger.info(f"📁 Raw: {self.path_resolver.get_raw_path(target_split)}")
        self.logger.info(f"📁 Augmented: {self.path_resolver.get_augmented_path(target_split)}")
        self.logger.info(f"📁 Preprocessed: {self.path_resolver.get_preprocessed_path(target_split)}")
    
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
        """Create enhanced final result dengan comprehensive summary"""
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
                    'files_flow': f"{aug_result.get('total_generated', 0)} augmented → {norm_result.get('total_normalized', 0)} normalized"
                }
            }
        }
        
        # Log final pipeline summary
        self._log_pipeline_summary(result)
        
        return result
    
    def _log_pipeline_summary(self, result: Dict[str, Any]):
        """📋 NEW: Log comprehensive pipeline summary"""
        pipeline_summary = result.get('pipeline_summary', {})
        overall = pipeline_summary.get('overall', {})
        
        self.logger.success(f"🎉 Pipeline augmentasi selesai dalam {overall.get('processing_time', 'N/A')}!")
        self.logger.info(f"🔄 Flow: {overall.get('files_flow', 'N/A')}")
        
        # Log phase summaries if available
        aug_summary = pipeline_summary.get('augmentation', {})
        if aug_summary:
            aug_output = aug_summary.get('output', {})
            self.logger.info(f"📈 Augmentation: {aug_output.get('success_rate', 'N/A')} success rate @ intensity {aug_output.get('intensity_applied', 'N/A')}")
        
        norm_summary = pipeline_summary.get('normalization', {})
        if norm_summary:
            norm_output = norm_summary.get('output', {})
            self.logger.info(f"🔧 Normalization: {norm_output.get('success_rate', 'N/A')} success rate → {norm_output.get('output_format', 'N/A')}")


def create_augmentation_service(config: Dict[str, Any], progress_tracker=None) -> AugmentationService:
    """Factory untuk create augmentation service"""
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