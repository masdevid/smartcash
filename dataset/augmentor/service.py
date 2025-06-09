"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Service augmentasi tanpa symlink dan dengan sampling service
"""

import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from smartcash.dataset.augmentor.core.engine import AugmentationEngine
from smartcash.dataset.augmentor.core.normalizer import NormalizationEngine
from smartcash.dataset.augmentor.utils.progress_bridge import ProgressBridge
from smartcash.dataset.augmentor.utils.path_resolver import PathResolver
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class AugmentationService:
    """🎯 Service augmentasi dengan sampling capability"""
    
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
        
    def run_augmentation_pipeline(self, target_split: str = "train", 
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🚀 Execute pipeline: augmentation + normalization (TANPA symlink)"""
        start_time = time.time()
        
        try:
            self._report_start(target_split)
            
            # Phase 1: Augmentation (0-50%)
            aug_result = self._execute_augmentation(target_split, progress_callback)
            if aug_result['status'] != 'success':
                return aug_result
                
            # Phase 2: Normalization (50-100%) 
            norm_result = self._execute_normalization(target_split, progress_callback)
            if norm_result['status'] != 'success':
                return norm_result
            
            return self._create_final_result(aug_result, norm_result, start_time)
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_generated': 0}
    
    def get_sampling(self, target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
        """📊 Get random samples untuk evaluasi (uuid, raw_image, aug_without_norm, aug_norm)"""
        try:
            import random
            import cv2
            import numpy as np
            
            # Get paths
            raw_path = Path(self.path_resolver.get_raw_path(target_split))
            aug_path = Path(self.path_resolver.get_augmented_path(target_split))
            norm_path = Path(self.path_resolver.get_preprocessed_path(target_split))
            
            # Find files dengan UUID yang sama di semua direktori
            samples = []
            raw_images = list((raw_path / 'images').glob('*.jpg')) if raw_path.exists() else []
            
            if not raw_images:
                return {'status': 'error', 'message': 'Tidak ada raw images ditemukan', 'samples': []}
            
            # Sample random files
            sampled_files = random.sample(raw_images, min(max_samples, len(raw_images)))
            
            for raw_file in sampled_files:
                try:
                    # Extract UUID dari filename (format: rp_001000_uuid_increment.jpg)
                    filename = raw_file.stem
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        uuid_part = parts[2]
                        
                        # Load raw image
                        raw_image = cv2.imread(str(raw_file))
                        
                        # Find corresponding augmented file
                        aug_pattern = f"aug_rp_*_{uuid_part}_*_*.jpg"
                        aug_files = list((aug_path / 'images').glob(aug_pattern)) if aug_path.exists() else []
                        aug_image = cv2.imread(str(aug_files[0])) if aug_files else None
                        
                        # Find corresponding normalized file (.npy)
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
                    self.logger.warning(f"Error processing sample {raw_file}: {str(e)}")
                    continue
            
            self.logger.info(f"Generated {len(samples)} samples dari {target_split}")
            
            return {
                'status': 'success',
                'samples': samples,
                'total_samples': len(samples),
                'target_split': target_split
            }
            
        except Exception as e:
            error_msg = f"Error generating samples: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'samples': []}
    
    def _execute_augmentation(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute augmentation dengan progress 0-50%"""
        self._update_progress("overall", 1, 4, "Memulai augmentasi", progress_callback)
        
        # Callback untuk map engine progress ke overall 0-50%
        def aug_progress_bridge(level: str, current: int, total: int, message: str):
            if level == "overall":
                mapped_progress = 1 + int((current / total) * 2)  # Map ke 1-3
                self._update_progress("overall", mapped_progress, 4, f"Augmentasi: {message}", progress_callback)
            else:
                self._update_progress(level, current, total, message, progress_callback)
        
        result = self.engine.augment_split(target_split, aug_progress_bridge)
        
        if result['status'] == 'success':
            self._update_progress("overall", 3, 4, f"Augmentasi selesai: {result['total_generated']} file", progress_callback)
            
        return result
    
    def _execute_normalization(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute normalization dengan progress 50-100%"""
        self._update_progress("overall", 3, 4, "Memulai normalisasi", progress_callback)
        
        # Callback untuk map normalization progress ke 75-100%
        def norm_progress_bridge(level: str, current: int, total: int, message: str):
            if level == "overall":
                mapped_progress = 3 + int((current / total) * 1)  # Map ke 3-4
                self._update_progress("overall", mapped_progress, 4, f"Normalisasi: {message}", progress_callback)
            else:
                self._update_progress(level, current, total, message, progress_callback)
        
        # Get paths
        aug_path = self.path_resolver.get_augmented_path(target_split)
        prep_path = self.path_resolver.get_preprocessed_path(target_split)
        
        result = self.normalizer.normalize_augmented_files(aug_path, prep_path, norm_progress_bridge)
        
        if result['status'] == 'success':
            self._update_progress("overall", 4, 4, f"Pipeline selesai: {result['total_normalized']} file dinormalisasi", progress_callback)
            
        return result
    
    def cleanup_augmented_data(self, target_split: str = None) -> Dict[str, Any]:
        """🧹 Cleanup augmented files (TANPA symlinks)"""
        try:
            from smartcash.dataset.augmentor.utils.cleanup_manager import CleanupManager
            
            cleanup_manager = CleanupManager(self.config, self.progress)
            return cleanup_manager.cleanup_augmented_data(target_split)
            
        except Exception as e:
            error_msg = f"Cleanup error: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """📊 Get comprehensive augmentation status"""
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
            
            # Check existing files
            for split in ['train', 'valid', 'test']:
                aug_path = self.path_resolver.get_augmented_path(split)
                prep_path = self.path_resolver.get_preprocessed_path(split)
                
                # Count .jpg files in augmented
                status[f'{split}_augmented'] = len(list(Path(aug_path).glob('aug_*.jpg'))) if Path(aug_path).exists() else 0
                
                # Count .npy files in preprocessed
                status[f'{split}_preprocessed'] = len(list(Path(prep_path).glob('*.npy'))) if Path(prep_path).exists() else 0
            
            return status
            
        except Exception as e:
            return {'service_ready': False, 'error': str(e)}
    
    def _report_start(self, target_split: str):
        """Report pipeline start"""
        self.logger.info(f"Memulai pipeline augmentasi untuk split: {target_split}")
        self.logger.info(f"Raw data: {self.path_resolver.get_raw_path(target_split)}")
        self.logger.info(f"Augmented: {self.path_resolver.get_augmented_path(target_split)}")
        self.logger.info(f"Preprocessed: {self.path_resolver.get_preprocessed_path(target_split)}")
    
    def _update_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Update progress dengan dual reporting"""
        if self.progress:
            self.progress.update(level, current, total, message)
        
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass
    
    def _create_final_result(self, aug_result: Dict, norm_result: Dict, start_time: float) -> Dict[str, Any]:
        """Create comprehensive final result TANPA symlinks"""
        processing_time = time.time() - start_time
        
        result = {
            'status': 'success',
            'total_generated': aug_result.get('total_generated', 0),
            'total_normalized': norm_result.get('total_normalized', 0),
            'processing_time': processing_time,
            'phases': {
                'augmentation': aug_result,
                'normalization': norm_result
            }
        }
        
        self.logger.success(f"Pipeline selesai dalam {processing_time:.1f}s")
        self.logger.info(f"Generated: {result['total_generated']}, Normalized: {result['total_normalized']}")
        
        return result


# Factory functions
def create_augmentation_service(config: Dict[str, Any], progress_tracker=None) -> AugmentationService:
    """Factory untuk create augmentation service"""
    return AugmentationService(config, progress_tracker)

def run_augmentation_pipeline(config: Dict[str, Any], target_split: str = "train", 
                            progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """One-liner untuk run complete pipeline"""
    service = create_augmentation_service(config, progress_tracker)
    return service.run_augmentation_pipeline(target_split, progress_callback)

def get_augmentation_samples(config: Dict[str, Any], target_split: str = "train", 
                           max_samples: int = 5, progress_tracker=None) -> Dict[str, Any]:
    """One-liner untuk get sampling data"""
    service = create_augmentation_service(config, progress_tracker)
    return service.get_sampling(target_split, max_samples)