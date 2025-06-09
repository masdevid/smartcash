"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Enhanced augmentation service dengan progress tracking dan normalization otomatis
"""

import time
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path

from smartcash.dataset.augmentor.core.engine import AugmentationEngine
from smartcash.dataset.augmentor.core.normalizer import NormalizationEngine
from smartcash.dataset.augmentor.utils.progress_bridge import ProgressBridge
from smartcash.dataset.augmentor.utils.path_resolver import PathResolver
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class AugmentationService:
    """🎯 Enhanced service dengan progress tracking dan auto-normalization"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_tracker=None):
        self.logger = get_logger(__name__)
        # Use defaults jika config tidak disediakan
        if config is None:
            self.config = get_default_augmentation_config()
            self.logger.info("📋 Menggunakan default config dari augmentation_config.yaml")
        else:
            # Validate dan merge dengan defaults
            self.config = validate_augmentation_config(config)
            self.logger.info("📋 Config validated dan merged dengan defaults")  
        
        # Setup progress bridge untuk UI integration
        self.progress = ProgressBridge(progress_tracker) if progress_tracker else None
        
        # Initialize core engines
        self.engine = AugmentationEngine(self.config, self.progress)
        self.normalizer = NormalizationEngine(self.config, self.progress)
        
        # Path resolver untuk smart path handling
        self.path_resolver = PathResolver(self.config)
        
    def run_augmentation_pipeline(self, target_split: str = "train", 
                                progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🚀 Execute full pipeline: augmentation + normalization + symlink"""
        start_time = time.time()
        
        try:
            self._report_start(target_split)
            
            # Phase 1: Augmentation (0-70%)
            aug_result = self._execute_augmentation(target_split, progress_callback)
            if aug_result['status'] != 'success':
                return aug_result
                
            # Phase 2: Normalization (70-90%) 
            norm_result = self._execute_normalization(target_split, progress_callback)
            if norm_result['status'] != 'success':
                return norm_result
                
            # Phase 3: Symlink creation (90-100%)
            symlink_result = self._create_symlinks(target_split, progress_callback)
            
            return self._create_final_result(aug_result, norm_result, symlink_result, start_time)
            
        except Exception as e:
            error_msg = f"🚨 Pipeline error: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_generated': 0}
    
    def _execute_augmentation(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute augmentation dengan progress 0-70%"""
        self._update_progress("overall", 5, 100, f"🎨 Memulai augmentasi {target_split}", progress_callback)
        
        # Callback untuk map engine progress ke overall 0-70%
        def aug_progress_bridge(level: str, current: int, total: int, message: str):
            if level == "overall":
                mapped_progress = int((current / total) * 70)  # Map ke 0-70%
                self._update_progress("overall", mapped_progress, 100, f"🎨 {message}", progress_callback)
            else:
                self._update_progress(level, current, total, message, progress_callback)
        
        result = self.engine.augment_split(target_split, aug_progress_bridge)
        
        if result['status'] == 'success':
            self._update_progress("overall", 70, 100, f"✅ Augmentasi selesai: {result['total_generated']} file", progress_callback)
            
        return result
    
    def _execute_normalization(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute normalization dengan progress 70-90%"""
        self._update_progress("overall", 72, 100, f"🔧 Memulai normalisasi", progress_callback)
        
        # Callback untuk map normalization progress ke 70-90%
        def norm_progress_bridge(level: str, current: int, total: int, message: str):
            if level == "overall":
                mapped_progress = 70 + int((current / total) * 20)  # Map ke 70-90%
                self._update_progress("overall", mapped_progress, 100, f"🔧 {message}", progress_callback)
            else:
                self._update_progress(level, current, total, message, progress_callback)
        
        # Get augmented files path
        aug_path = self.path_resolver.get_augmented_path(target_split)
        prep_path = self.path_resolver.get_preprocessed_path(target_split)
        
        result = self.normalizer.normalize_augmented_files(aug_path, prep_path, norm_progress_bridge)
        
        if result['status'] == 'success':
            self._update_progress("overall", 90, 100, f"✅ Normalisasi selesai: {result['total_normalized']} file", progress_callback)
            
        return result
    
    def _create_symlinks(self, target_split: str, progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Create symlinks dengan progress 90-100%"""
        self._update_progress("overall", 92, 100, f"🔗 Membuat symlink", progress_callback)
        
        try:
            from smartcash.dataset.augmentor.utils.symlink_manager import SymlinkManager
            
            symlink_manager = SymlinkManager(self.config)
            aug_path = self.path_resolver.get_augmented_path(target_split)
            prep_path = self.path_resolver.get_preprocessed_path(target_split)
            
            result = symlink_manager.create_augmented_symlinks(aug_path, prep_path)
            
            self._update_progress("overall", 100, 100, f"✅ Pipeline selesai", progress_callback)
            return result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Symlink creation failed: {str(e)}")
            return {'status': 'partial', 'message': 'Symlink creation failed but augmentation successful'}
    
    def cleanup_augmented_data(self, target_split: str = None) -> Dict[str, Any]:
        """🧹 Cleanup augmented files dan symlinks"""
        try:
            from smartcash.dataset.augmentor.utils.cleanup_manager import CleanupManager
            
            cleanup_manager = CleanupManager(self.config, self.progress)
            return cleanup_manager.cleanup_augmented_data(target_split)
            
        except Exception as e:
            error_msg = f"🚨 Cleanup error: {str(e)}"
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
                
                status[f'{split}_augmented'] = len(list(Path(aug_path).glob('aug_*.jpg'))) if Path(aug_path).exists() else 0
                status[f'{split}_preprocessed'] = len(list(Path(prep_path).glob('*.jpg'))) if Path(prep_path).exists() else 0
            
            return status
            
        except Exception as e:
            return {'service_ready': False, 'error': str(e)}
    
    def _report_start(self, target_split: str):
        """Report pipeline start"""
        self.logger.info(f"🚀 Memulai pipeline augmentasi untuk split: {target_split}")
        self.logger.info(f"📁 Raw data: {self.path_resolver.get_raw_path(target_split)}")
        self.logger.info(f"📁 Augmented: {self.path_resolver.get_augmented_path(target_split)}")
        self.logger.info(f"📁 Preprocessed: {self.path_resolver.get_preprocessed_path(target_split)}")
    
    def _update_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Update progress dengan dual reporting"""
        # Report ke progress tracker
        if self.progress:
            self.progress.update(level, current, total, message)
        
        # Report ke callback jika disediakan
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass  # Silent fail untuk callback errors
    
    def _create_final_result(self, aug_result: Dict, norm_result: Dict, 
                           symlink_result: Dict, start_time: float) -> Dict[str, Any]:
        """Create comprehensive final result"""
        processing_time = time.time() - start_time
        
        result = {
            'status': 'success',
            'total_generated': aug_result.get('total_generated', 0),
            'total_normalized': norm_result.get('total_normalized', 0),
            'symlinks_created': symlink_result.get('total_created', 0),
            'processing_time': processing_time,
            'phases': {
                'augmentation': aug_result,
                'normalization': norm_result,
                'symlinks': symlink_result
            }
        }
        
        self.logger.success(f"🎉 Pipeline selesai dalam {processing_time:.1f}s")
        self.logger.info(f"📊 Generated: {result['total_generated']}, Normalized: {result['total_normalized']}")
        
        return result


# Factory functions
def create_augmentation_service(config: Dict[str, Any], progress_tracker=None) -> AugmentationService:
    """🏭 Factory untuk create augmentation service"""
    return AugmentationService(config, progress_tracker)

def run_augmentation_pipeline(config: Dict[str, Any], target_split: str = "train", 
                            progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🚀 One-liner untuk run complete pipeline"""
    service = create_augmentation_service(config, progress_tracker)
    return service.run_augmentation_pipeline(target_split, progress_callback)