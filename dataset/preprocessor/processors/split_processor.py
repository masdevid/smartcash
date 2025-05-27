"""
File: smartcash/dataset/preprocessor/processors/split_processor.py
Deskripsi: Processor untuk single split dengan fixed callback dan better error handling
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.processors.batch_processor import BatchProcessor
from smartcash.dataset.preprocessor.utils.preprocessing_paths import PreprocessingPaths
from smartcash.dataset.preprocessor.storage.preprocessed_storage_manager import PreprocessedStorageManager

class SplitProcessor:
    """Processor untuk single split dataset dengan fixed callback"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        
        # Initialize components
        self.batch_processor = BatchProcessor(config, logger)
        self.path_manager = PreprocessingPaths(config, logger)
        self.storage = PreprocessedStorageManager(
            config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'), 
            logger
        )
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback dengan fixed handling"""
        self._progress_callback = callback
        
        # Fixed batch callback - simple positional args
        def batch_callback(progress, message):
            if self._progress_callback:
                try:
                    mapped_progress = 15 + (progress * 70 / 100)  # Map to 15-85% range
                    self._progress_callback(int(mapped_progress), f"Batch: {message}")
                except Exception as e:
                    self.logger.debug(f"🔧 Batch callback error: {str(e)}")
        
        self.batch_processor.register_progress_callback(batch_callback)
    
    def process_split_dataset(self, split: str, processing_config: Dict[str, Any], 
                            force_reprocess: bool = False) -> Dict[str, Any]:
        """Process single split dengan comprehensive error handling"""
        start_time = time.time()
        
        try:
            # Setup dan validasi (0-15%)
            self._notify_progress(5, f"🔧 Setup processing untuk split {split}")
            
            setup_result = self._setup_split_processing(split, processing_config, force_reprocess)
            if not setup_result['success']:
                return {'success': False, 'error': setup_result['message'], 'processed': 0, 'failed': 1}
            
            source_info = setup_result['source_info']
            target_info = setup_result['target_info']
            
            # Check existing preprocessed (skip if not force)
            if not force_reprocess and self._is_already_processed(split):
                processed_count = self._count_processed_files(target_info['images_dir'])
                self._notify_progress(100, f"✅ Split {split} sudah diproses ({processed_count} gambar)")
                return {'success': True, 'processed': processed_count, 'skipped': processed_count, 'failed': 0}
            
            self._notify_progress(15, f"📊 Validasi selesai: {source_info['image_count']} gambar ditemukan")
            
            # Batch processing (15-85%)
            self._notify_progress(20, f"🚀 Memulai batch processing split {split}")
            
            batch_result = self.batch_processor.process_image_batch_with_renaming(
                source_info['images_dir'], source_info['labels_dir'],
                target_info['images_dir'], target_info['labels_dir'],
                processing_config
            )
            
            if not batch_result['success']:
                return {'success': False, 'error': batch_result['message'], 'processed': 0, 'failed': 1}
            
            # Finalization (85-100%)
            self._notify_progress(90, f"📋 Finalisasi processing split {split}")
            
            final_stats = self._finalize_split_processing(split, batch_result, processing_config)
            final_stats['processing_time'] = time.time() - start_time
            
            self._notify_progress(100, f"✅ Split {split}: {final_stats['processed']} gambar dalam {final_stats['processing_time']:.1f}s")
            
            self.logger.success(f"✅ Split {split} selesai: {final_stats['processed']} gambar")
            return final_stats
            
        except Exception as e:
            error_msg = f"Error processing split {split}: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg, 'processed': 0, 'failed': 1}
    
    def _setup_split_processing(self, split: str, processing_config: Dict[str, Any], 
                              force_reprocess: bool) -> Dict[str, Any]:
        """Setup split processing dengan path validation"""
        try:
            # Resolve source paths
            source_paths = self.path_manager.resolve_source_paths(split)
            if not source_paths['valid']:
                return {'success': False, 'message': f"Source paths invalid: {source_paths['message']}"}
            
            # Setup target paths
            target_paths = self.path_manager.setup_target_paths(split)
            if not target_paths['success']:
                return {'success': False, 'message': f"Target setup failed: {target_paths['message']}"}
            
            # Count source files
            images_dir = Path(source_paths['images_dir'])
            labels_dir = Path(source_paths['labels_dir'])
            
            image_files = [f for f in images_dir.glob('*.*') 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            label_files = [f for f in labels_dir.glob('*.txt')] if labels_dir.exists() else []
            
            return {
                'success': True,
                'source_info': {
                    'images_dir': images_dir,
                    'labels_dir': labels_dir,
                    'image_count': len(image_files),
                    'label_count': len(label_files)
                },
                'target_info': target_paths
            }
            
        except Exception as e:
            return {'success': False, 'message': f"Setup error: {str(e)}"}
    
    def _is_already_processed(self, split: str) -> bool:
        """Check apakah split sudah diproses"""
        split_path = self.storage.get_split_path(split)
        images_path = split_path / 'images'
        return images_path.exists() and len(list(images_path.glob('*.*'))) > 0
    
    def _count_processed_files(self, target_dir: Path) -> int:
        """Count processed files"""
        if not target_dir.exists():
            return 0
        return len([f for f in target_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    def _finalize_split_processing(self, split: str, batch_result: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize split processing dengan stats"""
        final_stats = {
            'success': batch_result['success'],
            'processed': batch_result.get('processed', 0),
            'skipped': batch_result.get('skipped', 0),
            'failed': batch_result.get('failed', 0),
            'batch_count': batch_result.get('batch_count', 0)
        }
        
        # Save stats
        processing_stats = {**final_stats, 'config_used': processing_config, 'end_time': time.time()}
        self.storage.update_stats(split, processing_stats)
        
        return final_stats
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            'processor_ready': True,
            'batch_processor_status': self.batch_processor.get_batch_status(),
            'storage_ready': self.storage is not None,
            'progress_callback_registered': self._progress_callback is not None
        }
    
    def cleanup_processor_state(self) -> None:
        """Cleanup processor state"""
        self.batch_processor.cleanup_batch_state()
        self._progress_callback = None
    
    def _notify_progress(self, progress: int, message: str):
        """Internal progress notification dengan simple args"""
        if self._progress_callback:
            try:
                self._progress_callback(progress, message)
            except Exception as e:
                self.logger.debug(f"🔧 Split progress error: {str(e)}")