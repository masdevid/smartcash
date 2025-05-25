"""
File: smartcash/dataset/preprocessor/processors/split_processor.py
Deskripsi: Fixed processor dengan clean callback parameter handling - no conflicts
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.dataset.preprocessor.processors.batch_processor import BatchProcessor
from smartcash.dataset.preprocessor.utils.preprocessing_paths import PreprocessingPaths
from smartcash.dataset.preprocessor.storage.preprocessed_storage_manager import PreprocessedStorageManager


class SplitProcessor:
    """Fixed processor untuk single split dataset - clean callback handling."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize split processor dengan dependencies."""
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        
        # Initialize processing components
        self.batch_processor = BatchProcessor(config, logger)
        self.path_manager = PreprocessingPaths(config, logger)
        self.storage = PreprocessedStorageManager(
            config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'), 
            logger
        )
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk split processing updates."""
        self._progress_callback = callback
        # FIXED: Create clean batch callback tanpa parameter conflicts
        self.batch_processor.register_progress_callback(self._create_clean_batch_callback())
    
    def process_split_dataset(self, split: str, processing_config: Dict[str, Any], 
                            force_reprocess: bool = False) -> Dict[str, Any]:
        """Process single split dataset dengan comprehensive handling."""
        start_time = time.time()
        
        try:
            # Phase 1: Setup dan validasi (0-15%)
            self._notify_split_progress(5, f"Setup processing untuk split {split}")
            
            setup_result = self._setup_split_processing(split, processing_config, force_reprocess)
            if not setup_result['success']:
                return {'success': False, 'error': setup_result['message'], 'processed': 0, 'failed': 1}
            
            source_info = setup_result['source_info']
            target_info = setup_result['target_info']
            
            # Check jika sudah diproses dan skip jika tidak force
            if not force_reprocess and self._is_already_processed(split):
                processed_count = self._count_processed_files(target_info['images_dir'])
                self._notify_split_progress(100, f"Split {split} sudah diproses ({processed_count} gambar)")
                return {'success': True, 'processed': processed_count, 'skipped': processed_count, 'failed': 0}
            
            self._notify_split_progress(15, f"Validasi selesai: {source_info['image_count']} gambar ditemukan")
            
            # Phase 2: Batch processing (15-85%)
            self._notify_split_progress(20, f"Memulai batch processing split {split}")
            
            batch_result = self.batch_processor.process_image_batch(
                source_info['images_dir'], source_info['labels_dir'],
                target_info['images_dir'], target_info['labels_dir'],
                processing_config
            )
            
            if not batch_result['success']:
                return {'success': False, 'error': batch_result['message'], 'processed': 0, 'failed': 1}
            
            # Phase 3: Finalization (85-100%)
            self._notify_split_progress(90, f"Finalisasi processing split {split}")
            
            final_stats = self._finalize_split_processing(split, batch_result, processing_config)
            final_stats['processing_time'] = time.time() - start_time
            
            self._notify_split_progress(100, 
                f"Split {split} selesai: {final_stats['processed']} gambar dalam {final_stats['processing_time']:.1f}s")
            
            self.logger.success(f"✅ Split {split} processing selesai: {final_stats['processed']} gambar")
            
            return final_stats
            
        except Exception as e:
            error_msg = f"Error processing split {split}: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'success': False, 'error': error_msg, 'processed': 0, 'failed': 1}
    
    def _setup_split_processing(self, split: str, processing_config: Dict[str, Any], 
                              force_reprocess: bool) -> Dict[str, Any]:
        """Setup split processing dengan path validation dan directory creation."""
        try:
            # Get dan validate source paths
            source_paths = self.path_manager.resolve_source_paths(split)
            if not source_paths['valid']:
                return {'success': False, 'message': f"Source paths invalid: {source_paths['message']}"}
            
            # Get dan setup target paths
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
        """Check apakah split sudah diproses sebelumnya."""
        split_path = self.storage.get_split_path(split)
        images_path = split_path / 'images'
        
        if not images_path.exists():
            return False
        
        processed_count = len(list(images_path.glob('*.*')))
        return processed_count > 0
    
    def _count_processed_files(self, target_dir: Path) -> int:
        """Count processed files di target directory."""
        if not target_dir.exists():
            return 0
        return len([f for f in target_dir.glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
    
    def _finalize_split_processing(self, split: str, batch_result: Dict[str, Any], 
                                 processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize split processing dengan stats collection dan metadata saving."""
        # Prepare final statistics
        final_stats = {
            'success': batch_result['success'],
            'processed': batch_result.get('processed', 0),
            'skipped': batch_result.get('skipped', 0),
            'failed': batch_result.get('failed', 0),
            'batch_count': batch_result.get('batch_count', 0)
        }
        
        # Save processing stats
        processing_stats = {
            **final_stats,
            'config_used': processing_config,
            'end_time': time.time()
        }
        
        self.storage.update_stats(split, processing_stats)
        
        # Log completion stats
        if final_stats['processed'] > 0:
            self.logger.info(
                f"📊 Split {split} stats: {final_stats['processed']} processed, "
                f"{final_stats['skipped']} skipped, {final_stats['failed']} failed"
            )
        
        return final_stats
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Dapatkan status processor untuk monitoring."""
        return {
            'processor_ready': True,
            'batch_processor_status': self.batch_processor.get_batch_status(),
            'storage_ready': self.storage is not None,
            'progress_callback_registered': self._progress_callback is not None
        }
    
    def cleanup_processor_state(self) -> None:
        """Cleanup processor state dan resources."""
        self.batch_processor.cleanup_batch_state()
        self._progress_callback = None
        self.logger.debug("🧹 Split processor state cleaned up")
    
    def _create_clean_batch_callback(self) -> Callable:
        """FIXED: Create clean batch callback tanpa parameter conflicts."""
        def clean_batch_callback(batch_progress, batch_message):
            """Clean callback dengan explicit positional parameters."""
            if self._progress_callback:
                try:
                    # Map batch progress ke split progress (15-85% range)
                    mapped_progress = 15 + (batch_progress / 100) * 70
                    
                    # FIXED: Call dengan clean parameters
                    self._progress_callback(
                        progress=int(mapped_progress),
                        message=batch_message or 'Processing batch'
                    )
                except Exception as e:
                    self.logger.debug(f"🔧 Batch callback error: {str(e)}")
        
        return clean_batch_callback
    
    def _notify_split_progress(self, progress: int, message: str, **kwargs):
        """Internal progress notification untuk split processing."""
        if self._progress_callback:
            try:
                # FIXED: Clean parameter passing
                self._progress_callback(progress=progress, message=message)
            except Exception as e:
                self.logger.debug(f"🔧 Split progress callback error: {str(e)}")