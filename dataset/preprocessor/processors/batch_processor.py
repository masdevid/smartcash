"""
File: smartcash/dataset/preprocessor/processors/batch_processor.py
Deskripsi: Fixed batch processor dengan debug untuk zero processing issue
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count
from smartcash.dataset.preprocessor.processors.image_processor import ImageProcessor


class BatchProcessor:
    """Fixed batch processor dengan debug logging untuk zero processing issue."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize batch processor dengan threading optimization."""
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        
        # Initialize image processor
        self.image_processor = ImageProcessor(config, logger)
        
        # Threading configuration
        self.max_workers = min(
            config.get('preprocessing', {}).get('num_workers', 4),
            get_optimal_thread_count()
        )
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register progress callback untuk batch updates."""
        self._progress_callback = callback
    
    def process_image_batch(self, source_images_dir: Path, source_labels_dir: Path,
                          target_images_dir: Path, target_labels_dir: Path,
                          processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fixed process batch dengan comprehensive debugging."""
        start_time = time.time()
        
        try:
            # DEBUG: Log paths untuk validation
            self.logger.info(f"🔍 Source images: {source_images_dir} (exists: {source_images_dir.exists()})")
            self.logger.info(f"🔍 Source labels: {source_labels_dir} (exists: {source_labels_dir.exists()})")
            self.logger.info(f"🔍 Target images: {target_images_dir}")
            self.logger.info(f"🔍 Target labels: {target_labels_dir}")
            
            # Get image files untuk processing
            image_files = self._get_image_files(source_images_dir)
            if not image_files:
                self.logger.warning(f"⚠️ No image files found in {source_images_dir}")
                return {'success': True, 'processed': 0, 'skipped': 0, 'failed': 0, 'message': 'No images found'}
            
            total_images = len(image_files)
            self.logger.info(f"📊 Found {total_images} image files to process")
            self._notify_batch_progress(0, f"Starting batch processing: {total_images} images")
            
            # DEBUG: Check existing files untuk force_reprocess logic
            force_reprocess = processing_config.get('force_reprocess', False)
            self.logger.info(f"🔄 Force reprocess: {force_reprocess}")
            
            # Create batches untuk optimized processing
            batches = self._create_processing_batches(image_files, batch_size=50)
            self.logger.info(f"📦 Created {len(batches)} batches for processing")
            batch_results = []
            
            # Process batches dengan parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="BatchProc") as executor:
                # Submit batch processing tasks
                future_to_batch = {}
                for i, batch in enumerate(batches):
                    self.logger.debug(f"📤 Submitting batch {i+1}/{len(batches)} with {len(batch)} images")
                    future = executor.submit(
                        self._process_single_batch,
                        batch, i, len(batches), source_labels_dir, 
                        target_images_dir, target_labels_dir, processing_config
                    )
                    future_to_batch[future] = i
                
                # Collect results as completed
                completed_batches = 0
                for future in as_completed(future_to_batch):
                    batch_index = future_to_batch[future]
                    
                    try:
                        batch_result = future.result()
                        batch_results.append(batch_result)
                        completed_batches += 1
                        
                        # DEBUG: Log batch results
                        self.logger.info(f"✅ Batch {completed_batches}/{len(batches)} completed: {batch_result}")
                        
                        # Update progress
                        progress = int((completed_batches / len(batches)) * 100)
                        processed_count = sum(r.get('processed', 0) for r in batch_results)
                        
                        self._notify_batch_progress(
                            progress, 
                            f"Batch {completed_batches}/{len(batches)}: {processed_count}/{total_images} images processed"
                        )
                        
                    except Exception as e:
                        self.logger.error(f"❌ Batch {batch_index} error: {str(e)}")
                        batch_results.append({'processed': 0, 'skipped': 0, 'failed': len(batches[batch_index])})
            
            # Aggregate final results
            final_result = self._aggregate_batch_results(batch_results, time.time() - start_time)
            
            # DEBUG: Log final aggregated results
            self.logger.info(f"📊 Final batch results: {final_result}")
            
            self.logger.success(
                f"✅ Batch processing selesai: {final_result['processed']}/{total_images} images, "
                f"{final_result['processing_time']:.1f}s"
            )
            
            return final_result
            
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'success': False, 'message': error_msg, 'processed': 0, 'failed': 1}
    
    def _get_image_files(self, images_dir: Path) -> List[Path]:
        """Get list image files dari directory dengan debug logging."""
        if not images_dir.exists():
            self.logger.warning(f"⚠️ Images directory does not exist: {images_dir}")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        image_files = [f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
        
        self.logger.debug(f"🔍 Scanning {images_dir}: found {len(image_files)} image files")
        if len(image_files) == 0:
            # DEBUG: List all files untuk diagnosis
            all_files = list(images_dir.glob('*.*'))
            self.logger.debug(f"🔍 All files in directory: {[f.name for f in all_files[:10]]}")
        
        return image_files
    
    def _create_processing_batches(self, image_files: List[Path], batch_size: int = 50) -> List[List[Path]]:
        """Create batches untuk optimized parallel processing."""
        batches = []
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    def _process_single_batch(self, batch: List[Path], batch_index: int, total_batches: int,
                            source_labels_dir: Path, target_images_dir: Path, 
                            target_labels_dir: Path, processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Fixed process single batch dengan detailed logging."""
        batch_stats = {'processed': 0, 'skipped': 0, 'failed': 0}
        
        self.logger.debug(f"🔧 Processing batch {batch_index+1}/{total_batches} with {len(batch)} images")
        
        # FIXED: Anti-flood debug - log sample files only
        sample_logging = len(batch) > 10
        
        for i, image_path in enumerate(batch):
            try:
                # Process single image menggunakan image processor
                result = self.image_processor.process_single_image(
                    image_path, source_labels_dir, target_images_dir, 
                    target_labels_dir, processing_config
                )
                
                # Anti-flood: Log only first/last few samples
                if not sample_logging or i < 3 or i >= len(batch) - 2:
                    self.logger.debug(f"🖼️ Image {image_path.name}: {result.get('status', 'failed')}")
                
                if result['success']:
                    if result.get('status') == 'processed':
                        batch_stats['processed'] += 1
                    else:
                        batch_stats['skipped'] += 1
                else:
                    batch_stats['failed'] += 1
                    # Only log failures untuk important debugging
                    if not sample_logging or batch_stats['failed'] <= 5:
                        self.logger.warning(f"⚠️ Failed {image_path.name}: {result.get('message')}")
                    
            except Exception as e:
                batch_stats['failed'] += 1
                # Only log first few errors untuk prevent flood
                if batch_stats['failed'] <= 5:
                    self.logger.error(f"💥 Error {image_path.name}: {str(e)}")
        
        self.logger.info(f"📊 Batch {batch_index+1} stats: {batch_stats}")
        return batch_stats
    
    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]], 
                               processing_time: float) -> Dict[str, Any]:
        """Aggregate results dari semua batches."""
        aggregated = {
            'success': True,
            'processed': sum(r.get('processed', 0) for r in batch_results),
            'skipped': sum(r.get('skipped', 0) for r in batch_results),
            'failed': sum(r.get('failed', 0) for r in batch_results),
            'batch_count': len(batch_results),
            'processing_time': processing_time
        }
        
        # Calculate success rate
        total_attempted = aggregated['processed'] + aggregated['skipped'] + aggregated['failed']
        if total_attempted > 0:
            aggregated['success_rate'] = (aggregated['processed'] / total_attempted) * 100
        else:
            aggregated['success_rate'] = 0
        
        return aggregated
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Dapatkan status batch processor."""
        return {
            'processor_ready': True,
            'max_workers': self.max_workers,
            'image_processor_ready': self.image_processor is not None,
            'progress_callback_registered': self._progress_callback is not None
        }
    
    def cleanup_batch_state(self) -> None:
        """Cleanup batch processor state."""
        self.image_processor.cleanup_processor_state()
        self._progress_callback = None
        self.logger.debug("🧹 Batch processor state cleaned up")
    
    def _notify_batch_progress(self, progress: int, message: str, **kwargs):
        """Internal progress notification untuk batch processing."""
        if self._progress_callback:
            try:
                # FIXED: Call dengan positional args untuk avoid keyword conflicts
                self._progress_callback(progress, message)
            except Exception as e:
                self.logger.debug(f"🔧 Batch progress callback error: {str(e)}")