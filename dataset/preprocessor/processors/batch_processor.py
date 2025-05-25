"""
File: smartcash/dataset/preprocessor/processors/batch_processor.py
Deskripsi: Batch processor untuk parallel image processing dengan optimized threading
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count
from smartcash.dataset.preprocessor.processors.image_processor import ImageProcessor


class BatchProcessor:
    """Batch processor untuk parallel processing images dengan optimized performance."""
    
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
        """
        Process batch images dengan parallel execution dan progress tracking.
        
        Args:
            source_images_dir: Source directory images
            source_labels_dir: Source directory labels  
            target_images_dir: Target directory images
            target_labels_dir: Target directory labels
            processing_config: Configuration preprocessing
            
        Returns:
            Dictionary hasil batch processing
        """
        start_time = time.time()
        
        try:
            # Get image files untuk processing
            image_files = self._get_image_files(source_images_dir)
            if not image_files:
                return {'success': True, 'processed': 0, 'skipped': 0, 'failed': 0, 'message': 'No images found'}
            
            total_images = len(image_files)
            self._notify_batch_progress(0, f"Starting batch processing: {total_images} images")
            
            # Create batches untuk optimized processing
            batches = self._create_processing_batches(image_files, batch_size=50)
            batch_results = []
            
            # Process batches dengan parallel execution
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="BatchProc") as executor:
                # Submit batch processing tasks
                future_to_batch = {}
                for i, batch in enumerate(batches):
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
        """Get list image files dari directory."""
        if not images_dir.exists():
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        return [f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
    
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
        """Process single batch images dengan error handling per image."""
        batch_stats = {'processed': 0, 'skipped': 0, 'failed': 0}
        
        for image_path in batch:
            try:
                # Process single image menggunakan image processor
                result = self.image_processor.process_single_image(
                    image_path, source_labels_dir, target_images_dir, 
                    target_labels_dir, processing_config
                )
                
                if result['success']:
                    batch_stats['processed'] += 1
                else:
                    batch_stats['skipped'] += 1
                    
            except Exception as e:
                self.logger.debug(f"🔧 Image processing error {image_path.name}: {str(e)}")
                batch_stats['failed'] += 1
        
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
                self._progress_callback(progress=progress, message=message, **kwargs)
            except Exception as e:
                self.logger.debug(f"🔧 Batch progress callback error: {str(e)}")