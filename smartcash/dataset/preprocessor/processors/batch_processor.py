"""
File: smartcash/dataset/preprocessor/processors/batch_processor.py
Deskripsi: Batch processor yang terintegrasi dengan dataset_file_renamer tanpa duplikasi
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count
from smartcash.dataset.services.dataset_file_renamer import create_dataset_renamer

class BatchProcessor:
    """Batch processor dengan renamer integration untuk UUID consistency"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.logger = logger or get_logger()
        self._progress_callback: Optional[Callable] = None
        self.renamer = create_dataset_renamer(config)
        self.max_workers = min(config.get('preprocessing', {}).get('num_workers', 4), get_optimal_thread_count())
        
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback dengan simple parameter handling"""
        self._progress_callback = callback
    
    def process_image_batch_with_renaming(self, source_images_dir: Path, source_labels_dir: Path,
                                        target_images_dir: Path, target_labels_dir: Path,
                                        processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Batch processing dengan pre-renaming untuk UUID consistency"""
        start_time = time.time()
        
        try:
            # Phase 1: Pre-rename untuk UUID consistency (0-20%)
            self._notify_progress(10, "🔄 Pre-renaming files untuk UUID consistency")
            rename_result = self._pre_rename_source_files(source_images_dir.parent)
            
            if not rename_result['success']:
                self.logger.warning(f"⚠️ Pre-rename warning: {rename_result['message']}")
            
            # Phase 2: Process batch dengan UUID consistent files (20-100%) 
            return self._process_batch_with_uuid_consistency(
                source_images_dir, source_labels_dir, target_images_dir, 
                target_labels_dir, processing_config, start_time
            )
            
        except Exception as e:
            return {'success': False, 'message': f'Batch processing error: {str(e)}', 'processed': 0}
    
    def _pre_rename_source_files(self, source_dir: Path) -> Dict[str, Any]:
        """Pre-rename source files untuk UUID consistency"""
        try:
            preview = self.renamer.get_rename_preview(str(source_dir), limit=5)
            if preview['status'] == 'success' and preview['total_files'] == 0:
                return {'success': True, 'message': 'Files already UUID consistent', 'renamed': 0}
            
            # Execute rename dengan progress mapping
            def rename_progress(p, m):
                self._notify_progress(p // 5, f"Renaming: {m}")
            
            rename_result = self.renamer.batch_rename_dataset(
                str(source_dir), backup=False, progress_callback=rename_progress
            )
            
            return {
                'success': rename_result.get('status') == 'success',
                'message': rename_result.get('message', 'Rename completed'),
                'renamed': rename_result.get('renamed_files', 0)
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Pre-rename error: {str(e)}'}
    
    def _process_batch_with_uuid_consistency(self, source_images_dir: Path, source_labels_dir: Path,
                                           target_images_dir: Path, target_labels_dir: Path,
                                           processing_config: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Process batch dengan UUID consistency"""
        try:
            image_files = self._get_image_files(source_images_dir)
            if not image_files:
                return {'success': True, 'processed': 0, 'skipped': 0, 'failed': 0, 'message': 'No images'}
            
            total_images = len(image_files)
            self._notify_progress(25, f"📊 Processing {total_images} UUID consistent files")
            
            # Create batches
            batches = self._create_processing_batches(image_files, 50)
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="BatchProc") as executor:
                future_to_batch = {
                    executor.submit(self._process_single_batch, batch, i, len(batches), 
                                  source_labels_dir, target_images_dir, target_labels_dir, processing_config): i
                    for i, batch in enumerate(batches)
                }
                
                completed = 0
                for future in as_completed(future_to_batch):
                    result = future.result()
                    batch_results.append(result)
                    completed += 1
                    
                    progress = 25 + int((completed / len(batches)) * 70)
                    processed_count = sum(r.get('processed', 0) for r in batch_results)
                    self._notify_progress(progress, f"Batch {completed}/{len(batches)}: {processed_count} processed")
            
            # Aggregate results
            final_result = self._aggregate_batch_results(batch_results, time.time() - start_time)
            self.logger.success(f"✅ Batch processing: {final_result['processed']}/{total_images} in {final_result['processing_time']:.1f}s")
            
            return final_result
            
        except Exception as e:
            return {'success': False, 'message': f'UUID batch processing error: {str(e)}', 'processed': 0}
    
    def _process_single_batch(self, batch: List[Path], batch_index: int, total_batches: int,
                            source_labels_dir: Path, target_images_dir: Path, 
                            target_labels_dir: Path, processing_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process single batch dengan UUID awareness"""
        batch_stats = {'processed': 0, 'skipped': 0, 'failed': 0}
        
        # Import di sini untuk avoid circular import
        from smartcash.dataset.preprocessor.processors.image_processor import ImageProcessor
        processor = ImageProcessor(self.config, self.logger)
        
        for image_path in batch:
            try:
                result = processor.process_single_image(
                    image_path, source_labels_dir, target_images_dir, target_labels_dir, processing_config
                )
                
                if result['success']:
                    if result.get('status') == 'processed':
                        batch_stats['processed'] += 1
                    else:
                        batch_stats['skipped'] += 1
                else:
                    batch_stats['failed'] += 1
                    
            except Exception:
                batch_stats['failed'] += 1
        
        return batch_stats
    
    def _get_image_files(self, images_dir: Path) -> List[Path]:
        """Get image files dengan UUID awareness check"""
        if not images_dir.exists():
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = [f for f in images_dir.glob('*.*') if f.suffix.lower() in image_extensions]
        
        # Log UUID consistency status
        uuid_consistent = sum(1 for f in image_files[:10] if self.renamer.naming_manager.parse_existing_filename(f.name))
        if uuid_consistent > 0:
            self.logger.debug(f"🔍 UUID consistency: {uuid_consistent}/10 sample files")
        
        return image_files
    
    def _create_processing_batches(self, image_files: List[Path], batch_size: int = 50) -> List[List[Path]]:
        """Create batches untuk processing"""
        return [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
    
    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]], processing_time: float) -> Dict[str, Any]:
        """Aggregate results dari batch processing"""
        aggregated = {
            'success': True,
            'processed': sum(r.get('processed', 0) for r in batch_results),
            'skipped': sum(r.get('skipped', 0) for r in batch_results),
            'failed': sum(r.get('failed', 0) for r in batch_results),
            'batch_count': len(batch_results),
            'processing_time': processing_time
        }
        
        total = aggregated['processed'] + aggregated['skipped'] + aggregated['failed']
        aggregated['success_rate'] = (aggregated['processed'] / total) * 100 if total > 0 else 0
        
        return aggregated
    
    def get_batch_status(self) -> Dict[str, Any]:
        """Get batch processor status"""
        return {
            'processor_ready': True,
            'max_workers': self.max_workers,
            'renamer_integrated': self.renamer is not None,
            'uuid_registry_size': len(self.renamer.naming_manager.uuid_registry),
            'progress_callback_registered': self._progress_callback is not None
        }
    
    def cleanup_batch_state(self) -> None:
        """Cleanup batch processor state"""
        self._progress_callback = None
    
    def _notify_progress(self, progress: int, message: str):
        """Progress notification dengan simple args"""
        if self._progress_callback:
            try:
                self._progress_callback(progress, message)
            except Exception as e:
                self.logger.debug(f"🔧 Batch progress error: {str(e)}")

# Factory dan utilities
create_batch_processor = lambda config: BatchProcessor(config)
process_batch_with_renaming = lambda src_imgs, src_labels, tgt_imgs, tgt_labels, config: create_batch_processor(config).process_image_batch_with_renaming(Path(src_imgs), Path(src_labels), Path(tgt_imgs), Path(tgt_labels), config)