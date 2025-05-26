"""
File: smartcash/dataset/augmentor/processors/batch.py
Deskripsi: Batch processing operations dengan sync/async support untuk augmentasi parallel dan sequential
"""

import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional, Union
from functools import partial
from smartcash.common.logger import get_logger
from .image import ImageProcessor
from .bbox import BBoxProcessor
from .file import FileProcessor

# One-liner helper functions
get_optimal_workers = lambda: min(8, (os.cpu_count() or 4))
get_io_workers = lambda: min(16, (os.cpu_count() or 4) * 2)
chunk_list = lambda lst, n: [lst[i:i + n] for i in range(0, len(lst), n)]
flatten_results = lambda results: [item for sublist in results for item in (sublist if isinstance(sublist, list) else [sublist])]

def process_files_sync(
    files: List[str], 
    process_func: Callable, 
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Process files synchronously dengan progress callback."""
    results = []
    for i, file in enumerate(files):
        if progress_callback:
            progress_callback("processing", i, len(files), f"Processing {i+1}/{len(files)}")
        results.append(process_func(file, **kwargs))
    return results

def process_files_async(
    files: List[str], 
    process_func: Callable, 
    max_workers: int = None,
    executor_type: str = "thread",
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Process files asynchronously dengan executor selection."""
    max_workers = max_workers or (get_io_workers() if executor_type == "thread" else get_optimal_workers())
    executor_class = ThreadPoolExecutor if executor_type == "thread" else ProcessPoolExecutor
    
    results = []
    completed = 0
    
    with executor_class(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, file, **kwargs) for file in files]
        
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            if progress_callback:
                progress_callback("processing", completed, len(files), f"Completed {completed}/{len(files)}")
    
    return results

class BatchProcessor:
    """Processor untuk batch operations dengan optimized parallel/sequential processing."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi BatchProcessor.
        
        Args:
            logger: Logger untuk logging operations
        """
        self.logger = logger or get_logger(__name__)
        self.image_processor = ImageProcessor(logger)
        self.bbox_processor = BBoxProcessor(logger)
        self.file_processor = FileProcessor(logger)
        
        self.processed_batches = 0
        self.total_files_processed = 0
        self.processing_time = 0.0
    
    def process_augmentation_batch(
        self, 
        image_files: List[str],
        pipeline,
        labels_dir: str = None,
        output_images_dir: str = None,
        output_labels_dir: str = None,
        variations: int = 2,
        output_prefix: str = "aug",
        parallel: bool = False,
        max_workers: int = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Process batch augmentasi dengan pilihan parallel atau sequential.
        
        Args:
            image_files: List file gambar
            pipeline: Pipeline augmentasi
            labels_dir: Directory label input
            output_images_dir: Directory output gambar
            output_labels_dir: Directory output label
            variations: Jumlah variasi per gambar
            output_prefix: Prefix untuk output files
            parallel: Gunakan parallel processing
            max_workers: Jumlah workers untuk parallel
            progress_callback: Callback untuk progress updates
            
        Returns:
            List hasil processing per file
        """
        start_time = time.time()
        
        # Setup progress callback
        def report_progress(current: int, total: int, message: str = ""):
            if progress_callback:
                progress_callback("augmentation", current, total, message)
        
        # Ensure output directories
        if output_images_dir:
            self.file_processor.ensure_dir(output_images_dir)
        if output_labels_dir:
            self.file_processor.ensure_dir(output_labels_dir)
        
        # Setup processing function
        process_func = partial(
            self._process_single_file_augmentation,
            pipeline=pipeline,
            labels_dir=labels_dir,
            output_images_dir=output_images_dir,
            output_labels_dir=output_labels_dir,
            variations=variations,
            output_prefix=output_prefix
        )
        
        # Process berdasarkan mode
        if parallel and len(image_files) > 1:
            self.logger.info(f"🚀 Parallel augmentasi {len(image_files)} files dengan {max_workers or get_optimal_workers()} workers")
            results = self._process_parallel_with_progress(
                image_files, process_func, max_workers or get_optimal_workers(), report_progress
            )
        else:
            self.logger.info(f"🔄 Sequential augmentasi {len(image_files)} files")
            results = self._process_sequential_with_progress(
                image_files, process_func, report_progress
            )
        
        # Update statistics
        processing_time = time.time() - start_time
        self.processed_batches += 1
        self.total_files_processed += len(image_files)
        self.processing_time += processing_time
        
        # Calculate success rate
        successful_results = [r for r in results if r.get('status') == 'success']
        success_rate = len(successful_results) / len(results) * 100 if results else 0
        
        self.logger.info(f"✅ Batch selesai: {len(successful_results)}/{len(results)} berhasil ({success_rate:.1f}%) dalam {processing_time:.2f}s")
        
        return results
    
    def _process_single_file_augmentation(
        self,
        image_file: str,
        pipeline,
        labels_dir: str = None,
        output_images_dir: str = None,
        output_labels_dir: str = None,
        variations: int = 2,
        output_prefix: str = "aug"
    ) -> Dict[str, Any]:
        """
        Process single file untuk augmentasi dengan image dan bbox.
        
        Args:
            image_file: Path file gambar
            pipeline: Pipeline augmentasi
            labels_dir: Directory label input
            output_images_dir: Directory output gambar
            output_labels_dir: Directory output label
            variations: Jumlah variasi
            output_prefix: Prefix output
            
        Returns:
            Dictionary hasil processing
        """
        try:
            from pathlib import Path
            
            image_stem = Path(image_file).stem
            label_file = str(Path(labels_dir) / f"{image_stem}.txt") if labels_dir else None
            
            # Process gambar
            image_results = self.image_processor.process_single_image(
                image_file, pipeline, 
                str(Path(output_images_dir) / output_prefix) if output_images_dir else f"{output_prefix}_{image_stem}",
                variations
            )
            
            # Process bounding boxes jika ada
            bbox_results = []
            if label_file and output_labels_dir and Path(label_file).exists():
                original_bboxes = self.bbox_processor.read_yolo_labels(label_file)
                
                if original_bboxes:
                    for var_idx in range(variations):
                        output_label_path = str(Path(output_labels_dir) / f"{output_prefix}_{image_stem}_var{var_idx+1}.txt")
                        
                        # Untuk augmentasi sederhana, gunakan bbox original
                        # TODO: Implement proper bbox transformation berdasarkan pipeline transforms
                        save_success = self.bbox_processor.save_yolo_labels(original_bboxes, output_label_path)
                        
                        bbox_results.append({
                            'variation': var_idx + 1,
                            'output_path': output_label_path if save_success else None,
                            'bbox_count': len(original_bboxes),
                            'status': 'success' if save_success else 'error'
                        })
            
            # Combine results
            successful_images = sum(1 for r in image_results if r.get('status') == 'success')
            successful_bboxes = sum(1 for r in bbox_results if r.get('status') == 'success')
            
            return {
                'status': 'success' if successful_images > 0 else 'error',
                'image_file': image_file,
                'image_stem': image_stem,
                'variations_created': successful_images,
                'bboxes_processed': successful_bboxes,
                'image_results': image_results,
                'bbox_results': bbox_results,
                'generated': successful_images  # For compatibility
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'image_file': image_file,
                'error': str(e),
                'generated': 0
            }
    
    def _process_sequential_with_progress(
        self,
        files: List[str],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Process files sequential dengan progress reporting via callback only."""
        results = []
        
        for i, file in enumerate(files):
            result = process_func(file)
            results.append(result)
            
            # Report progress via callback only
            if progress_callback:
                progress_callback(i + 1, len(files), f"Processing {i+1}/{len(files)}")
        
        return results
    
    def _process_parallel_with_progress(
        self,
        files: List[str],
        process_func: Callable,
        max_workers: int,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Process files parallel dengan progress reporting via callback only."""
        results = []
        completed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(process_func, file): i for i, file in enumerate(files)}
            
            # Process results as they complete
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append((futures[future], result))  # Keep original order
                except Exception as e:
                    file_idx = futures[future]
                    results.append((file_idx, {
                        'status': 'error',
                        'image_file': files[file_idx],
                        'error': str(e),
                        'generated': 0
                    }))
                
                completed += 1
                
                # Report progress via callback only
                if progress_callback:
                    progress_callback(completed, len(files), f"Completed {completed}/{len(files)}")
        
        # Sort results by original order dan extract result only
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def batch_copy_augmented_files(
        self,
        source_dir: str,
        target_dir: str,
        file_prefix: str = "aug",
        splits: List[str] = ['train'],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Batch copy hasil augmentasi ke target directory dengan structure.
        
        Args:
            source_dir: Source directory (augmented results)
            target_dir: Target directory (preprocessed structure)
            file_prefix: Prefix file yang akan di-copy
            splits: List splits untuk di-copy
            parallel: Gunakan parallel processing untuk copy
            
        Returns:
            Dictionary hasil batch copy
        """
        from .file import find_files
        
        copy_results = {'splits': {}, 'total_copied': 0, 'errors': []}
        
        for split in splits:
            # Find augmented files
            aug_pattern = f"{file_prefix}_*.*"
            source_files = find_files(source_dir, aug_pattern, recursive=True)
            
            if not source_files:
                copy_results['splits'][split] = {'copied': 0, 'message': 'No augmented files found'}
                continue
            
            # Prepare copy mappings
            copy_mappings = []
            for source_file in source_files:
                from pathlib import Path
                
                file_path = Path(source_file)
                file_ext = file_path.suffix
                
                # Determine target subdirectory
                if file_ext.lower() in ['.jpg', '.jpeg', '.png']:
                    target_subdir = 'images'
                elif file_ext.lower() == '.txt':
                    target_subdir = 'labels'
                else:
                    continue
                
                target_file = Path(target_dir) / split / target_subdir / file_path.name
                copy_mappings.append({'src': str(file_path), 'dst': str(target_file)})
            
            # Execute copy
            if parallel and len(copy_mappings) > 10:
                copy_result = self._batch_copy_parallel(copy_mappings)
            else:
                copy_result = self.file_processor.batch_copy(copy_mappings)
            
            copy_results['splits'][split] = {
                'copied': len(copy_result['success']),
                'failed': len(copy_result['failed']),
                'skipped': len(copy_result['skipped'])
            }
            copy_results['total_copied'] += len(copy_result['success'])
            
            if copy_result['failed']:
                copy_results['errors'].extend(copy_result['failed'])
        
        self.logger.info(f"📦 Batch copy completed: {copy_results['total_copied']} files ke {len(splits)} splits")
        
        return copy_results
    
    def _batch_copy_parallel(self, copy_mappings: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute batch copy dengan parallel processing."""
        # Chunk mappings untuk parallel processing
        chunk_size = max(1, len(copy_mappings) // get_io_workers())
        chunks = chunk_list(copy_mappings, chunk_size)
        
        all_results = {'success': [], 'failed': [], 'skipped': []}
        
        with ThreadPoolExecutor(max_workers=get_io_workers()) as executor:
            futures = [executor.submit(self.file_processor.batch_copy, chunk) for chunk in chunks]
            
            for future in as_completed(futures):
                chunk_result = future.result()
                all_results['success'].extend(chunk_result['success'])
                all_results['failed'].extend(chunk_result['failed'])
                all_results['skipped'].extend(chunk_result['skipped'])
        
        return all_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik batch processing.
        
        Returns:
            Dictionary statistik
        """
        avg_time_per_batch = self.processing_time / self.processed_batches if self.processed_batches > 0 else 0
        avg_files_per_second = self.total_files_processed / self.processing_time if self.processing_time > 0 else 0
        
        return {
            'processed_batches': self.processed_batches,
            'total_files_processed': self.total_files_processed,
            'total_processing_time': round(self.processing_time, 2),
            'avg_time_per_batch': round(avg_time_per_batch, 2),
            'avg_files_per_second': round(avg_files_per_second, 2),
            'image_stats': self.image_processor.get_processing_stats(),
            'bbox_stats': self.bbox_processor.get_processing_stats(),
            'file_stats': self.file_processor.get_processing_stats()
        }
    
    def reset_stats(self) -> None:
        """Reset semua counter statistik."""
        self.processed_batches = 0
        self.total_files_processed = 0
        self.processing_time = 0.0
        
        self.image_processor.reset_stats()
        self.bbox_processor.reset_stats()
        self.file_processor.reset_stats()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BatchProcessor(batches={self.processed_batches}, files={self.total_files_processed}, time={self.processing_time:.2f}s)"