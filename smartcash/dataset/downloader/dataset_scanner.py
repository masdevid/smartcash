"""
File: smartcash/dataset/downloader/dataset_scanner.py
Deskripsi: UPDATED dataset scanner yang menggunakan config workers
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from smartcash.dataset.downloader.base import BaseDownloaderComponent
from smartcash.common.environment import get_environment_manager

class DatasetScanner(BaseDownloaderComponent):
    """Dataset scanner dengan config-aware workers"""
    
    def __init__(self, logger=None, max_workers: int = None):
        super().__init__(logger)
        self.env_manager = get_environment_manager()
        self.dataset_path = self.env_manager.get_dataset_path()
        self.img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        # Use provided max_workers or get optimal count
        if max_workers is None:
            from smartcash.common.threadpools import get_optimal_thread_count
            max_workers = get_optimal_thread_count('io')
        
        self.max_workers = max_workers
        self.logger.info(f"🔍 DatasetScanner initialized with {self.max_workers} workers")
    
    def _scan_splits_directories(self) -> Dict[str, Any]:
        """Scan splits dengan optimal parallel processing"""
        splits = ['train', 'valid', 'test']
        split_results = {}
        
        # Use optimal workers untuk parallel scanning
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(splits))) as executor:
            future_to_split = {
                executor.submit(self._scan_single_split, split): split 
                for split in splits
            }
            
            for future in future_to_split:
                split_name = future_to_split[future]
                try:
                    split_results[split_name] = future.result()
                except Exception as e:
                    split_results[split_name] = {
                        'status': 'error', 'message': str(e),
                        'images': 0, 'labels': 0
                    }
        
        return split_results
    
    def scan_existing_dataset_parallel(self) -> Dict[str, Any]:
        """Enhanced scan dengan parallel file counting"""
        self._notify_progress("scan_start", 0, 100, f"🔍 Starting parallel scan with {self.max_workers} workers...")
        
        try:
            if not self.dataset_path.exists():
                self._notify_progress("scan_warning", 0, 100, f"⚠️ Dataset path tidak ditemukan: {self.dataset_path}")
                return self._create_empty_scan_result("Dataset directory tidak ditemukan")
            
            self._notify_progress("scan_structure", 20, 100, "🔍 Analyzing dataset structure...")
            structure = self._analyze_dataset_structure()
            
            self._notify_progress("scan_downloads", 40, 100, "📂 Scanning downloads directory...")
            downloads_result = self._scan_downloads_directory_parallel()
            
            self._notify_progress("scan_splits", 60, 100, "📊 Scanning splits directories...")
            splits_result = self._scan_splits_directories()
            
            self._notify_progress("scan_aggregate", 80, 100, "📈 Aggregating statistics...")
            
            result = {
                'status': 'success',
                'dataset_path': str(self.dataset_path),
                'structure': structure,
                'downloads': downloads_result,
                'splits': splits_result,
                'summary': self._create_summary(downloads_result, splits_result),
                'scan_time': self._get_scan_timestamp(),
                'workers_used': self.max_workers
            }
            
            self._notify_progress("scan_complete", 100, 100, f"✅ Parallel scan complete: {result['summary']['total_images']} images")
            return result
            
        except Exception as e:
            error_msg = f"Error saat parallel scanning: {str(e)}"
            self._notify_progress("scan_error", 0, 100, f"❌ {error_msg}")
            return self._create_error_result(error_msg)
    
    def _scan_downloads_directory_parallel(self) -> Dict[str, Any]:
        """Parallel scan downloads directory"""
        downloads_dir = self.dataset_path / 'downloads'
        
        if not downloads_dir.exists():
            return {'status': 'not_found', 'file_count': 0, 'total_size': 0}
        
        try:
            # Get all files first
            all_files = list(downloads_dir.rglob('*'))
            file_chunks = [all_files[i:i + 100] for i in range(0, len(all_files), 100)]
            
            file_stats = {}
            total_files = 0
            total_size = 0
            
            # Process chunks in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._count_file_chunk, chunk)
                    for chunk in file_chunks
                ]
                
                for future in futures:
                    try:
                        chunk_stats, chunk_files, chunk_size = future.result()
                        
                        # Merge chunk stats
                        for ext, stats in chunk_stats.items():
                            if ext not in file_stats:
                                file_stats[ext] = {'count': 0, 'size': 0}
                            file_stats[ext]['count'] += stats['count']
                            file_stats[ext]['size'] += stats['size']
                        
                        total_files += chunk_files
                        total_size += chunk_size
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error processing file chunk: {str(e)}")
            
            return {
                'status': 'success',
                'file_count': total_files,
                'total_size': total_size,
                'size_formatted': self._format_file_size(total_size),
                'file_types': file_stats
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'file_count': 0, 'total_size': 0}
    
    def _count_file_chunk(self, file_chunk: List[Path]) -> Tuple[Dict, int, int]:
        """Count files dalam chunk untuk parallel processing"""
        chunk_stats = {}
        chunk_files = 0
        chunk_size = 0
        
        for file_path in file_chunk:
            if file_path.is_file():
                try:
                    ext = file_path.suffix.lower()
                    size = file_path.stat().st_size
                    
                    if ext not in chunk_stats:
                        chunk_stats[ext] = {'count': 0, 'size': 0}
                    
                    chunk_stats[ext]['count'] += 1
                    chunk_stats[ext]['size'] += size
                    chunk_files += 1
                    chunk_size += size
                    
                except Exception:
                    continue  # Skip files yang tidak bisa diakses
        
        return chunk_stats, chunk_files, chunk_size


def create_dataset_scanner(logger=None, max_workers: int = None) -> DatasetScanner:
    """Factory dengan optimal workers support"""
    return DatasetScanner(logger, max_workers)