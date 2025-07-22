"""
File: smartcash/common/threadpools.py
Deskripsi: Thread pool utilities untuk optimal performance dengan one-liner style
"""

from typing import Optional, List, Dict, Any, Callable

# Import centralized worker utilities
from smartcash.common.worker_utils import (
    get_optimal_worker_count as get_optimal_thread_count,
    get_file_operation_workers,
    get_download_workers,
    get_rename_workers,
    optimal_io_workers,
    optimal_cpu_workers,
    safe_worker_count
)

# Ekspos fungsi-fungsi untuk diimpor
__all__ = [
    'get_optimal_thread_count', 'get_file_operation_workers',
    'get_download_workers', 'get_rename_workers',
    'process_in_parallel', 'process_with_stats',
    'optimal_io_workers', 'optimal_cpu_workers', 'safe_worker_count'
]

# Worker count functions are now imported from worker_utils.py

def process_in_parallel(items: List[Any], process_func: Callable[[Any], Any], max_workers: Optional[int] = None, desc: Optional[str] = None, show_progress: bool = True) -> List[Any]:
    """Proses items secara parallel menggunakan ThreadPoolExecutor"""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm
    
    # One-liner untuk menentukan jumlah worker
    max_workers = max_workers or get_optimal_thread_count('io')
    
    # One-liner untuk progress bar
    pbar = tqdm(total=len(items), desc=desc) if show_progress else None
    results = []
    
    # Proses items secara parallel dengan one-liner untuk submit
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, item) for item in items]
        
        # One-liner untuk mengumpulkan hasil dan update progress
        [results.append(future.result()) or (pbar and pbar.update(1)) for future in futures]
    
    # One-liner untuk menutup progress bar
    pbar and pbar.close()
    
    return results

def process_with_stats(
    items: List[Any],
    process_func: Callable[[Any], Dict[str, int]],
    max_workers: Optional[int] = None,
    desc: Optional[str] = None,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict[str, int]:
    """Process items in parallel and collect statistics with progress tracking.
    
    Args:
        items: List of items to process
        process_func: Function to process each item, should return a dict of stats
        max_workers: Maximum number of worker threads (default: optimal for I/O)
        desc: Description for progress display
        show_progress: Whether to show tqdm progress bar (ignored if progress_callback is provided)
        progress_callback: Optional callback function(current: int, total: int) for progress updates
        
    Returns:
        Dictionary of accumulated statistics
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm.auto import tqdm
    from collections import defaultdict
    
    total = len(items)
    if total == 0:
        return {}
        
    # Set up progress tracking
    use_tqdm = show_progress and progress_callback is None
    pbar = tqdm(total=total, desc=desc) if use_tqdm else None
    
    # Initialize statistics
    stats = defaultdict(int)
    
    # Process items in parallel
    max_workers = max_workers or get_optimal_thread_count('io')
    completed = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
        
        for future in as_completed(futures):
            try:
                result = future.result()
                # Update statistics
                if isinstance(result, dict):
                    for k, v in result.items():
                        stats[k] += v
                
                # Update progress
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
                elif pbar:
                    pbar.update(1)
                    
            except Exception as e:
                # Update error count but continue processing other items
                stats['errors'] = stats.get('errors', 0) + 1
    
    # Clean up progress bar if used
    if pbar:
        pbar.close()
    
    return dict(stats)
# Convenience functions are now imported from worker_utils.py