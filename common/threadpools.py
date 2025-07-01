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

def process_with_stats(items: List[Any], process_func: Callable[[Any], Dict[str, int]], max_workers: Optional[int] = None, desc: Optional[str] = None, show_progress: bool = True) -> Dict[str, int]:
    """Proses items secara parallel dan kumpulkan statistik"""
    from concurrent.futures import ThreadPoolExecutor
    from tqdm.auto import tqdm
    from collections import defaultdict
    
    # One-liner untuk menentukan jumlah worker
    max_workers = max_workers or get_optimal_thread_count('io')
    
    # One-liner untuk progress bar
    pbar = tqdm(total=len(items), desc=desc) if show_progress else None
    
    # Inisialisasi statistik
    stats = defaultdict(int)
    
    # Proses items secara parallel dengan one-liner untuk submit
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, item) for item in items]
        
        # One-liner untuk mengumpulkan statistik dan update progress
        for future in futures:
            result = future.result()
            # Update stats dengan one-liner yang benar
            if isinstance(result, dict):
                for k, v in result.items():
                    stats[k] += v
            # Update progress dengan one-liner
            pbar and pbar.update(1)
    
    # One-liner untuk menutup progress bar
    pbar and pbar.close()
    
    return dict(stats)
# Convenience functions are now imported from worker_utils.py