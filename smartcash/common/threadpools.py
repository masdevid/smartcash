"""
File: smartcash/common/threadpools.py
Deskripsi: Thread pool utilities untuk optimal performance dengan one-liner style
"""

import os
from typing import Optional

def get_optimal_thread_count(operation_type: str = 'io') -> int:
    """
    Get optimal thread count berdasarkan operation type dan system specs.
    
    Args:
        operation_type: 'io' untuk I/O bound, 'cpu' untuk CPU bound
        
    Returns:
        Optimal thread count
    """
    cpu_count = os.cpu_count() or 1
    
    if operation_type == 'io':
        # I/O bound: CPU count + 1, max 8 untuk avoid overhead
        return min(8, cpu_count + 1)
    elif operation_type == 'cpu':
        # CPU bound: CPU count, max CPU cores
        return cpu_count
    else:
        # Default: conservative approach
        return min(4, cpu_count)

def get_file_operation_workers(file_count: int) -> int:
    """One-liner optimal workers untuk file operations"""
    return min(get_optimal_thread_count('io'), max(2, file_count // 100))

def get_download_workers() -> int:
    """One-liner optimal workers untuk download operations"""
    return min(4, get_optimal_thread_count('io'))

def get_rename_workers(total_files: int) -> int:
    """One-liner optimal workers untuk rename operations"""
    return min(6, max(2, total_files // 500))

# Convenience functions
optimal_io_workers = lambda: get_optimal_thread_count('io')
optimal_cpu_workers = lambda: get_optimal_thread_count('cpu')
safe_worker_count = lambda count: min(8, max(1, count))