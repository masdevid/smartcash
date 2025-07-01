"""
File: smartcash/common/worker_utils.py
Deskripsi: Centralized worker count utilities for optimal threading performance across SmartCash

This module provides a centralized set of utilities for determining optimal worker counts
for various types of operations across the SmartCash codebase. It helps ensure consistent
and efficient thread pool sizing based on operation type and system resources.

Key Features:
- Operation-specific worker counts (IO-bound, CPU-bound, mixed workloads)
- System-aware scaling based on available CPU cores
- Specialized functions for common operations (download, file operations, renaming)
- Safe limits to prevent thread explosion
- Consistent API for all worker count needs

Usage Examples:
    # For I/O bound operations (network, disk)
    workers = get_optimal_worker_count('io')
    
    # For CPU bound operations (processing)
    workers = get_optimal_worker_count('cpu')
    
    # For specific operations
    workers = get_download_workers()
    workers = get_file_operation_workers(file_count=1000)
    
    # Using convenience functions
    workers = optimal_io_workers()
    workers = optimal_cpu_workers()
    workers = optimal_mixed_workers()
"""

import os
from typing import Dict, Any, Optional, List, Union, Literal

# Define operation types for better type hinting
OperationType = Literal['io', 'cpu', 'mixed', 'io_bound', 'cpu_bound']

# Ekspos fungsi-fungsi untuk diimpor
__all__ = [
    'get_optimal_worker_count',
    'get_worker_counts_for_operations',
    'get_file_operation_workers',
    'get_download_workers',
    'get_rename_workers',
    'safe_worker_count',
    'optimal_io_workers',
    'optimal_cpu_workers',
    'optimal_mixed_workers'
]

def get_optimal_worker_count(operation_type: OperationType = 'io') -> int:
    """
    Get optimal worker count based on operation type and system specs.
    
    Args:
        operation_type: Type of operation to optimize for
            - 'io' or 'io_bound': I/O bound operations (network, disk)
            - 'cpu' or 'cpu_bound': CPU bound operations (processing)
            - 'mixed': Balanced between I/O and CPU
            
    Returns:
        Optimal worker count for the specified operation type
        
    Raises:
        ValueError: If operation_type is not recognized
    """
    cpu_count = os.cpu_count() or 1
    
    # Normalize operation type
    if operation_type in ('io', 'io_bound'):
        # I/O bound: CPU count + 1, max 8 to avoid overhead
        return min(8, cpu_count + 1)
    elif operation_type in ('cpu', 'cpu_bound'):
        # CPU bound: CPU count, max CPU cores
        return cpu_count
    elif operation_type == 'mixed':
        # Mixed workload: balance between CPU and I/O
        return min(6, max(2, cpu_count))
    else:
        # Invalid operation type
        raise ValueError(f"Invalid operation type: {operation_type}. " 
                         f"Must be one of: 'io', 'io_bound', 'cpu', 'cpu_bound', 'mixed'")

def get_worker_counts_for_operations() -> Dict[str, int]:
    """
    Get optimal worker counts for all standard operations.
    
    Returns:
        Dictionary with worker counts for different operation types
    """
    return {
        'download': get_optimal_worker_count('io'),
        'validation': get_optimal_worker_count('cpu'),
        'uuid_renaming': get_optimal_worker_count('mixed'),
        'preprocessing': get_optimal_worker_count('cpu'),
        'cleanup': get_optimal_worker_count('io'),
        'file_operations': get_file_operation_workers(500)  # Default for medium dataset
    }

def get_file_operation_workers(file_count: int) -> int:
    """
    Calculate optimal workers for file operations based on file count.
    
    Args:
        file_count: Number of files to process
        
    Returns:
        Optimal worker count for file operations
    """
    return min(get_optimal_worker_count('io'), max(2, file_count // 100))

def get_download_workers() -> int:
    """
    Get optimal workers for download operations.
    
    Returns:
        Optimal worker count for download operations
    """
    return min(4, get_optimal_worker_count('io'))

def get_rename_workers(total_files: int) -> int:
    """
    Calculate optimal workers for file renaming operations.
    
    Args:
        total_files: Total number of files to rename
        
    Returns:
        Optimal worker count for rename operations
    """
    return min(6, max(2, total_files // 500))

def safe_worker_count(count: int) -> int:
    """
    Ensure worker count is within safe limits.
    
    Args:
        count: Requested worker count
        
    Returns:
        Worker count clamped to safe limits (1-8)
    """
    return min(8, max(1, count))

# Convenience functions as one-liners with descriptive docstrings

def optimal_io_workers() -> int:
    """Get optimal worker count for I/O bound operations (network, disk).
    
    This is a convenience function that calls get_optimal_worker_count('io').
    Typically used for file operations, network requests, and downloads.
    
    Returns:
        int: Optimal number of workers for I/O bound tasks
    """
    return get_optimal_worker_count('io')

def optimal_cpu_workers() -> int:
    """Get optimal worker count for CPU bound operations (processing).
    
    This is a convenience function that calls get_optimal_worker_count('cpu').
    Typically used for data processing, image transformations, and calculations.
    
    Returns:
        int: Optimal number of workers for CPU bound tasks
    """
    return get_optimal_worker_count('cpu')

def optimal_mixed_workers() -> int:
    """Get optimal worker count for mixed operations (balanced I/O and CPU).
    
    This is a convenience function that calls get_optimal_worker_count('mixed').
    Typically used for tasks that involve both file operations and processing,
    such as reading files and performing transformations.
    
    Returns:
        int: Optimal number of workers for mixed workload tasks
    """
    return get_optimal_worker_count('mixed')
