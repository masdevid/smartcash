"""
File: smartcash/common/threadpools.py
Deskripsi: Utilitas untuk menstandarisasi penggunaan ThreadPoolExecutor
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, TypeVar, Any, Dict
from tqdm.auto import tqdm

T = TypeVar('T')  # Input type
R = TypeVar('R')  # Result type

def get_optimal_thread_count() -> int:
    """
    Dapatkan jumlah thread optimal berdasarkan CPU dan lingkungan.
    
    Returns:
        Jumlah worker thread yang direkomendasikan
    """
    # Dapatkan jumlah CPU yang tersedia
    cpu_count = os.cpu_count() or 4
    
    # Gunakan 2x cpu_count untuk workload I/O-bound (seperti pemrosesan file)
    return min(32, cpu_count * 2)

def process_in_parallel(
    items: List[T], 
    process_func: Callable[[T], R], 
    max_workers: int = None,
    desc: str = None,
    unit: str = "item",
    show_progress: bool = True
) -> List[R]:
    """
    Proses daftar item secara paralel menggunakan ThreadPoolExecutor.
    
    Args:
        items: Daftar item yang akan diproses
        process_func: Fungsi untuk memproses setiap item
        max_workers: Jumlah maksimum worker (default: optimal berdasarkan CPU)
        desc: Deskripsi untuk progress bar
        unit: Unit untuk progress bar
        show_progress: Apakah menampilkan progress bar
        
    Returns:
        List hasil pemrosesan
    """
    if not items:
        return []
        
    # Gunakan jumlah thread optimal jika tidak ditentukan
    if max_workers is None:
        max_workers = get_optimal_thread_count()
    
    results = []
    
    # Setup progress bar
    pbar = tqdm(total=len(items), desc=desc, unit=unit, disable=not show_progress)
    
    # Proses dengan ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit semua tugas
        futures = [executor.submit(process_func, item) for item in items]
        
        # Kumpulkan hasil
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                # Log error jika diperlukan dan lanjutkan
                results.append(None)
            finally:
                # Update progress
                pbar.update(1)
    
    pbar.close()
    return results

def process_with_stats(
    items: List[T], 
    process_func: Callable[[T], Dict], 
    max_workers: int = None,
    desc: str = None,
    show_progress: bool = True
) -> Dict[str, int]:
    """
    Versi khusus process_in_parallel yang mengumpulkan statistik dari hasil.
    
    Args:
        items: Daftar item yang akan diproses
        process_func: Fungsi yang mengembalikan status per item (dict dengan keys seperti 'success', 'error', dll)
        max_workers: Jumlah maksimum worker
        desc: Deskripsi untuk progress bar
        show_progress: Apakah menampilkan progress bar
        
    Returns:
        Dictionary berisi akumulasi statistik
    """
    results = process_in_parallel(items, process_func, max_workers, desc, show_progress=show_progress)
    
    # Inisialisasi stats
    stats = {}
    
    # Gabungkan hasil statistik
    for result in results:
        if result:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    stats[key] = stats.get(key, 0) + value
    
    return stats