"""
File: smartcash/dataset/augmentor/utils/batch_processor.py
Deskripsi: SRP module untuk batch processing dengan real-time progress updates
"""

from typing import List, Callable, Any, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from smartcash.common.threadpools import get_optimal_thread_count
from smartcash.dataset.augmentor.utils.progress_tracker import ProgressTracker

def process_batch(items: List[Any], process_func: Callable, max_workers: int = None, 
                 progress_tracker: ProgressTracker = None, operation_name: str = "processing") -> List[Dict[str, Any]]:
    """Fixed batch processing dengan real-time progress updates"""
    max_workers = max_workers or min(get_optimal_thread_count(), 8)
    results = []
    
    if not items:
        if progress_tracker:
            progress_tracker.log_info("⚠️ Tidak ada item untuk diproses")
        return results
    
    total_items = len(items)
    if progress_tracker:
        progress_tracker.log_info(f"🚀 Memulai {operation_name}: {total_items} item")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
        
        for completed_count, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)
                
                # Real-time progress update
                if progress_tracker:
                    progress_tracker.progress("current", completed_count, total_items, 
                                            f"{operation_name}: {completed_count}/{total_items}")
                
                # Log progress setiap 10%
                if completed_count % max(1, total_items // 10) == 0:
                    successful = sum(1 for r in results if r.get('status') == 'success')
                    success_rate = (successful / completed_count) * 100
                    if progress_tracker:
                        progress_tracker.log_info(
                        f"📊 Progress: {completed_count}/{total_items} ({success_rate:.1f}% berhasil)"
                    )
                    
            except Exception as e:
                results.append({'status': 'error', 'error': str(e)})
    
    # Final summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    if progress_tracker:
        progress_tracker.log_success(
        f"✅ {operation_name} selesai: {successful}/{total_items} berhasil"
    )
    
    return results

def process_batch_split_aware(items: List[Any], process_func: Callable, max_workers: int = None, 
                            progress_tracker: ProgressTracker = None, operation_name: str = "processing",
                            split_context: str = None) -> List[Dict[str, Any]]:
    """Batch processing dengan split context"""
    context_msg = f" untuk split {split_context}" if split_context else ""
    return process_batch(items, process_func, max_workers, progress_tracker, f"{operation_name}{context_msg}")

# One-liner utilities
create_batch_processor = lambda max_workers=None: lambda items, func, tracker, name: process_batch(items, func, max_workers, tracker, name)