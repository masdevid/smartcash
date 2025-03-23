"""
File: smartcash/dataset/services/augmentor/helpers/parallel_helper.py
Deskripsi: Helper untuk eksekusi paralel augmentasi dengan ProcessPoolExecutor
"""

import time
from typing import Dict, List, Any, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm

from smartcash.dataset.services.augmentor.augmentation_worker import process_single_file

def process_files_with_executor(
    files: List[str],
    params: Dict,
    n_workers: int,
    desc: str,
    progress_callback: Callable = None,
    class_id: str = None,
    class_idx: int = None,
    total_classes: int = None
) -> List[Dict]:
    """
    Proses multiple file dengan ProcessPoolExecutor dan progress tracking.
    
    Args:
        files: List file yang akan diproses
        params: Dictionary parameter untuk worker
        n_workers: Jumlah worker
        desc: Deskripsi untuk tqdm progress bar
        progress_callback: Callback untuk melaporkan progress
        class_id: ID kelas untuk tracking (opsional)
        class_idx: Index kelas dalam iterasi (opsional)
        total_classes: Total jumlah kelas (opsional)
        
    Returns:
        List hasil augmentasi dari setiap file
    """
    # Setup untuk multiprocessing
    results, total_files = [], len(files)
    
    # Submit dan process dengan ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit semua tugas
        futures = {
            executor.submit(process_single_file, file_path, **params): file_path 
            for file_path in files
        }
        
        # Proses hasil dengan progress tracking
        for i, future in enumerate(tqdm(as_completed(futures), total=total_files, desc=desc)):
            try:
                results.append(future.result())
            except Exception as e:
                error_msg = f"❌ Error saat memproses {futures[future]}: {str(e)}"
                # Tambahkan informasi error ke results
                results.append({
                    "status": "error",
                    "message": error_msg,
                    "image_id": Path(futures[future]).stem,
                    "generated": 0
                })
                
            # Report progress dengan throttling
            if progress_callback and (i % max(1, total_files // 10) == 0 or i == total_files - 1):
                percentage = int((i+1) / total_files * 100)
                progress_args = {
                    'progress': i+1, 
                    'total': total_files, 
                    'message': f"{desc} ({percentage}%): {i+1}/{total_files} file", 
                    'status': "info", 
                    'step': 1
                }
                
                if class_id is not None and class_idx is not None and total_classes is not None:
                    progress_args.update({
                        'current_progress': i+1, 
                        'current_total': total_files, 
                        'class_id': class_id,
                        'class_idx': class_idx, 
                        'total_classes': total_classes
                    })
                    
                progress_callback(**progress_args)
                
    return results

def process_single_file_with_progress(
    idx: int, 
    file_path: str, 
    params: Dict, 
    total: int,
    progress_callback: Callable = None,
    desc: str = None, 
    class_id: str = None, 
    class_idx: int = None, 
    total_classes: int = None
) -> Dict:
    """
    Proses satu file dengan progress tracking.
    
    Args:
        idx: Index file dalam iterasi
        file_path: Path file yang diproses
        params: Parameter untuk worker
        total: Total jumlah file
        progress_callback: Callback untuk progress
        desc: Deskripsi proses
        class_id: ID kelas
        class_idx: Index kelas
        total_classes: Total kelas
        
    Returns:
        Hasil augmentasi file
    """
    # Tambahkan tracking multi-class ke params
    params_with_tracking = params.copy()
    params_with_tracking['track_multi_class'] = True
    
    # Proses file
    result = process_single_file(file_path, **params_with_tracking)
    
    # Report progress dengan throttling
    if progress_callback and (idx % max(1, total // 10) == 0 or idx == total - 1):
        progress_args = {
            'progress': idx+1, 
            'total': total, 
            'message': f"{desc}: {idx+1}/{total} file", 
            'status': "info", 
            'step': 1
        }
        
        if class_id is not None and class_idx is not None and total_classes is not None:
            progress_args.update({
                'current_progress': idx+1, 
                'current_total': total, 
                'class_id': class_id,
                'class_idx': class_idx, 
                'total_classes': total_classes
            })
            
        progress_callback(**progress_args)
        
    return result

def execute_batch_files(
    files: List[str],
    params: Dict,
    n_workers: int,
    desc: str,
    progress_callback: Callable = None,
    meta: Dict = None
) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Eksekusi batch file dengan batching otomatis dan tracking.
    
    Args:
        files: List file yang akan diproses
        params: Parameter untuk worker
        n_workers: Jumlah worker
        desc: Deskripsi proses
        progress_callback: Callback progress
        meta: Metadata untuk tracking
        
    Returns:
        Tuple (hasil augmentasi, metadata updated)
    """
    # Ekstrak informasi meta
    class_id = meta.get('class_id') if meta else None
    class_idx = meta.get('class_idx') if meta else None
    total_classes = meta.get('total_classes') if meta else None
    
    # Siapkan batch file berdasarkan ukuran worker
    batch_size = 100  # Default untuk batch processing
    
    # Eksekusi berdasarkan jumlah file dan worker
    if n_workers <= 1 or len(files) <= 1:
        # Mode single process
        results = [
            process_single_file_with_progress(
                i, file, params, len(files), progress_callback,
                desc, class_id, class_idx, total_classes
            ) for i, file in enumerate(tqdm(files, desc=desc))
        ]
    else:
        # Mode multiprocessing
        results = process_files_with_executor(
            files, params, n_workers, desc, progress_callback,
            class_id, class_idx, total_classes
        )
    
    # Generate metadata hasil
    result_meta = {
        'total_files': len(files),
        'successful': len([r for r in results if r.get('status') == 'success']),
        'failed': len([r for r in results if r.get('status') != 'success']),
        'generated': sum(r.get('generated', 0) for r in results),
    }
    
    # Multi-class updates dengan tracking yang lebih akurat
    result_meta['multi_class_updates'] = {}
    for result in results:
        if result.get('status') == 'success' and 'multi_class_update' in result:
            for cls, count in result['multi_class_update'].items():
                if cls not in result_meta['multi_class_updates']:
                    result_meta['multi_class_updates'][cls] = 0
                result_meta['multi_class_updates'][cls] += count
    
    return results, result_meta