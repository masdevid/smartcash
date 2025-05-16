"""
File: smartcash/dataset/services/augmentor/helpers/parallel_helper.py
Deskripsi: Helper untuk pemrosesan file parallel dengan pelaporan progress yang akurat dan efisien
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Optional
from tqdm.auto import tqdm

from smartcash.dataset.services.augmentor.augmentation_worker import process_single_file

def process_files_with_executor(
    file_list: List[str], 
    augmentation_params: Dict[str, Any], 
    num_workers: int,
    description: str = "Processing files",
    progress_callback: Optional[Callable] = None,
    **callback_params
) -> List[Dict[str, Any]]:
    """
    Proses file secara parallel dengan ProcessPoolExecutor dan progress reporting yang dioptimalkan.
    
    Args:
        file_list: List file yang akan diproses
        augmentation_params: Parameter untuk augmentasi
        num_workers: Jumlah worker untuk paralelisme
        description: Deskripsi untuk progress bar
        progress_callback: Callback untuk melaporkan progress
        **callback_params: Parameter tambahan untuk callback
        
    Returns:
        List hasil proses per file
    """
    # Validasi jumlah worker dan batasi berdasarkan CPU
    num_workers = max(1, min(os.cpu_count() or 4, num_workers))
    results = []
    
    # Hitung total file untuk progress reporting
    total_files = len(file_list)
    
    # Notifikasi awal dengan callback
    if progress_callback and total_files > 0:
        # Panggil callback untuk progress global
        progress_callback(
            message=f"🔄 Memproses {total_files} file dengan {num_workers} workers",
            status="info",
            progress=0,  # Progress global 0%
            total=100,  # Dari total 100%
            current_progress=0, 
            current_total=total_files,
            **callback_params
        )
    
    # Proses sekuensial untuk kasus sederhana
    if total_files == 1 or num_workers == 1:
        with tqdm(total=total_files, desc=description, disable=total_files < 3) as pbar:
            for i, file_path in enumerate(file_list):
                # Proses file tunggal
                result = process_single_file(file_path, **augmentation_params)
                results.append(result)
                pbar.update(1)
                
                # Report progress dengan parameter yang optimnal
                if progress_callback:
                    # Hitung progress global (0-100)
                    global_progress = int((i + 1) / total_files * 100)
                    progress_callback(
                        message=f"Memproses file {i+1}/{total_files}",
                        progress=global_progress,  # Progress global (0-100)
                        total=100,  # Dari total 100%
                        current_progress=i + 1,
                        current_total=total_files,
                        **callback_params
                    )
        return results
    
    # Proses parallel dengan executor    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit semua tugas dengan dictionary comprehension
        futures = {executor.submit(process_single_file, file_path, **augmentation_params): i 
                 for i, file_path in enumerate(file_list)}
        
        # Track progress dengan tqdm
        with tqdm(total=total_files, desc=description, disable=total_files < 3) as pbar:
            for i, future in enumerate(as_completed(futures)):
                # Dapatkan hasil dengan error handling
                try:
                    result = future.result()
                except Exception as e:
                    result = {"status": "error", "message": f"Error pada worker: {str(e)}", "generated": 0}
                
                # Tambah hasil dan update progress
                results.append(result)
                pbar.update(1)
                
                # Report progress dengan batasan frekuensi
                if progress_callback:
                    current_progress = i + 1
                    percent = int(current_progress / total_files * 100)
                    # Kurangi batasan frekuensi untuk update lebih sering
                    should_report = (i % max(1, total_files // 20) == 0 or i == 0 or i == total_files - 1 or percent % 5 == 0)
                    
                    if should_report:
                        progress_callback(
                            message=f"Memproses file {current_progress}/{total_files} ({percent}%)",
                            progress=percent,  # Progress global (0-100)
                            total=100,  # Dari total 100%
                            current_progress=current_progress,
                            current_total=total_files,
                            **callback_params
                        )
    
    # Report finalisasi
    if progress_callback:
        progress_callback(
            message=f"✅ Selesai memproses {total_files} file",
            status="success",
            progress=100,  # Progress global 100%
            total=100,    # Dari total 100%
            current_progress=total_files,
            current_total=total_files,
            **callback_params
        )
    
    return results