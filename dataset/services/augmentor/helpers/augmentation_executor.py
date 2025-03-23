"""
File: smartcash/dataset/services/augmentor/helpers/augmentation_executor.py
Deskripsi: Helper untuk eksekusi augmentasi dengan tracking dinamis dan prioritisasi kelas
"""

import time
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict
from smartcash.dataset.services.augmentor.helpers.parallel_helper import process_files_with_executor
from smartcash.dataset.services.augmentor.helpers.tracking_helper import (
    track_class_progress, prioritize_classes_by_need
)

def execute_augmentation_with_tracking(
    service, 
    class_data: Dict[str, Any], 
    augmentation_types: List[str], 
    num_variations: int,
    output_prefix: str, 
    validate_results: bool, 
    process_bboxes: bool, 
    n_workers: int,
    paths: Dict[str, str], 
    split: str, 
    target_count: int, 
    start_time: float
) -> Dict[str, Any]:
    """
    Eksekusi augmentasi dengan tracking dinamis kelas dan prioritisasi.
    
    Args:
        service: Instance AugmentationService
        class_data: Data kelas dan file untuk augmentasi
        augmentation_types: Tipe augmentasi
        num_variations: Jumlah variasi per file
        output_prefix: Prefix untuk file output
        validate_results: Validasi hasil
        process_bboxes: Proses bounding box
        n_workers: Jumlah worker
        paths: Path direktori
        split: Nama split dataset
        target_count: Target jumlah instance per kelas
        start_time: Waktu mulai
        
    Returns:
        Dictionary hasil augmentasi
    """
    # Buat pipeline augmentasi
    try: 
        pipeline = service.pipeline_factory.create_pipeline(
            augmentation_types=augmentation_types or ['combined'],
            img_size=(640, 640),
            include_normalize=False,
            intensity=1.0
        )
        service.logger.info(f"✅ Pipeline augmentasi dibuat: {', '.join(augmentation_types or ['combined'])}")
    except Exception as e: 
        return {"status": "error", "message": f"Error membuat pipeline augmentasi: {str(e)}"}
    
    # Dapatkan files yang terpilih untuk augmentasi
    selected_files = class_data.get('selected_files', [])
    if not selected_files:
        return {"status": "info", "message": "Tidak ada file yang memerlukan augmentasi", "generated": 0}
        
    # Siapkan parameter augmentasi
    augmentation_params = {
        'pipeline': pipeline, 
        'num_variations': num_variations, 
        'output_prefix': output_prefix,
        'process_bboxes': process_bboxes, 
        'validate_results': validate_results, 
        'bbox_augmentor': service.bbox_augmentor,
        'labels_input_dir': paths['labels_input_dir'], 
        'images_output_dir': paths['images_output_dir'],
        'labels_output_dir': paths['labels_output_dir']
    }
    
    # Notifikasi mulai augmentasi
    service.report_progress(
        message=f"🚀 Memulai augmentasi {len(selected_files)} file dengan tracking dinamis",
        status="info", step=1
    )
    
    # Proses augmentasi dengan prioritas kelas
    augmentation_results = execute_prioritized_class_augmentation(
        service, class_data, augmentation_params, n_workers, target_count, paths
    )
    
    # Durasi total
    duration = time.time() - start_time
    
    # Return hasil augmentasi
    return {
        'original': sum(class_data.get('class_counts', {}).values()),
        'selected_files': len(selected_files),
        'generated': augmentation_results['total_generated'],
        'files_augmented': augmentation_results['total_files_augmented'], 
        'class_stats': augmentation_results['class_stats'],
        'duration': duration,
        'augmentation_types': augmentation_types or ['combined'],
        'status': 'success',
        'split': split
    }

def execute_prioritized_class_augmentation(
    service, 
    class_data: Dict[str, Any], 
    augmentation_params: Dict[str, Any],
    n_workers: int,
    target_count: int,
    paths: Dict[str, str]
) -> Dict[str, Any]:
    """
    Eksekusi augmentasi dengan prioritasi kelas dan tracking kelas yang sudah mencapai target.
    
    Args:
        service: Instance AugmentationService
        class_data: Data kelas dan file untuk augmentasi
        augmentation_params: Parameter augmentasi
        n_workers: Jumlah worker
        target_count: Target jumlah instance per kelas
        paths: Path direktori
        
    Returns:
        Dictionary hasil augmentasi
    """
    # Inisialisasi tracking hasil dengan one-liner
    result_stats = {
        'total_files_augmented': 0,
        'total_generated': 0,
        'class_stats': {},
        'fulfilled_classes': set()
    }
    
    # Ekstrak data kelas untuk prioritisasi dan tracking
    class_files = class_data.get('files_by_class', {})
    class_counts = class_data.get('class_counts', {})
    class_needs = class_data.get('augmentation_needs', {})
    
    # Prioritaskan augmentasi kelas yang paling dibutuhkan
    classes_to_augment = [cls for cls, need in sorted(class_needs.items(), 
                                                    key=lambda x: (x[1], -class_counts.get(x[0], 0)), 
                                                    reverse=True) 
                         if need > 0]
    
    # Pelacakan dinamis kelas yang sudah mencapai target
    current_class_counts = dict(class_counts)
    
    # Track file yang sudah diproses untuk mencegah duplikasi
    processed_files = set()
    
    service.logger.info(f"🔄 Menjalankan augmentasi dengan prioritasi untuk {len(classes_to_augment)} kelas")
    
    # Proses kelas secara berurutan berdasarkan prioritas kebutuhan
    for i, class_id in enumerate(classes_to_augment):
        if service._stop_signal:
            break
            
        # Dapatkan kebutuhan saat ini (mungkin sudah berubah karena augmentasi kelas lain)
        current_need = max(0, target_count - current_class_counts.get(class_id, 0))
        if current_need <= 0:
            service.logger.info(f"✅ Kelas {class_id} sudah mencapai target, dilewati")
            continue
            
        service.logger.info(f"🔄 Memproses kelas {class_id} ({i+1}/{len(classes_to_augment)}): perlu {current_need} instances")
        
        # Pilih file untuk kelas ini dengan prioritas tertinggi
        files_for_class = _select_files_for_class(
            class_id, class_files, current_class_counts, 
            target_count, processed_files, class_data
        )
        
        if not files_for_class:
            service.logger.warning(f"⚠️ Tidak ada file tersedia untuk kelas {class_id}, dilewati")
            continue
            
        service.logger.info(f"📑 Menggunakan {len(files_for_class)} file untuk augmentasi kelas {class_id}")
        
        # Notifikasi progres
        service.report_progress(
            message=f"Augmentasi kelas {class_id} ({i+1}/{len(classes_to_augment)})",
            status="info", step=1, current_progress=i, current_total=len(classes_to_augment),
            class_id=class_id
        )
        
        # Augmentasi untuk kelas ini
        desc = f"Augmentasi kelas {class_id}"
        class_results = process_files_with_executor(
            files_for_class, 
            {**augmentation_params, 'class_id': class_id},
            n_workers,
            desc,
            lambda *args, **kwargs: service.report_progress(*args, **kwargs),
            class_id=class_id,
            class_idx=i,
            total_classes=len(classes_to_augment)
        )
        
        # Update kelas yang sudah diproses
        processed_files.update(files_for_class)
        
        # Update tracking hasil
        generated_for_class = sum(r.get('generated', 0) for r in class_results)
        success_for_class = sum(1 for r in class_results if r.get('status') == 'success')
        
        # Simpan statistik kelas
        result_stats['class_stats'][class_id] = {
            'original': len(class_files.get(class_id, [])),
            'files_augmented': len(files_for_class),
            'target': target_count,
            'before_augmentation': current_class_counts.get(class_id, 0),
            'generated': generated_for_class,
            'success_rate': success_for_class / max(1, len(files_for_class))
        }
        
        # Update counter global
        result_stats['total_files_augmented'] += len(files_for_class)
        result_stats['total_generated'] += generated_for_class
        
        # Update tracking progres dengan helper
        tracking_result = track_class_progress(
            class_results, 
            current_class_counts, 
            target_count, 
            service.logger
        )
        
        # Update variabel tracking
        current_class_counts = tracking_result['updated_counts']
        result_stats['fulfilled_classes'].update(tracking_result['fulfilled_classes'])
        
        # Log hasil augmentasi kelas
        service.logger.info(f"✅ Kelas {class_id}: {generated_for_class} variasi dibuat dari {len(files_for_class)} file " +
                          f"({current_class_counts.get(class_id, 0)}/{target_count})")
        
        # Laporkan progres hasil
        service.report_progress(
            message=f"✅ Kelas {class_id}: {generated_for_class} variasi dibuat ({current_class_counts.get(class_id, 0)}/{target_count})",
            status="success", step=1, current_progress=i+1, current_total=len(classes_to_augment),
            class_id=class_id
        )
    
    # Tambahkan informasi count akhir dan fulfilled classes
    result_stats['class_counts_after'] = current_class_counts
    
    # Laporan akhir
    summary_message = f"✅ Augmentasi selesai: {result_stats['total_generated']} variasi dihasilkan"
    service.logger.info(summary_message)
    service.report_progress(message=summary_message, status="success", step=2)
    
    return result_stats

def _select_files_for_class(
    class_id: str, 
    files_by_class: Dict[str, List[str]],
    current_counts: Dict[str, int], 
    target_count: int,
    processed_files: Set[str],
    class_data: Dict[str, Any]
) -> List[str]:
    """
    Pilih file untuk augmentasi kelas berdasarkan prioritas.
    
    Args:
        class_id: ID kelas
        files_by_class: Dictionary file per kelas
        current_counts: Jumlah instance kelas saat ini
        target_count: Target jumlah instance per kelas
        processed_files: Set file yang sudah diproses
        class_data: Data kelas dan metadata
        
    Returns:
        List file untuk diaugmentasi
    """
    # File yang memiliki kelas ini
    class_candidates = files_by_class.get(class_id, [])
    
    # Hitung berapa file dibutuhkan berdasarkan kebutuhan
    current_need = max(0, target_count - current_counts.get(class_id, 0))
    num_files_needed = min(len(class_candidates), current_need)
    
    if num_files_needed <= 0:
        return []
        
    # Filter file yang belum diproses
    eligible_files = [f for f in class_candidates if f not in processed_files]
    
    if not eligible_files:
        return []
    
    # Prioritaskan file berdasarkan kompleksitas kelas
    file_metadata = class_data.get('files_metadata', {})
    prioritized_files = []
    
    # Dapatkan kelas yang sudah terpenuhi target
    fulfilled_classes = set(cls for cls, count in current_counts.items() if count >= target_count)
    
    # High priority: File dengan kelas yang diperlukan dan tidak memiliki kelas yang sudah terpenuhi
    high_priority = [f for f in eligible_files 
                   if f in file_metadata 
                   and class_id in file_metadata[f].get('classes', set())
                   and not any(cls in fulfilled_classes 
                              for cls in file_metadata[f].get('classes', set()))]
    
    # Medium priority: File dengan jumlah kelas sedikit
    medium_priority = [f for f in eligible_files 
                     if f in file_metadata 
                     and f not in high_priority
                     and len(file_metadata[f].get('classes', set())) <= 2]
    
    # Low priority: File lainnya
    low_priority = [f for f in eligible_files if f not in high_priority and f not in medium_priority]
    
    # Urutkan berdasarkan jumlah kelas (lebih sedikit lebih prioritas) dengan one-liner
    high_priority.sort(key=lambda f: len(file_metadata.get(f, {}).get('classes', set())) if f in file_metadata else float('inf'))
    medium_priority.sort(key=lambda f: len(file_metadata.get(f, {}).get('classes', set())) if f in file_metadata else float('inf'))
    
    # Gabungkan prioritas dengan one-liner
    prioritized_files = high_priority + medium_priority + low_priority
    
    # Batasi sesuai kebutuhan dengan one-liner
    return prioritized_files[:num_files_needed]