"""
File: smartcash/dataset/services/augmentor/helpers/augmentation_executor.py
Deskripsi: Helper untuk eksekusi augmentasi dengan tracking dinamis, prioritisasi kelas, dan pelaporan progress yang dioptimalkan
"""

import time
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict

# Import utils yang sudah direfactor
from smartcash.dataset.services.augmentor.helpers.parallel_helper import process_files_with_executor
from smartcash.dataset.services.augmentor.helpers.tracking_helper import track_class_progress
from smartcash.dataset.utils.file_mapping_utils import select_prioritized_files_for_class

def execute_prioritized_class_augmentation(
    service, 
    class_data: Dict[str, Any], 
    augmentation_params: Dict[str, Any],
    n_workers: int,
    target_count: int,
    paths: Dict[str, str]
) -> Dict[str, Any]:
    """Eksekusi augmentasi dengan prioritasi kelas dan tracking kelas yang sudah mencapai target."""
    # Inisialisasi tracking hasil dengan one-liner
    result_stats = {'total_files_augmented': 0, 'total_generated': 0, 'class_stats': {}, 'fulfilled_classes': set()}
    
    # Ekstrak data kelas untuk prioritisasi dan tracking
    class_files = class_data.get('files_by_class', {})
    class_counts = class_data.get('class_counts', {})
    class_needs = class_data.get('augmentation_needs', {})
    selected_files = class_data.get('selected_files', [])
    
    # Cek apakah ini mode non-balancing (balancer dinonaktifkan)
    is_non_balancing_mode = not class_files and selected_files
    
    if is_non_balancing_mode:
        service.logger.info(f"🔄 Menjalankan augmentasi tanpa balancing untuk {len(selected_files)} file")
        
        # Untuk mode non-balancing, proses semua file sekaligus
        augmentation_results = process_files_with_executor(
            selected_files,
            augmentation_params,
            n_workers,
            "Augmentasi semua file",
            service.report_progress
        )
        
        # Hitung hasil
        total_generated = sum(r.get('generated', 0) for r in augmentation_results)
        success_count = sum(1 for r in augmentation_results if r.get('status') == 'success')
        
        # Update statistik
        result_stats['total_files_augmented'] = len(selected_files)
        result_stats['total_generated'] = total_generated
        
        # Tambahkan statistik umum
        result_stats['class_stats']['all'] = {
            'original': len(selected_files),
            'files_augmented': len(selected_files),
            'target': target_count,
            'before_augmentation': len(selected_files),
            'generated': total_generated,
            'success_rate': success_count / max(1, len(selected_files))
        }
        
        # Log hasil
        service.logger.info(f"✅ Augmentasi selesai: {total_generated} variasi dibuat dari {len(selected_files)} file")
        
        # Report progres hasil
        service.report_progress(
            message=f"✅ Augmentasi selesai: {total_generated} variasi dibuat",
            status="success", step=1,
            progress=len(selected_files),
            total=len(selected_files)
        )
        
        return result_stats
    
    # Mode balancing - prioritaskan kelas dengan one-liner sort berdasarkan kebutuhan dan jumlah instance
    classes_to_augment = [cls for cls, need in sorted(class_needs.items(), 
                                                     key=lambda x: (x[1], -class_counts.get(x[0], 0)), 
                                                     reverse=True) 
                         if need > 0]
    
    # Jika tidak ada kelas yang perlu diaugmentasi tapi ada file yang dipilih (forced augmentation)
    if not classes_to_augment and selected_files:
        service.logger.info(f"🔄 Menjalankan augmentasi yang dipaksa untuk {len(selected_files)} file")
        
        # Untuk mode forced augmentation, proses semua file sekaligus
        augmentation_results = process_files_with_executor(
            selected_files,
            augmentation_params,
            n_workers,
            "Augmentasi yang dipaksa",
            service.report_progress
        )
        
        # Hitung hasil
        total_generated = sum(r.get('generated', 0) for r in augmentation_results)
        success_count = sum(1 for r in augmentation_results if r.get('status') == 'success')
        
        # Update statistik
        result_stats['total_files_augmented'] = len(selected_files)
        result_stats['total_generated'] = total_generated
        
        # Tambahkan statistik umum
        result_stats['class_stats']['forced'] = {
            'original': len(selected_files),
            'files_augmented': len(selected_files),
            'target': target_count,
            'before_augmentation': 0,
            'generated': total_generated,
            'success_rate': success_count / max(1, len(selected_files))
        }
        
        # Log hasil
        service.logger.info(f"✅ Augmentasi yang dipaksa selesai: {total_generated} variasi dibuat dari {len(selected_files)} file")
        
        # Report progres hasil
        service.report_progress(
            message=f"✅ Augmentasi yang dipaksa selesai: {total_generated} variasi dibuat",
            status="success", step=1,
            progress=len(selected_files),
            total=len(selected_files)
        )
        
        return result_stats
    
    # Tracking dinamis dan initialized state
    current_class_counts, processed_files = dict(class_counts), set()
    
    service.logger.info(f"🔄 Menjalankan augmentasi dengan prioritasi untuk {len(classes_to_augment)} kelas")
    
    # Hitung total file untuk semua kelas dengan list comprehension
    total_files_all_classes = sum(len(select_prioritized_files_for_class(cls, class_files, current_class_counts, 
                                                            target_count, processed_files, class_data))
                                 for cls in classes_to_augment)
    
    # Tracking progress global
    processed_file_count = 0
    
    # Proses kelas secara berurutan berdasarkan prioritas
    for i, class_id in enumerate(classes_to_augment):
        if service._stop_signal: break
            
        # Cek kebutuhan kelas saat ini
        if (current_need := max(0, target_count - current_class_counts.get(class_id, 0))) <= 0:
            service.logger.info(f"✅ Kelas {class_id} sudah mencapai target, dilewati")
            continue
            
        service.logger.info(f"🔄 Memproses kelas {class_id} ({i+1}/{len(classes_to_augment)}): perlu {current_need} instances")
        
        # Pilih file untuk kelas dengan strategi prioritas
        files_for_class = select_prioritized_files_for_class(class_id, class_files, current_class_counts, 
                                              target_count, processed_files, class_data)
        
        if not files_for_class:
            service.logger.warning(f"⚠️ Tidak ada file tersedia untuk kelas {class_id}, dilewati")
            continue
            
        service.logger.info(f"📑 Menggunakan {len(files_for_class)} file untuk augmentasi kelas {class_id}")
        
        # Report progress dengan informasi class step
        class_step = f"Kelas {class_id} ({i+1}/{len(classes_to_augment)})"
        service.report_progress(
            message=f"Augmentasi kelas {class_id} ({i+1}/{len(classes_to_augment)})",
            status="info", step=1, 
            current_progress=processed_file_count, 
            current_total=total_files_all_classes,
            split_step=class_step,
            class_id=class_id
        )
        
        # Augmentasi kelas dengan proses parallel
        class_results = process_files_with_executor(
            files_for_class, 
            {**augmentation_params, 'class_id': class_id},
            n_workers,
            f"Augmentasi kelas {class_id}",
            service.report_progress,
            class_id=class_id,
            class_idx=i,
            total_classes=len(classes_to_augment)
        )
        
        # Update tracking file dan progress
        processed_files.update(files_for_class)
        processed_file_count += len(files_for_class)
        
        # Hitung statistik kelas dengan list comprehension
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
        
        # Update tracking dengan helper
        tracking_result = track_class_progress(class_results, current_class_counts, target_count, service.logger)
        current_class_counts, fulfilled_classes_update = tracking_result['updated_counts'], tracking_result['fulfilled_classes']
        result_stats['fulfilled_classes'].update(fulfilled_classes_update)
        
        # Log hasil dengan string formatting
        service.logger.info(f"✅ Kelas {class_id}: {generated_for_class} variasi dibuat dari {len(files_for_class)} file " +
                          f"({current_class_counts.get(class_id, 0)}/{target_count})")
        
        # Report progres hasil
        service.report_progress(
            message=f"✅ Kelas {class_id}: {generated_for_class} variasi dibuat ({current_class_counts.get(class_id, 0)}/{target_count})",
            status="success", step=1, 
            progress=processed_file_count, 
            total=total_files_all_classes,
            current_progress=len(files_for_class), 
            current_total=len(files_for_class),
            split_step=class_step,
            class_id=class_id
        )
    
    # Tambahkan informasi count akhir
    result_stats['class_counts_after'] = current_class_counts
    
    # Laporan akhir dengan string formatting
    summary_message = f"✅ Augmentasi selesai: {result_stats['total_generated']} variasi dihasilkan"
    service.logger.info(summary_message)
    
    # Progress final
    service.report_progress(
        message=summary_message, 
        status="success", 
        step=2,
        progress=total_files_all_classes,
        total=total_files_all_classes
    )
    
    return result_stats

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
    """Eksekusi augmentasi dengan tracking dinamis kelas dan prioritisasi."""
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
        
    # Siapkan parameter augmentasi dengan dictionary unpacking
    augmentation_params = {
        'pipeline': pipeline, 
        'num_variations': num_variations, 
        'output_prefix': output_prefix,
        'process_bboxes': process_bboxes, 
        'validate_results': validate_results, 
        'bbox_augmentor': service.bbox_augmentor,
        'labels_input_dir': paths['labels_input_dir'], 
        'images_output_dir': paths['images_output_dir'],
        'labels_output_dir': paths['labels_output_dir'],
        'track_multi_class': True
    }
    
    # Laporkan total file untuk normalisasi progress
    total_files_to_process = len(selected_files)
    service.report_progress(
        message=f"🚀 Memulai augmentasi {total_files_to_process} file dengan tracking dinamis",
        status="info", step=1,
        total_files_all=total_files_to_process
    )
    
    # Proses augmentasi dengan prioritas kelas
    augmentation_results = execute_prioritized_class_augmentation(
        service, class_data, augmentation_params, n_workers, target_count, paths
    )
    
    # Durasi total dan hasil
    duration = time.time() - start_time
    
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