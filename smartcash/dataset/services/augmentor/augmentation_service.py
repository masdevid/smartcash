"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan utama untuk augmentasi dataset dengan dukungan multi-processing, balancing class dan tracking progres per kelas
"""

import os, time, glob, random
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import albumentations as A
from tqdm.auto import tqdm
from collections import defaultdict

from smartcash.common.logger import get_logger
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.services.augmentor.bbox_augmentor import BBoxAugmentor
from smartcash.dataset.services.augmentor.class_balancer import ClassBalancer
from smartcash.dataset.services.augmentor.augmentation_worker import process_single_file
from smartcash.dataset.utils.augmentor_utils import move_files_to_preprocessed, process_augmentation_results

class AugmentationService:
    """
    Layanan augmentasi dataset dengan dukungan multiprocessing, balancing class dan tracking progres per kelas.
    Implementasi sesuai SRP dengan delegasi tugas ke worker dan balancer.
    """
    
    def __init__(self, config: Dict = None, data_dir: str = 'data', logger=None, num_workers: int = None):
        """Inisialisasi AugmentationService dengan parameter utama."""
        self.config, self.data_dir, self.logger = config or {}, data_dir, logger or get_logger("augmentation_service")
        self.num_workers = num_workers if num_workers is not None else self.config.get('augmentation', {}).get('num_workers', 4)
        self.logger.debug(f"🔧 Menggunakan {self.num_workers} worker untuk augmentasi")
        
        # Inisialisasi komponen-komponen utama
        self.pipeline_factory = AugmentationPipelineFactory(self.config, self.logger)
        self.bbox_augmentor = BBoxAugmentor(self.config, self.logger)
        self.class_balancer = ClassBalancer(self.config, self.logger)
        
        # State untuk progress tracking
        self._stop_signal, self._progress_callbacks = False, []
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback untuk progress tracking."""
        if callback and callable(callback): self._progress_callbacks.append(callback)
    
    def report_progress(self, progress: int = None, total: int = None, message: str = None, status: str = 'info', **kwargs) -> None:
        """Laporkan progress dengan callback dan hindari duplikasi parameter."""
        for callback in self._progress_callbacks:
            try:
                # Menghindari duplikasi parameter dengan one-liner
                kwargs.update({'progress': progress, 'total': total, 'message': message, 'status': status})
                [kwargs.pop(k, None) for k in ['current_progress', 'current_total'] if k in kwargs and 'current_progress' in kwargs]
                callback(**kwargs)
            except Exception as e: self.logger.warning(f"⚠️ Error pada progress callback: {str(e)}")
    
    def augment_dataset(
        self, split: str = 'train', augmentation_types: List[str] = None, num_variations: int = 2,
        output_prefix: str = 'aug', validate_results: bool = True, resume: bool = False,
        process_bboxes: bool = True, target_balance: bool = False, num_workers: int = None,
        move_to_preprocessed: bool = True, target_count: int = 1000) -> Dict[str, Any]:
        """Augmentasi dataset dengan penggunaan multiprocessing, balancing class dan tracking progres per kelas."""
        # Reset stop signal dan catat waktu mulai
        self._stop_signal, start_time = False, time.time()
        
        # Dapatkan paths dengan one-liner
        paths = {
            'preprocessed_dir': self.config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed'),
            'augmented_dir': self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
        }
        paths.update({
            'input_dir': os.path.join(paths['preprocessed_dir'], split),
            'images_input_dir': os.path.join(paths['preprocessed_dir'], split, 'images'),
            'labels_input_dir': os.path.join(paths['preprocessed_dir'], split, 'labels'),
            'output_dir': paths['augmented_dir'],
            'images_output_dir': os.path.join(paths['augmented_dir'], 'images'),
            'labels_output_dir': os.path.join(paths['augmented_dir'], 'labels'),
            'final_output_dir': paths['preprocessed_dir']
        })
        
        # Buat direktori output jika belum ada dengan one-liner
        [os.makedirs(paths[key], exist_ok=True) for key in ['images_output_dir', 'labels_output_dir']]
        
        # Dapatkan daftar file input dan validasi
        file_prefix = self.config.get('preprocessing', {}).get('file_prefix', 'rp')
        image_files = glob.glob(os.path.join(paths['images_input_dir'], f"{file_prefix}_*.jpg"))
        
        if not image_files:
            message = f"Tidak ada file gambar ditemukan dengan pola {file_prefix}_*.jpg di direktori {paths['images_input_dir']}"
            self.logger.warning(f"⚠️ {message}")
            return {"status": "error", "message": message}
        
        # Persiapkan struktur data untuk balancing dan tracking
        class_data = {'files': defaultdict(list), 'counts': defaultdict(int), 'needs': {}}
        
        # Persiapkan balancing jika diminta
        if target_balance:
            try:
                # Report progress dan gunakan balancer
                self.report_progress(message=f"🔍 Menganalisis peta distribusi kelas untuk balancing", status="info", step=0)
                
                result = self.class_balancer.prepare_balanced_dataset(
                    image_files=image_files, labels_dir=paths['labels_input_dir'],
                    target_count=target_count, filter_single_class=True, 
                    progress_callback=lambda *args, **kwargs: self.report_progress(*args, **kwargs)
                )
                
                # Ekstrak data balancing dengan one-liner
                class_data = {
                    'files': result.get('class_files', {}),
                    'counts': {k: len(v) for k, v in result.get('class_files', {}).items()},
                    'needs': result.get('augmentation_needs', {})
                }
                
                # Log statistik balancing dan update file yang akan diproses
                classes_needing = sum(1 for v in class_data['needs'].values() if v > 0)
                total_needed = sum(class_data['needs'].values())
                
                self.logger.info(f"📊 Statistik Balancing Kelas (target: {target_count}/kelas): {classes_needing} kelas perlu ditambah {total_needed} sampel")
                
                # Update file yang akan diproses dengan one-liner
                balanced_files = result.get('selected_files', [])
                image_files = balanced_files if balanced_files else image_files
                self.logger.info(f"🔄 Menggunakan {len(image_files)} file untuk augmentasi ({classes_needing} kelas)")
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat balancing class: {str(e)}. Menggunakan semua file.")
        
        if len(image_files) == 0:
            message = "Tidak ada file yang valid untuk diaugmentasi"
            self.logger.warning(f"⚠️ {message}")
            return {"status": "error", "message": message}
        
        # Log jumlah file dan cek resume
        self.logger.info(f"🔍 Ditemukan {len(image_files)} file untuk augmentasi")
        if resume: self.logger.info(f"⏩ Resume augmentation belum diimplementasikan, melanjutkan dengan semua file")
        
        # Setup pipeline dan validasi
        try:
            pipeline = self.pipeline_factory.create_pipeline(
                augmentation_types=augmentation_types or ['combined'],
                img_size=(640, 640),
                include_normalize=False,
                intensity=1.0
            )
            self.logger.info(f"✅ Pipeline augmentasi berhasil dibuat: {', '.join(augmentation_types or ['combined'])}")
        except Exception as e:
            message = f"Error membuat pipeline augmentasi: {str(e)}"
            self.logger.error(f"❌ {message}")
            return {"status": "error", "message": message}
        
        # Siapkan parameter augmentasi dengan one-liner
        augmentation_params = {
            'pipeline': pipeline, 'num_variations': num_variations, 'output_prefix': output_prefix,
            'process_bboxes': process_bboxes, 'validate_results': validate_results, 'bbox_augmentor': self.bbox_augmentor,
            'labels_input_dir': paths['labels_input_dir'], 'images_output_dir': paths['images_output_dir'],
            'labels_output_dir': paths['labels_output_dir']
        }
        
        # Eksekusi augmentasi berdasarkan mode balancing
        results = self._augment_with_class_tracking(
            class_data, augmentation_params, target_count,
            num_workers, paths, split, augmentation_types or ['combined'], start_time
        ) if target_balance and class_data['files'] else self._augment_without_class_tracking(
            image_files, augmentation_params, num_workers,
            paths, split, augmentation_types or ['combined'], start_time
        )
        
        # Tambahkan informasi move_to_preprocessed ke hasil
        results['move_to_preprocessed'] = move_to_preprocessed
        
        return results
    
    def _augment_with_class_tracking(
        self, class_data: Dict, augmentation_params: Dict, target_count: int,
        num_workers: int, paths: Dict, split: str, augmentation_types: List[str], start_time: float
    ) -> Dict[str, Any]:
        """Augmentasi dengan tracking per kelas untuk balancing dengan pendekatan DRY."""
        # Ekstrak data class untuk readability
        class_files, class_counts, class_needs = class_data['files'], class_data['counts'], class_data['needs']
        
        # Setup tracking dan statistik
        classes_to_augment = [cls for cls, need in class_needs.items() if need > 0]
        n_workers = num_workers if num_workers is not None else self.num_workers
        
        # Inisialisasi statistik hasil
        result_stats = {'total_augmented': 0, 'total_generated': 0, 'failed': 0, 'class_stats': {}, 'success': True}
        
        # Mulai proses augmentasi per kelas
        self.report_progress(
            message=f"🚀 Memulai augmentasi untuk {len(classes_to_augment)} kelas dengan tracking per kelas",
            status="info", step=1
        )
        
        # Proses satu kelas pada satu waktu
        for i, class_id in enumerate(classes_to_augment):
            # Validasi data kelas
            if not (needed := class_needs.get(class_id, 0)) or not (files := class_files.get(class_id, [])): continue
            
            # Report dimulainya processing kelas
            self.logger.info(f"🔄 Memproses kelas {class_id} ({i+1}/{len(classes_to_augment)}): perlu {needed} instances")
            self.report_progress(message=f"Memproses kelas {class_id} ({i+1}/{len(classes_to_augment)})", 
                                status="info", step=1, current_progress=i, current_total=len(classes_to_augment), class_id=class_id)
            
            # Pilih file yang akan diaugmentasi untuk kelas ini
            files_to_augment = random.sample(files, min(len(files), needed)) if len(files) > needed else files
            
            # Augmentasi dengan multiprocessing atau sequential berdasarkan jumlah file dan worker
            class_results = self._execute_augmentation(
                files_to_augment, augmentation_params.copy(), 
                n_workers, f"Augmentasi kelas {class_id}",
                class_id=class_id, class_idx=i, total_classes=len(classes_to_augment)
            )
            
            # Proses hasil kelas
            generated_for_class = sum(result.get('generated', 0) for result in class_results)
            success_for_class = sum(1 for result in class_results if result.get('status') == 'success')
            
            # Update statistik per kelas dan total
            result_stats['class_stats'][class_id] = {
                'original': len(files), 'files_augmented': len(files_to_augment), 'target': target_count,
                'generated': generated_for_class, 'variations_per_file': generated_for_class / max(1, len(files_to_augment)),
                'success_rate': success_for_class / max(1, len(files_to_augment))
            }
            result_stats['total_augmented'] += len(files_to_augment)
            result_stats['total_generated'] += generated_for_class
            
            # Report completiton kelas
            self.logger.info(f"✅ Kelas {class_id} selesai: {generated_for_class} variasi dibuat dari {len(files_to_augment)} file")
            self.report_progress(
                message=f"✅ Kelas {class_id} selesai: {generated_for_class} variasi dibuat", status="success",
                step=1, current_progress=i+1, current_total=len(classes_to_augment), class_id=class_id
            )
        
        # Finalisasi hasil dan pindahkan file jika perlu
        result_stats.update(self._finalize_augmentation(
            result_stats, augmentation_params, paths, split, augmentation_types, 
            class_counts=class_counts, start_time=start_time
        ))
        
        return result_stats
    
    def _augment_without_class_tracking(
        self, image_files: List[str], augmentation_params: Dict, num_workers: int,
        paths: Dict, split: str, augmentation_types: List[str], start_time: float
    ) -> Dict[str, Any]:
        """Augmentasi tanpa tracking per kelas dengan pendekatan DRY."""
        # Setup tracking dan statistik
        n_workers, total_files = num_workers if num_workers is not None else self.num_workers, len(image_files)
        result_stats = {'total_augmented': 0, 'total_generated': 0, 'failed': 0, 'success': True}
        
        # Report mulai processing
        self.report_progress(message=f"🚀 Memulai augmentasi {total_files} file", status="info", step=1)
        
        # Augmentasi dengan multiprocessing atau sequential berdasarkan jumlah file dan worker
        all_results = self._execute_augmentation(image_files, augmentation_params, n_workers, "Augmentasi")
        
        # Proses semua hasil
        successful_results = [result for result in all_results if result.get('status') == 'success']
        result_stats['total_augmented'] = len(successful_results)
        result_stats['total_generated'] = sum(result.get('generated', 0) for result in all_results)
        
        # Finalisasi hasil dan pindahkan file jika perlu
        result_stats.update(self._finalize_augmentation(
            result_stats, augmentation_params, paths, split, augmentation_types, 
            total_files=total_files, start_time=start_time
        ))
        
        return result_stats
    
    def _execute_augmentation(
        self, files: List[str], params: Dict, n_workers: int, desc: str, 
        class_id: str = None, class_idx: int = None, total_classes: int = None
    ) -> List[Dict]:
        """Eksekusi augmentasi dengan single process atau multiprocessing sesuai kebutuhan."""
        # Jika class_id diberikan, tambahkan ke parameter
        if class_id: params['class_id'] = class_id
        
        # One-liner conditional execution berdasarkan jumlah file dan worker
        return (
            # Single process untuk file sedikit atau worker sedikit
            [self._process_single_file_with_progress(i, file, params, len(files), desc, class_id, class_idx, total_classes) 
             for i, file in enumerate(tqdm(files, desc=desc))]
            if len(files) == 1 or n_workers <= 1 else
            # Multiprocessing untuk file banyak dan worker banyak
            self._process_files_with_multiprocessing(files, params, n_workers, desc, class_id, class_idx, total_classes)
        )
    
    def _process_single_file_with_progress(
        self, idx: int, file_path: str, params: Dict, total: int, 
        desc: str, class_id: str = None, class_idx: int = None, total_classes: int = None
    ) -> Dict:
        """Proses satu file dengan progress reporting."""
        # Proses file
        result = process_single_file(file_path, **params)
        
        # Update progress dengan throttling (hanya setiap 10% atau file terakhir)
        if idx % max(1, total // 10) == 0 or idx == total - 1:
            progress_args = {
                'progress': idx+1, 'total': total,
                'message': f"{desc}: {idx+1}/{total} file",
                'status': "info", 'step': 1
            }
            
            # Tambahkan informasi kelas jika tersedia
            if class_id and class_idx is not None and total_classes is not None:
                progress_args.update({
                    'current_progress': idx+1, 'current_total': total,
                    'class_id': class_id, 'class_idx': class_idx, 'total_classes': total_classes
                })
                
            self.report_progress(**progress_args)
            
        return result
    
    def _process_files_with_multiprocessing(
        self, files: List[str], params: Dict, n_workers: int, desc: str,
        class_id: str = None, class_idx: int = None, total_classes: int = None
    ) -> List[Dict]:
        """Proses multiple file dengan multiprocessing dan progress reporting."""
        results, total_files = [], len(files)
        
        # Gunakan ProcessPoolExecutor untuk multiprocessing
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # Submit semua tugas
            futures = {executor.submit(process_single_file, file_path, **params): file_path for file_path in files}
            
            # Proses hasil selesai dan report progress
            for i, future in enumerate(tqdm(as_completed(futures), total=total_files, desc=desc)):
                try:
                    results.append(future.result())
                except Exception as e:
                    self.logger.error(f"❌ Error saat memproses {futures[future]}: {str(e)}")
                
                # Update progress dengan throttling (hanya setiap 10% atau file terakhir)
                if i % max(1, total_files // 10) == 0 or i == total_files - 1:
                    percentage = int((i+1) / total_files * 100)
                    progress_args = {
                        'progress': i+1, 'total': total_files,
                        'message': f"{desc} ({percentage}%): {i+1}/{total_files} file",
                        'status': "info", 'step': 1
                    }
                    
                    # Tambahkan informasi kelas jika tersedia
                    if class_id and class_idx is not None and total_classes is not None:
                        progress_args.update({
                            'current_progress': i+1, 'current_total': total_files,
                            'class_id': class_id, 'class_idx': class_idx, 'total_classes': total_classes
                        })
                        
                    self.report_progress(**progress_args)
        
        return results
    
    def _finalize_augmentation(
        self, result_stats: Dict, augmentation_params: Dict, paths: Dict, 
        split: str, augmentation_types: List[str], total_files: int = None, 
        class_counts: Dict = None, start_time: float = None
    ) -> Dict:
        """Finalisasi hasil augmentasi dan pindahkan file jika perlu."""
        # Hitung durasi
        duration = time.time() - (start_time or 0)
        result_stats['duration'] = duration
        
        # Pindahkan file ke preprocessed jika diminta
        self.report_progress(
            message=f"🔄 Memindahkan {result_stats['total_generated']} file ke direktori preprocessed",
            status="info", step=2
        )
        
        # Pindahkan file dengan menggunakan fungsi bantuan (DRY)
        move_success = move_files_to_preprocessed(
            paths['images_output_dir'], paths['labels_output_dir'],
            augmentation_params['output_prefix'], paths['final_output_dir'],
            split, self.logger
        )
        
        # Update path output dan log status
        result_stats['final_output_dir'] = paths['final_output_dir'] if move_success else paths['output_dir']
        self.logger.info(f"{'✅ File augmentasi berhasil dipindahkan ke' if move_success else '⚠️ Gagal memindahkan file augmentasi ke'} {paths['final_output_dir']}/{split}")
        
        # Log statistik final
        summary_message = f"✅ Augmentasi selesai dalam {duration:.2f} detik: {result_stats['total_generated']} variasi dihasilkan"
        self.logger.info(summary_message)
        self.report_progress(message=summary_message, status="success", step=2)
        
        # Tambahkan info tambahan ke hasil
        result_stats.update({
            'original': sum(class_counts.values()) if class_counts else (total_files or 0),
            'generated': result_stats['total_generated'],
            'augmentation_types': augmentation_types,
            'status': 'success',
            'split': split,
            'output_dir': paths['output_dir'],
            'preprocessed_dir': paths['preprocessed_dir']
        })
        
        return result_stats