"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan augmentasi dataset dengan pendekatan one-liner, optimasi multiprocessing, dan pencegahan duplikasi parameter
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
    """Layanan augmentasi dataset dengan dukungan multiprocessing, balancing class dan tracking progres per kelas."""
    
    def __init__(self, config: Dict = None, data_dir: str = 'data', logger=None, num_workers: int = None):
        """Inisialisasi AugmentationService dengan parameter utama."""
        self.config, self.data_dir, self.logger = config or {}, data_dir, logger or get_logger("augmentation_service")
        self.num_workers = num_workers if num_workers is not None else self.config.get('augmentation', {}).get('num_workers', 4)
        self.logger.debug(f"🔧 Menggunakan {self.num_workers} worker untuk augmentasi")
        
        # Inisialisasi komponen-komponen utama dengan one-liner
        self.pipeline_factory, self.bbox_augmentor, self.class_balancer = AugmentationPipelineFactory(self.config, self.logger), BBoxAugmentor(self.config, self.logger), ClassBalancer(self.config, self.logger)
        self._stop_signal, self._progress_callbacks = False, []
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback untuk progress tracking."""
        if callback and callable(callback): self._progress_callbacks.append(callback)
    
    def report_progress(self, progress: int = None, total: int = None, message: str = None, status: str = 'info', **kwargs) -> None:
        """Laporkan progress dengan callback dan hindari duplikasi parameter."""
        for callback in self._progress_callbacks:
            try:
                # Buat params bersih dengan one-liner
                explicit_params = ['progress', 'total', 'message', 'status', 'current_progress', 'current_total', 'class_id']
                filtered_kwargs = {k: v for k, v in kwargs.items() if k not in explicit_params}
                
                # Gabungkan parameter yang ada nilainya dengan one-liner
                params = {k: v for k, v in {
                    'message': message, 'status': status, 
                    'progress': progress, 'total': total,
                    'current_progress': kwargs.get('current_progress'), 
                    'current_total': kwargs.get('current_total'),
                    'class_id': kwargs.get('class_id')
                }.items() if v is not None}
                
                # Gabungkan dan panggil callback dengan one-liner
                params.update(filtered_kwargs)
                callback(**params)
            except Exception as e: self.logger.warning(f"⚠️ Error pada progress callback: {str(e)}")
    
    def augment_dataset(
        self, split: str = 'train', augmentation_types: List[str] = None, num_variations: int = 2,
        output_prefix: str = 'aug', validate_results: bool = True, resume: bool = False,
        process_bboxes: bool = True, target_balance: bool = False, num_workers: int = None,
        move_to_preprocessed: bool = True, target_count: int = 1000) -> Dict[str, Any]:
        """Augmentasi dataset dengan pendekatan one-liner dan optimasi multiprocessing."""
        # Reset stop signal dan setup paths dengan one-liner
        self._stop_signal, start_time = False, time.time()
        paths = self._setup_paths(split)
        
        # Buat direktori output dan dapatkan file input dengan one-liner
        [os.makedirs(paths[key], exist_ok=True) for key in ['images_output_dir', 'labels_output_dir']]
        file_prefix = self.config.get('preprocessing', {}).get('file_prefix', 'rp')
        image_files = glob.glob(os.path.join(paths['images_input_dir'], f"{file_prefix}_*.jpg"))
        
        # Validasi input dengan early return
        if not image_files:
            message = f"Tidak ada file gambar ditemukan dengan pola {file_prefix}_*.jpg di direktori {paths['images_input_dir']}"
            self.logger.warning(f"⚠️ {message}")
            return {"status": "error", "message": message}
        
        # Persiapkan balancing dengan one-liner
        class_data = {'files': defaultdict(list), 'counts': defaultdict(int), 'needs': {}}
        if target_balance: class_data = self._prepare_balancing(image_files, paths, target_count) or class_data
        
        # Update file yang akan diproses berdasarkan balancing
        balanced_files = class_data.get('selected_files', []) if target_balance else []
        image_files = balanced_files if balanced_files else image_files
        if len(image_files) == 0: return {"status": "error", "message": "Tidak ada file yang valid untuk diaugmentasi"}
        
        # Log info dan setup pipeline dengan one-liner
        self.logger.info(f"🔍 Ditemukan {len(image_files)} file untuk augmentasi")
        if resume: self.logger.info(f"⏩ Resume augmentation belum diimplementasikan, melanjutkan dengan semua file")
        try: pipeline = self._create_pipeline(augmentation_types)
        except Exception as e: return {"status": "error", "message": f"Error membuat pipeline augmentasi: {str(e)}"}
        
        # Setup parameter augmentasi dengan one-liner
        augmentation_params = {
            'pipeline': pipeline, 'num_variations': num_variations, 'output_prefix': output_prefix,
            'process_bboxes': process_bboxes, 'validate_results': validate_results, 'bbox_augmentor': self.bbox_augmentor,
            'labels_input_dir': paths['labels_input_dir'], 'images_output_dir': paths['images_output_dir'],
            'labels_output_dir': paths['labels_output_dir']
        }
        
        # Eksekusi augmentasi berdasarkan balancing dengan ternary one-liner
        results = (self._augment_with_class_tracking(class_data, augmentation_params, target_count, num_workers, paths, split, augmentation_types or ['combined'], start_time) 
                  if target_balance and class_data.get('files') else 
                  self._augment_without_class_tracking(image_files, augmentation_params, num_workers, paths, split, augmentation_types or ['combined'], start_time))
        
        # Tambahkan info move_to_preprocessed dan return
        results['move_to_preprocessed'] = move_to_preprocessed
        return results
    
    def _setup_paths(self, split: str) -> Dict[str, str]:
        """Setup paths dengan one-liner style."""
        # Base paths dengan one-liner
        paths = {
            'preprocessed_dir': self.config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed'),
            'augmented_dir': self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
        }
        
        # Derived paths dengan one-liner
        return {**paths, **{
            'input_dir': os.path.join(paths['preprocessed_dir'], split),
            'images_input_dir': os.path.join(paths['preprocessed_dir'], split, 'images'),
            'labels_input_dir': os.path.join(paths['preprocessed_dir'], split, 'labels'),
            'output_dir': paths['augmented_dir'],
            'images_output_dir': os.path.join(paths['augmented_dir'], 'images'),
            'labels_output_dir': os.path.join(paths['augmented_dir'], 'labels'),
            'final_output_dir': paths['preprocessed_dir']
        }}
    
    def _create_pipeline(self, augmentation_types: List[str] = None) -> A.Compose:
        """Buat pipeline augmentasi dengan one-liner."""
        pipeline = self.pipeline_factory.create_pipeline(
            augmentation_types=augmentation_types or ['combined'],
            img_size=(640, 640),
            include_normalize=False,
            intensity=1.0
        )
        self.logger.info(f"✅ Pipeline augmentasi berhasil dibuat: {', '.join(augmentation_types or ['combined'])}")
        return pipeline
    
    def _prepare_balancing(self, image_files: List[str], paths: Dict, target_count: int) -> Dict[str, Any]:
        """Persiapkan balancing kelas dengan one-liner."""
        try:
            # Report progress dan gunakan balancer dengan one-liner
            self.report_progress(message="🔍 Menganalisis peta distribusi kelas untuk balancing", status="info", step=0)
            result = self.class_balancer.prepare_balanced_dataset(
                image_files=image_files, labels_dir=paths['labels_input_dir'],
                target_count=target_count, filter_single_class=True, 
                progress_callback=lambda *args, **kwargs: self.report_progress(*args, **kwargs)
            )
            
            # Ekstrak dan konversi data dengan one-liner
            class_data = {
                'files': result.get('class_files', {}),
                'counts': {k: len(v) for k, v in result.get('class_files', {}).items()},
                'needs': result.get('augmentation_needs', {}),
                'selected_files': result.get('selected_files', [])
            }
            
            # Log statistik balancing dengan one-liner
            classes_needing, total_needed = sum(1 for v in class_data['needs'].values() if v > 0), sum(class_data['needs'].values())
            self.logger.info(f"📊 Statistik Balancing Kelas (target: {target_count}/kelas): {classes_needing} kelas perlu ditambah {total_needed} sampel")
            self.logger.info(f"🔄 Menggunakan {len(class_data['selected_files'])} file untuk augmentasi ({classes_needing} kelas)")
            
            return class_data
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat balancing class: {str(e)}. Menggunakan semua file.")
            return None
    
    def _augment_with_class_tracking(
        self, class_data: Dict, augmentation_params: Dict, target_count: int,
        num_workers: int, paths: Dict, split: str, augmentation_types: List[str], start_time: float
    ) -> Dict[str, Any]:
        """Augmentasi dengan tracking per kelas dengan one-liner."""
        # Ekstrak data class dan setup dengan one-liner
        class_files, class_counts, class_needs = class_data['files'], class_data['counts'], class_data['needs']
        classes_to_augment = [cls for cls, need in class_needs.items() if need > 0]
        n_workers = num_workers if num_workers is not None else self.num_workers
        result_stats = {'total_augmented': 0, 'total_generated': 0, 'failed': 0, 'class_stats': {}, 'success': True}
        
        # Report mulai proses dengan one-liner
        self.report_progress(message=f"🚀 Memulai augmentasi untuk {len(classes_to_augment)} kelas dengan tracking per kelas", status="info", step=1)
        
        # Proses satu kelas pada satu waktu dengan one-liner loop
        for i, class_id in enumerate(classes_to_augment):
            # Validasi dan prepare data kelas dengan one-liner
            if not (needed := class_needs.get(class_id, 0)) or not (files := class_files.get(class_id, [])): continue
            self.logger.info(f"🔄 Memproses kelas {class_id} ({i+1}/{len(classes_to_augment)}): perlu {needed} instances")
            self.report_progress(message=f"Memproses kelas {class_id} ({i+1}/{len(classes_to_augment)})", 
                                status="info", step=1, current_progress=i, current_total=len(classes_to_augment), class_id=class_id)
            
            # Pilih file yang akan diaugmentasi dengan one-liner
            files_to_augment = random.sample(files, min(len(files), needed)) if len(files) > needed else files
            
            # Augmentasi dan proses hasil dengan one-liner
            class_results = self._execute_augmentation(files_to_augment, augmentation_params.copy(), n_workers, 
                                                     f"Augmentasi kelas {class_id}", class_id=class_id, 
                                                     class_idx=i, total_classes=len(classes_to_augment))
            
            # Update statistik dengan one-liner
            generated_for_class, success_for_class = sum(r.get('generated', 0) for r in class_results), sum(1 for r in class_results if r.get('status') == 'success')
            result_stats['class_stats'][class_id] = {
                'original': len(files), 'files_augmented': len(files_to_augment), 'target': target_count,
                'generated': generated_for_class, 'variations_per_file': generated_for_class / max(1, len(files_to_augment)),
                'success_rate': success_for_class / max(1, len(files_to_augment))
            }
            result_stats['total_augmented'] += len(files_to_augment)
            result_stats['total_generated'] += generated_for_class
            
            # Report progress dengan one-liner
            self.logger.info(f"✅ Kelas {class_id} selesai: {generated_for_class} variasi dibuat dari {len(files_to_augment)} file")
            self.report_progress(message=f"✅ Kelas {class_id} selesai: {generated_for_class} variasi dibuat", status="success",
                               step=1, current_progress=i+1, current_total=len(classes_to_augment), class_id=class_id)
        
        # Finalisasi hasil dengan one-liner
        return {**result_stats, **self._finalize_augmentation(result_stats, augmentation_params, paths, split, 
                                                            augmentation_types, class_counts=class_counts, start_time=start_time)}
    
    def _augment_without_class_tracking(
        self, image_files: List[str], augmentation_params: Dict, num_workers: int,
        paths: Dict, split: str, augmentation_types: List[str], start_time: float
    ) -> Dict[str, Any]:
        """Augmentasi tanpa tracking per kelas dengan one-liner."""
        # Setup dan execute augmentasi dengan one-liner
        n_workers, total_files = num_workers if num_workers is not None else self.num_workers, len(image_files)
        result_stats = {'total_augmented': 0, 'total_generated': 0, 'failed': 0, 'success': True}
        self.report_progress(message=f"🚀 Memulai augmentasi {total_files} file", status="info", step=1)
        
        # Execute augmentasi dan proses hasil dengan one-liner
        all_results = self._execute_augmentation(image_files, augmentation_params, n_workers, "Augmentasi")
        result_stats['total_augmented'] = len([r for r in all_results if r.get('status') == 'success'])
        result_stats['total_generated'] = sum(r.get('generated', 0) for r in all_results)
        
        # Finalisasi hasil dengan one-liner
        return {**result_stats, **self._finalize_augmentation(result_stats, augmentation_params, paths, split, 
                                                           augmentation_types, total_files=total_files, start_time=start_time)}
    
    def _execute_augmentation(
        self, files: List[str], params: Dict, n_workers: int, desc: str, 
        class_id: str = None, class_idx: int = None, total_classes: int = None
    ) -> List[Dict]:
        """Eksekusi augmentasi dengan single process atau multiprocessing dalam one-liner."""
        # Tambahkan class_id ke params jika tersedia
        if class_id: params['class_id'] = class_id
        
        # Execute berdasarkan jumlah file dan worker dengan one-liner
        return ([self._process_single_file_with_progress(i, file, params, len(files), desc, class_id, class_idx, total_classes) 
                for i, file in enumerate(tqdm(files, desc=desc))] if len(files) == 1 or n_workers <= 1 
                else self._process_files_with_multiprocessing(files, params, n_workers, desc, class_id, class_idx, total_classes))
    
    def _process_single_file_with_progress(
        self, idx: int, file_path: str, params: Dict, total: int, 
        desc: str, class_id: str = None, class_idx: int = None, total_classes: int = None
    ) -> Dict:
        """Proses satu file dengan progress reporting dalam one-liner."""
        # Proses file dengan progress throttling dalam one-liner
        result = process_single_file(file_path, **params)
        if idx % max(1, total // 10) == 0 or idx == total - 1:
            progress_args = {'progress': idx+1, 'total': total, 'message': f"{desc}: {idx+1}/{total} file", 'status': "info", 'step': 1}
            if class_id is not None and class_idx is not None and total_classes is not None:
                progress_args.update({'current_progress': idx+1, 'current_total': total, 'class_id': class_id, 
                                    'class_idx': class_idx, 'total_classes': total_classes})
            self.report_progress(**progress_args)
        return result
    
    def _process_files_with_multiprocessing(
        self, files: List[str], params: Dict, n_workers: int, desc: str,
        class_id: str = None, class_idx: int = None, total_classes: int = None
    ) -> List[Dict]:
        """Proses multiple file dengan multiprocessing dalam one-liner."""
        # Setup dan hasil dengan one-liner
        results, total_files = [], len(files)
        
        # Submit dan process dengan one-liner dalam ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_single_file, file_path, **params): file_path for file_path in files}
            for i, future in enumerate(tqdm(as_completed(futures), total=total_files, desc=desc)):
                try: results.append(future.result())
                except Exception as e: self.logger.error(f"❌ Error saat memproses {futures[future]}: {str(e)}")
                
                # Report progress dengan throttling one-liner
                if i % max(1, total_files // 10) == 0 or i == total_files - 1:
                    percentage = int((i+1) / total_files * 100)
                    progress_args = {'progress': i+1, 'total': total_files, 'message': f"{desc} ({percentage}%): {i+1}/{total_files} file", 'status': "info", 'step': 1}
                    if class_id is not None and class_idx is not None and total_classes is not None:
                        progress_args.update({'current_progress': i+1, 'current_total': total_files, 'class_id': class_id, 
                                            'class_idx': class_idx, 'total_classes': total_classes})
                    self.report_progress(**progress_args)
        
        return results
    
    def _finalize_augmentation(
        self, result_stats: Dict, augmentation_params: Dict, paths: Dict, 
        split: str, augmentation_types: List[str], total_files: int = None, 
        class_counts: Dict = None, start_time: float = None
    ) -> Dict:
        """Finalisasi hasil augmentasi dengan one-liner."""
        # Hitung durasi dan update result dengan one-liner
        duration = time.time() - (start_time or 0)
        result_stats['duration'] = duration
        
        # Pindah file dengan one-liner
        self.report_progress(message=f"🔄 Memindahkan {result_stats['total_generated']} file ke direktori preprocessed", status="info", step=2)
        move_success = move_files_to_preprocessed(paths['images_output_dir'], paths['labels_output_dir'], 
                                               augmentation_params['output_prefix'], paths['final_output_dir'], split, self.logger)
        
        # Log hasil dengan one-liner
        result_stats['final_output_dir'] = paths['final_output_dir'] if move_success else paths['output_dir']
        self.logger.info(f"{'✅ File augmentasi berhasil dipindahkan ke' if move_success else '⚠️ Gagal memindahkan file augmentasi ke'} {paths['final_output_dir']}/{split}")
        
        # Report summary dan tambah info dengan one-liner
        summary_message = f"✅ Augmentasi selesai dalam {duration:.2f} detik: {result_stats['total_generated']} variasi dihasilkan"
        self.logger.info(summary_message)
        self.report_progress(message=summary_message, status="success", step=2)
        
        # Return hasil dengan one-liner
        return {
            'original': sum(class_counts.values()) if class_counts else (total_files or 0),
            'generated': result_stats['total_generated'],
            'augmentation_types': augmentation_types,
            'status': 'success',
            'split': split,
            'output_dir': paths['output_dir'],
            'preprocessed_dir': paths['preprocessed_dir']
        }