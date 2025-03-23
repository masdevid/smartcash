"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan utama untuk augmentasi dataset dengan dukungan multi-processing, balancing class dan tracking progres per kelas
"""

import os
import time
import glob
import shutil
import random
import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import albumentations as A
from tqdm.auto import tqdm
import multiprocessing
from collections import defaultdict

from smartcash.common.logger import get_logger
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.services.augmentor.bbox_augmentor import BBoxAugmentor
from smartcash.dataset.services.augmentor.class_balancer import ClassBalancer
from smartcash.dataset.services.augmentor.augmentation_worker import process_single_file

# Import utils yang telah dipindahkan
from smartcash.dataset.utils.augmentor_utils import move_files_to_preprocessed

class AugmentationService:
    """
    Layanan augmentasi dataset dengan dukungan multiprocessing, balancing class dan tracking progres per kelas.
    Implementasi sesuai SRP dengan delegasi tugas ke worker dan balancer.
    """
    
    def __init__(self, config: Dict = None, data_dir: str = 'data', logger=None, num_workers: int = None):
        """
        Inisialisasi AugmentationService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Path direktori data
            logger: Logger kustom
            num_workers: Jumlah worker untuk multiprocessing (otomatis jika None)
        """
        self.config = config or {}
        self.data_dir = data_dir
        self.logger = logger or get_logger("augmentation_service")
        
        # Override num_workers dari config jika tidak disediakan secara eksplisit
        if num_workers is None:
            self.num_workers = self.config.get('augmentation', {}).get('num_workers', 4)
        else:
            self.num_workers = num_workers
            
        # Log nilai num_workers yang dipakai
        self.logger.debug(f"🔧 Menggunakan {self.num_workers} worker untuk augmentasi")
        
        # Inisialisasi pipeline factory
        self.pipeline_factory = AugmentationPipelineFactory(self.config, self.logger)
        self.bbox_augmentor = BBoxAugmentor(self.config, self.logger)
        self.class_balancer = ClassBalancer(self.config, self.logger)
        
        # Tanda untuk progress tracking
        self._stop_signal = False
        self._progress_callbacks = []
    
    def register_progress_callback(self, callback: Callable) -> None:
        """Register callback untuk progress tracking."""
        if callback and callable(callback):
            self._progress_callbacks.append(callback)
    
    def report_progress(self, progress: int = None, total: int = None, message: str = None, 
                       status: str = 'info', **kwargs) -> None:
        """Laporkan progress dengan callback."""
        for callback in self._progress_callbacks:
            try:
                # Menghindari duplikasi parameter
                if 'current_progress' in kwargs and 'current_total' in kwargs:
                    # Jangan gunakan current_total jika sudah diberikan dalam kwargs
                    current_progress = kwargs.pop('current_progress', None)
                    if current_progress is not None:
                        kwargs['current_progress'] = current_progress
                        
                callback(progress=progress, total=total, message=message, status=status, **kwargs)
            except Exception as e:
                self.logger.warning(f"⚠️ Error pada progress callback: {str(e)}")
    
    def augment_dataset(
        self,
        split: str = 'train',
        augmentation_types: List[str] = None,
        num_variations: int = 2,
        output_prefix: str = 'aug',
        validate_results: bool = True,
        resume: bool = False,
        process_bboxes: bool = True,
        target_balance: bool = False,
        num_workers: int = None,
        move_to_preprocessed: bool = True,
        target_count: int = 1000
    ) -> Dict[str, Any]:
        """
        Augmentasi dataset dengan penggunaan multiprocessing, balancing class dan tracking progres per kelas.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            augmentation_types: List jenis augmentasi ('position', 'lighting', 'combined', dll)
            num_variations: Jumlah variasi per gambar
            output_prefix: Prefix untuk file output
            validate_results: Validasi hasil augmentasi
            resume: Lanjutkan augmentasi sebelumnya
            process_bboxes: Augmentasi bbox juga
            target_balance: Coba balancing jumlah sampel per kelas
            num_workers: Override num_workers untuk proses ini
            move_to_preprocessed: Pindahkan hasil ke direktori preprocessed
            target_count: Jumlah target sampel per kelas (default: 1000)
            
        Returns:
            Dictionary berisi hasil augmentasi
        """
        # Reset stop signal
        self._stop_signal = False
        
        # Dapatkan lokasi sumber dan tujuan
        start_time = time.time()
        
        # Dapatkan paths dari config, dengan fallback ke default
        preprocessed_dir = self.config.get('preprocessing', {}).get('preprocessed_dir', 'data/preprocessed')
        augmented_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')

        # Path untuk source data
        input_dir = os.path.join(preprocessed_dir, split)
        images_input_dir = os.path.join(input_dir, 'images')
        labels_input_dir = os.path.join(input_dir, 'labels')

        # Path untuk output data
        output_dir = augmented_dir
        images_output_dir = os.path.join(output_dir, 'images')
        labels_output_dir = os.path.join(output_dir, 'labels')

        # Path untuk hasil akhir (jika move_to_preprocessed)
        final_output_dir = preprocessed_dir

        # Pastikan direktori ada
        os.makedirs(images_output_dir, exist_ok=True)
        os.makedirs(labels_output_dir, exist_ok=True)

        # Dapatkan daftar file input dengan file pattern yang sesuai
        file_prefix = self.config.get('preprocessing', {}).get('file_prefix', 'rp')
        file_pattern = f"{file_prefix}_*.jpg"
        image_files = glob.glob(os.path.join(images_input_dir, file_pattern))

        if not image_files:
            message = f"Tidak ada file gambar ditemukan dengan pola {file_pattern} di direktori {images_input_dir}"
            self.logger.warning(f"⚠️ {message}")
            return {"status": "error", "message": message}

        # Dictionary untuk menyimpan data per kelas untuk balancing dan progres tracking
        class_files = defaultdict(list)
        class_counts = defaultdict(int)
        class_augmentation_needs = {}

        # PERBAIKAN: Memindahkan persiapan balancing ke blok terpisah untuk konsolidasi log
        if target_balance:
            # Report progress untuk tahap analisis
            self.report_progress(
                message=f"🔍 Menganalisis peta distribusi kelas untuk balancing",
                status="info",
                step=0  # Tahap persiapan
            )
            
            # Gunakan class balancer untuk mendapatkan file yang perlu diaugmentasi
            try:
                result = self.class_balancer.prepare_balanced_dataset(
                    image_files=image_files,
                    labels_dir=labels_input_dir,
                    target_count=target_count,
                    filter_single_class=True  # Filter untuk file yang hanya memiliki 1 class
                )
                
                # Gunakan file yang direkomendasikan untuk balancing
                balanced_files = result.get('selected_files', [])
                class_stats = result.get('class_counts', {})
                class_augmentation_needs = result.get('augmentation_needs', {})
                
                # PERBAIKAN: Struktur ulang data per kelas untuk progres tracking
                for class_id, files in result.get('class_files', {}).items():
                    class_files[class_id] = files
                    class_counts[class_id] = len(files)
                
                # PERBAIKAN: Konsolidasi log kebutuhan augmentasi per kelas
                classes_needing_augmentation = [class_id for class_id, needed in class_augmentation_needs.items() if needed > 0]
                total_needed = sum(class_augmentation_needs.values())
                
                # Log rangkuman statistik balancing dengan konsolidasi
                summary_message = f"📊 Statistik Balancing Kelas (target: {target_count}/kelas): {len(classes_needing_augmentation)} kelas perlu ditambah {total_needed} sampel"
                self.logger.info(summary_message)
                self.report_progress(message=summary_message, status="info", step=0)
                
                # Update file yang akan diproses
                if balanced_files:
                    self.logger.info(f"🔄 Menggunakan {len(balanced_files)} dari {len(image_files)} file untuk balancing class")
                    image_files = balanced_files
                else:
                    self.logger.warning("⚠️ Tidak ada file yang cocok untuk balancing, menggunakan semua file")
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat balancing class: {str(e)}. Menggunakan semua file.")

        total_files = len(image_files)
        if total_files == 0:
            message = "Tidak ada file yang valid untuk diaugmentasi"
            self.logger.warning(f"⚠️ {message}")
            return {"status": "error", "message": message}

        # Log jumlah file
        self.logger.info(f"🔍 Ditemukan {total_files} file untuk augmentasi")

        # Cek resume jika diminta
        if resume:
            self.logger.info(f"⏩ Resume augmentation belum diimplementasikan, melanjutkan dengan semua file")

        # Setup augmentation pipeline berdasarkan jenis yang diminta
        if not augmentation_types:
            augmentation_types = ['combined']
            
        # Inisialisasi pipeline augmentasi
        try:
            pipeline = self.pipeline_factory.create_pipeline(
                augmentation_types=augmentation_types,
                img_size=(640, 640),
                include_normalize=False,
                intensity=1.0
            )
            
            self.logger.info(f"✅ Pipeline augmentasi berhasil dibuat: {', '.join(augmentation_types)}")
        except Exception as e:
            message = f"Error membuat pipeline augmentasi: {str(e)}"
            self.logger.error(f"❌ {message}")
            return {"status": "error", "message": message}

        # Siapkan parameter untuk worker
        augmentation_params = {
            'pipeline': pipeline,
            'num_variations': num_variations,
            'output_prefix': output_prefix,
            'process_bboxes': process_bboxes,
            'validate_results': validate_results,
            'bbox_augmentor': self.bbox_augmentor,
            'labels_input_dir': labels_input_dir,
            'images_output_dir': images_output_dir,
            'labels_output_dir': labels_output_dir,
        }

        # PERBAIKAN: Proses augmentasi per kelas jika target_balance aktif
        if target_balance and class_files:
            return self._augment_with_class_tracking(
                class_files, class_counts, class_augmentation_needs, 
                augmentation_params, target_count, num_workers, 
                move_to_preprocessed, final_output_dir, preprocessed_dir,
                output_dir, images_output_dir, labels_output_dir, split,
                augmentation_types, start_time
            )
        else:
            # Lanjutkan dengan augmentasi standar jika tidak ada target balancing
            return self._augment_without_class_tracking(
                image_files, augmentation_params, num_workers,
                move_to_preprocessed, final_output_dir, preprocessed_dir,
                output_dir, images_output_dir, labels_output_dir, split,
                augmentation_types, start_time
            )

    def _move_files_to_preprocessed(self, images_output_dir, labels_output_dir, 
                            output_prefix, final_output_dir, split):
        """Pindahkan file hasil augmentasi ke direktori preprocessed."""
        return move_files_to_preprocessed(
            images_output_dir, labels_output_dir, output_prefix,
            final_output_dir, split, self.logger
        )
    def _augment_with_class_tracking(
        self,
        class_files,
        class_counts,
        class_augmentation_needs,
        augmentation_params,
        target_count,
        num_workers,
        move_to_preprocessed,
        final_output_dir,
        preprocessed_dir,
        output_dir,
        images_output_dir,
        labels_output_dir,
        split,
        augmentation_types,
        start_time
    ) -> Dict[str, Any]:
        """
        Augmentasi dataset dengan tracking progress per kelas untuk balancing.
        
        Args:
            class_files: Dictionary mapping kelas ke file
            class_counts: Dictionary jumlah file per kelas
            class_augmentation_needs: Dictionary kebutuhan augmentasi per kelas
            augmentation_params: Parameter untuk worker augmentasi
            target_count: Target jumlah sampel per kelas
            num_workers: Jumlah worker untuk multiprocessing
            move_to_preprocessed: Pindahkan hasil ke preprocessed
            final_output_dir: Direktori target akhir
            preprocessed_dir: Direktori preprocessed
            output_dir: Direktori output sementara
            images_output_dir: Direktori output gambar
            labels_output_dir: Direktori output label
            split: Split dataset
            augmentation_types: Jenis augmentasi
            start_time: Waktu mulai augmentasi
            
        Returns:
            Dictionary hasil augmentasi dengan statistik per kelas
        """
        # Override num_workers jika diberikan
        n_workers = num_workers if num_workers is not None else self.num_workers
        
        # Jumlah kelas yang perlu diaugmentasi
        classes_to_augment = [cls for cls, need in class_augmentation_needs.items() if need > 0]
        total_classes = len(classes_to_augment)
        
        # Statisik hasil augmentasi
        result_stats = {
            'total_augmented': 0,
            'total_generated': 0,
            'failed': 0,
            'class_stats': {},
            'success': True
        }
        
        # Report start of processing
        self.report_progress(
            message=f"🚀 Memulai augmentasi untuk {total_classes} kelas dengan tracking per kelas",
            status="info",
            step=1  # Tahap processing
        )
        
        # Proses satu kelas pada satu waktu
        for i, class_id in enumerate(classes_to_augment):
            needed = class_augmentation_needs[class_id]
            files = class_files[class_id]
            
            if not files or needed <= 0:
                continue
                
            # Report class processing start
            class_message = f"Memproses kelas {class_id} ({i+1}/{total_classes}): perlu {needed} instances"
            self.logger.info(f"🔄 {class_message}")
            self.report_progress(
                message=class_message,
                status="info",
                step=1,  # Tahap processing
                current_progress=i,
                current_total=total_classes,
                class_id=class_id
            )
            
            # Pilih file yang akan diaugmentasi untuk kelas ini
            files_to_augment = files
            if len(files) > needed:
                files_to_augment = random.sample(files, min(len(files), needed))
            
            # Augmentasi file untuk kelas ini dengan multiprocessing
            class_results = []
            
            # Jika hanya 1 file atau worker, gunakan processing sederhana
            if len(files_to_augment) == 1 or n_workers <= 1:
                for file_path in tqdm(files_to_augment, desc=f"Augmentasi kelas {class_id}"):
                    # Tambahkan class_id ke parameter augmentasi
                    params = augmentation_params.copy()
                    params['class_id'] = class_id
                    
                    # Proses file
                    result = process_single_file(file_path, **params)
                    class_results.append(result)
                    
                    # Update progress
                    self.report_progress(
                        message=f"Memproses file {file_path.split('/')[-1]} untuk kelas {class_id}",
                        status="info",
                        step=1,  # Tahap processing
                        current_progress=i,
                        current_total=total_classes,
                        class_id=class_id
                    )
            else:
                # Gunakan multiprocessing untuk augmentasi
                with ProcessPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all tasks
                    futures = {}
                    for file_path in files_to_augment:
                        # Tambahkan class_id ke parameter augmentasi
                        params = augmentation_params.copy()
                        params['class_id'] = class_id
                        
                        # Submit task
                        future = executor.submit(process_single_file, file_path, **params)
                        futures[future] = file_path
                    
                    # Process results as they complete
                    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc=f"Augmentasi kelas {class_id}")):
                        file_path = futures[future]
                        try:
                            result = future.result()
                            class_results.append(result)
                        except Exception as e:
                            self.logger.error(f"❌ Error saat memproses {file_path}: {str(e)}")
                            result_stats['failed'] += 1
                        
                        # Update progress per file with throttling (only every 10%)
                        if i % max(1, len(futures) // 10) == 0 or i == len(futures) - 1:
                            self.report_progress(
                                progress=i+1,
                                total=len(futures),
                                message=f"Augmentasi kelas {class_id}: {i+1}/{len(futures)} file",
                                status="info",
                                step=1,  # Tahap processing
                                current_progress=i+1,
                                current_total=len(futures),
                                class_id=class_id
                            )
            
            # Process class results
            generated_for_class = sum(result.get('generated', 0) for result in class_results)
            success_for_class = sum(1 for result in class_results if result.get('status') == 'success')
            
            # Update class statistics
            result_stats['class_stats'][class_id] = {
                'original': len(files),
                'files_augmented': len(files_to_augment),
                'target': target_count,
                'generated': generated_for_class,
                'variations_per_file': generated_for_class / max(1, len(files_to_augment)),
                'success_rate': success_for_class / max(1, len(files_to_augment))
            }
            
            # Update total statistics
            result_stats['total_augmented'] += len(files_to_augment)
            result_stats['total_generated'] += generated_for_class
            
            # Report class completion
            class_complete_message = f"✅ Kelas {class_id} selesai: {generated_for_class} variasi dibuat dari {len(files_to_augment)} file"
            self.logger.info(class_complete_message)
            self.report_progress(
                message=class_complete_message,
                status="success",
                step=1,  # Tahap processing
                current_progress=i+1,
                current_total=total_classes,
                class_id=class_id
            )
        
        # Semua kelas selesai, finalisasi hasil
        duration = time.time() - start_time
        result_stats['duration'] = duration
        
        # Pindahkan file ke preprocessed jika diminta
        if move_to_preprocessed:
            self.report_progress(
                message=f"🔄 Memindahkan {result_stats['total_generated']} file ke direktori preprocessed",
                status="info",
                step=2  # Tahap finalisasi
            )
            
            # Pindahkan file dengan menggunakan fungsi bantuan
            move_success = move_files_to_preprocessed(
                images_output_dir, labels_output_dir,
                augmentation_params['output_prefix'], final_output_dir,
                split, self.logger
            )
            
            # Report status pemindahan
            if move_success:
                self.logger.info(f"✅ File augmentasi berhasil dipindahkan ke {final_output_dir}/{split}")
                result_stats['final_output_dir'] = final_output_dir
            else:
                self.logger.warning(f"⚠️ Gagal memindahkan file augmentasi ke {final_output_dir}/{split}")
                result_stats['final_output_dir'] = output_dir
        
        # Log final statistics
        summary_message = (
            f"✅ Augmentasi selesai dalam {duration:.2f} detik:\n"
            f"- {result_stats['total_augmented']} file diaugmentasi dari {total_classes} kelas\n"
            f"- {result_stats['total_generated']} variasi dihasilkan"
        )
        self.logger.info(summary_message)
        self.report_progress(
            message=f"✅ Augmentasi selesai dalam {duration:.2f} detik: {result_stats['total_generated']} variasi dihasilkan",
            status="success",
            step=2  # Tahap finalisasi
        )
        
        # Add additional info to result
        result_stats.update({
            'original': sum(class_counts.values()),
            'augmentation_types': augmentation_types,
            'status': 'success',
            'split': split,
            'output_dir': output_dir,
            'preprocessed_dir': preprocessed_dir
        })
        
        return result_stats

    def _augment_without_class_tracking(
        self,
        image_files,
        augmentation_params,
        num_workers,
        move_to_preprocessed,
        final_output_dir,
        preprocessed_dir,
        output_dir,
        images_output_dir,
        labels_output_dir,
        split,
        augmentation_types,
        start_time
    ) -> Dict[str, Any]:
        """
        Augmentasi dataset tanpa tracking progress per kelas.
        
        Args:
            image_files: List file gambar yang akan diaugmentasi
            augmentation_params: Parameter untuk worker augmentasi
            num_workers: Jumlah worker untuk multiprocessing
            move_to_preprocessed: Pindahkan hasil ke preprocessed
            final_output_dir: Direktori target akhir
            preprocessed_dir: Direktori preprocessed
            output_dir: Direktori output sementara
            images_output_dir: Direktori output gambar
            labels_output_dir: Direktori output label
            split: Split dataset
            augmentation_types: Jenis augmentasi
            start_time: Waktu mulai augmentasi
            
        Returns:
            Dictionary hasil augmentasi dengan statistik
        """
        # Override num_workers jika diberikan
        n_workers = num_workers if num_workers is not None else self.num_workers
        total_files = len(image_files)
        
        # Statistik hasil augmentasi
        result_stats = {
            'total_augmented': 0,
            'total_generated': 0,
            'failed': 0,
            'success': True
        }
        
        # Report start of processing
        self.report_progress(
            message=f"🚀 Memulai augmentasi {total_files} file",
            status="info",
            step=1  # Tahap processing
        )
        
        # Proses semua file, dengan atau tanpa multiprocessing
        all_results = []
        
        # Jika hanya 1 file atau worker, gunakan processing sederhana
        if total_files == 1 or n_workers <= 1:
            for i, file_path in enumerate(tqdm(image_files, desc="Augmentasi")):
                # Proses file
                result = process_single_file(file_path, **augmentation_params)
                all_results.append(result)
                
                # Update progress dengan throttling (hanya setiap 10%)
                if i % max(1, total_files // 10) == 0 or i == total_files - 1:
                    percentage = int((i+1) / total_files * 100)
                    self.report_progress(
                        progress=i+1,
                        total=total_files,
                        message=f"Augmentasi ({percentage}%): {i+1}/{total_files} file",
                        status="info",
                        step=1,  # Tahap processing
                        current_progress=i+1,
                        current_total=total_files
                    )
        else:
            # Gunakan multiprocessing untuk augmentasi
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit all tasks
                futures = {}
                for file_path in image_files:
                    future = executor.submit(process_single_file, file_path, **augmentation_params)
                    futures[future] = file_path
                
                # Process results as they complete
                for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="Augmentasi")):
                    file_path = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as e:
                        self.logger.error(f"❌ Error saat memproses {file_path}: {str(e)}")
                        result_stats['failed'] += 1
                    
                    # Update progress per file dengan throttling (hanya setiap 10%)
                    if i % max(1, total_files // 10) == 0 or i == total_files - 1:
                        percentage = int((i+1) / total_files * 100)
                        self.report_progress(
                            progress=i+1,
                            total=total_files,
                            message=f"Augmentasi ({percentage}%): {i+1}/{total_files} file",
                            status="info",
                            step=1,  # Tahap processing
                            current_progress=i+1,
                            current_total=total_files
                        )
        
        # Process all results
        successful_results = [result for result in all_results if result.get('status') == 'success']
        result_stats['total_augmented'] = len(successful_results)
        result_stats['total_generated'] = sum(result.get('generated', 0) for result in all_results)
        
        # Finalisasi hasil
        duration = time.time() - start_time
        result_stats['duration'] = duration
        
        # Pindahkan file ke preprocessed jika diminta
        if move_to_preprocessed:
            self.report_progress(
                message=f"🔄 Memindahkan {result_stats['total_generated']} file ke direktori preprocessed",
                status="info",
                step=2  # Tahap finalisasi
            )
            
            # Pindahkan file dengan menggunakan fungsi bantuan
            move_success = move_files_to_preprocessed(
                images_output_dir, labels_output_dir,
                augmentation_params['output_prefix'], final_output_dir,
                split, self.logger
            )
            
            # Report status pemindahan
            if move_success:
                self.logger.info(f"✅ File augmentasi berhasil dipindahkan ke {final_output_dir}/{split}")
                result_stats['final_output_dir'] = final_output_dir
            else:
                self.logger.warning(f"⚠️ Gagal memindahkan file augmentasi ke {final_output_dir}/{split}")
                result_stats['final_output_dir'] = output_dir
        
        # Log final statistics
        summary_message = (
            f"✅ Augmentasi selesai dalam {duration:.2f} detik:\n"
            f"- {result_stats['total_augmented']} file diaugmentasi\n"
            f"- {result_stats['total_generated']} variasi dihasilkan"
        )
        self.logger.info(summary_message)
        self.report_progress(
            message=f"✅ Augmentasi selesai dalam {duration:.2f} detik: {result_stats['total_generated']} variasi dihasilkan",
            status="success",
            step=2  # Tahap finalisasi
        )
        
        # Add additional info to result
        result_stats.update({
            'original': total_files,
            'generated': result_stats['total_generated'],
            'augmentation_types': augmentation_types,
            'status': 'success',
            'split': split,
            'output_dir': output_dir,
            'preprocessed_dir': preprocessed_dir
        })
        
        return result_stats
    