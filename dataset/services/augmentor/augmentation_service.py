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
    
    def _augment_with_class_tracking(
        self, 
        class_files: Dict[str, List[str]], 
        class_counts: Dict[str, int],
        class_augmentation_needs: Dict[str, int],
        augmentation_params: Dict[str, Any],
        target_count: int,
        num_workers: Optional[int],
        move_to_preprocessed: bool,
        final_output_dir: str,
        preprocessed_dir: str,
        output_dir: str,
        images_output_dir: str,
        labels_output_dir: str,
        split: str,
        augmentation_types: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Proses augmentasi dengan tracking progress per kelas untuk balancing.
        
        Args:
            class_files: Dictionary file per kelas
            class_counts: Jumlah file asli per kelas
            class_augmentation_needs: Jumlah file yang perlu ditambahkan per kelas
            augmentation_params: Parameter augmentasi
            target_count: Target jumlah instance per kelas
            num_workers: Jumlah worker
            move_to_preprocessed: Apakah hasil dipindahkan ke direktori preprocessed
            final_output_dir: Direktori akhir hasil
            preprocessed_dir: Direktori preprocessed
            output_dir: Direktori output sementara
            images_output_dir: Direktori output gambar
            labels_output_dir: Direktori output label
            split: Split dataset
            augmentation_types: Jenis augmentasi
            start_time: Waktu mulai proses
            
        Returns:
            Dictionary hasil augmentasi
        """
        # Persiapkan tracking progres
        self.logger.info(f"🚀 Memulai augmentasi dengan tracking per kelas ({len(class_files)} kelas)")
        
        # Filter hanya kelas yang membutuhkan augmentasi
        classes_to_augment = [cls_id for cls_id, needed in class_augmentation_needs.items() if needed > 0]
        
        # Tampilkan progress awal
        self.report_progress(
            0, len(classes_to_augment),  # Overall progress: jumlah kelas yang diproses
            f"Memulai augmentasi {len(classes_to_augment)} kelas dengan {len(augmentation_types)} jenis augmentasi",
            status="info",
            step=1,  # Tahap proses utama
            total_classes=len(classes_to_augment)
        )
        
        # Simpan statistik dan hasil
        total_augmented_count = 0
        total_generated = 0
        total_errors = 0
        result_by_class = {}
        
        # Batasi worker sesuai kebutuhan
        use_workers = num_workers if num_workers is not None else self.num_workers
        max_workers = min(use_workers, multiprocessing.cpu_count(), 16)
        
        # Proses setiap kelas yang membutuhkan augmentasi
        for class_idx, class_id in enumerate(classes_to_augment):
            # Jika signal stop, hentikan proses
            if self._stop_signal:
                self.logger.warning("⚠️ Augmentasi dihentikan oleh pengguna")
                return {"status": "cancelled", "message": "Augmentasi dihentikan oleh pengguna"}
                
            # Ambil file untuk kelas ini dan kebutuhan augmentasi
            files_for_class = class_files.get(class_id, [])
            needed_augmentations = class_augmentation_needs.get(class_id, 0)
            current_count = class_counts.get(class_id, 0)
            
            # Log mulai augmentasi kelas
            class_start_message = f"🔄 Memulai augmentasi kelas {class_id} ({class_idx+1}/{len(classes_to_augment)}): {current_count}/{target_count} instance"
            self.logger.info(class_start_message)
            self.report_progress(
                class_idx, len(classes_to_augment),  # Overall progress
                class_start_message,
                status="info",
                step=1,  # Tahap proses utama
                class_id=class_id,
                class_progress=0,
                class_total=needed_augmentations,
                total_classes=len(classes_to_augment),
                current_class_index=class_idx
            )
            
            # Periksa jika ada file untuk kelas ini
            if not files_for_class:
                self.logger.warning(f"⚠️ Tidak ada file untuk kelas {class_id}, melewati")
                continue
                
            # Tracking progress untuk kelas ini
            class_augmented = 0
            class_generated = 0
            class_errors = 0
            
            try:
                # Proses dengan executor untuk paralelisme
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    # Hitung berapa file yang perlu diproses dan berapa variasi per file
                    # Jika needed_augmentations > jumlah file, maka kita perlu augmentasi lebih dari 1 variasi per file
                    base_variations = augmentation_params['num_variations']
                    
                    # Hitung jumlah total variasi yang dibutuhkan
                    total_variations_needed = needed_augmentations
                    
                    # Pilih file secara random jika ada banyak file
                    if len(files_for_class) > 0:
                        selected_files = files_for_class
                        # Jika jumlah file lebih banyak dari yang dibutuhkan, pilih secara random
                        if len(selected_files) > needed_augmentations and needed_augmentations > 0:
                            selected_files = random.sample(files_for_class, min(needed_augmentations, len(files_for_class)))
                    else:
                        selected_files = []
                    
                    # Log jumlah file terpilih
                    self.logger.info(f"📄 Kelas {class_id}: terpilih {len(selected_files)} file untuk menghasilkan {total_variations_needed} variasi")
                    
                    # Submit semua task
                    futures = {}
                    for img_path in selected_files:
                        # Copy parameter augmentasi dan tambahkan parameter spesifik kelas
                        params_for_class = augmentation_params.copy()
                        
                        # Konfigurasi task
                        future = executor.submit(
                            process_single_file, 
                            image_path=img_path, 
                            **params_for_class
                        )
                        futures[future] = img_path
                    
                    # Progress tracking
                    processed_files = 0
                    class_progress_report_interval = max(1, len(selected_files) // 10)  # Report setiap 10% progress
                    
                    # Proses hasil
                    for future in as_completed(futures):
                        if self._stop_signal:
                            self.logger.warning("⚠️ Augmentasi dihentikan oleh pengguna")
                            executor.shutdown(wait=False)
                            return {"status": "cancelled", "message": "Augmentasi dihentikan oleh pengguna"}
                        
                        # Get result
                        img_path = futures[future]
                        try:
                            result = future.result()
                            if result['status'] == 'success':
                                class_augmented += 1
                                class_generated += result.get('generated', 0)
                            else:
                                class_errors += 1
                                self.logger.debug(f"⚠️ Error augmentasi {img_path}: {result.get('message')}")
                        except Exception as e:
                            class_errors += 1
                            self.logger.debug(f"⚠️ Error task augmentasi {img_path}: {str(e)}")
                        
                        # Update progress
                        processed_files += 1
                        
                        # Report progress per kelas dengan throttling
                        if processed_files % class_progress_report_interval == 0 or processed_files == len(selected_files):
                            progress_percentage = int(processed_files / len(selected_files) * 100)
                            self.report_progress(
                                class_idx, len(classes_to_augment),  # Overall progress
                                f"Augmentasi kelas {class_id}: {processed_files}/{len(selected_files)} file ({progress_percentage}%)",
                                status="info",
                                step=1,  # Tahap proses utama
                                class_id=class_id,
                                class_progress=processed_files,
                                class_total=len(selected_files),
                                total_classes=len(classes_to_augment),
                                current_class_index=class_idx
                            )
                
                # Update statistik total
                total_augmented_count += class_augmented
                total_generated += class_generated
                total_errors += class_errors
                
                # Simpan hasil per kelas
                result_by_class[class_id] = {
                    'processed': class_augmented,
                    'generated': class_generated,
                    'errors': class_errors,
                    'original': len(files_for_class)
                }
                
                # Log selesai augmentasi kelas
                class_end_message = f"✅ Selesai augmentasi kelas {class_id}: {class_generated} gambar baru dari {class_augmented} file"
                self.logger.info(class_end_message)
                self.report_progress(
                    class_idx + 1, len(classes_to_augment),  # Overall progress
                    class_end_message,
                    status="success",
                    step=1,  # Tahap proses utama
                    class_id=class_id,
                    class_progress=len(selected_files),
                    class_total=len(selected_files),
                    total_classes=len(classes_to_augment),
                    current_class_index=class_idx
                )
                
            except Exception as e:
                self.logger.error(f"❌ Error saat augmentasi kelas {class_id}: {str(e)}")
                total_errors += 1
        
        # Pindahkan hasil ke direktori preprocessed jika diminta
        if move_to_preprocessed and total_generated > 0:
            move_message = f"📦 Memindahkan {total_generated} file hasil augmentasi ke {final_output_dir}"
            self.logger.info(move_message)
            self.report_progress(
                len(classes_to_augment), len(classes_to_augment),
                move_message,
                status="info",
                step=2  # Tahap finalisasi
            )
            
            try:
                self._move_files_to_preprocessed(
                    images_output_dir, 
                    labels_output_dir, 
                    augmentation_params['output_prefix'], 
                    final_output_dir,
                    split
                )
                self.logger.info(f"✅ Hasil augmentasi berhasil dipindahkan ke {final_output_dir}")
            except Exception as e:
                message = f"Error saat memindahkan file: {str(e)}"
                self.logger.error(f"❌ {message}")
                return {
                    "status": "partial_error", 
                    "message": message,
                    "original": sum(class_counts.values()),
                    "generated": total_generated,
                    "augmentation_types": augmentation_types,
                    "duration": time.time() - start_time,
                    "results_by_class": result_by_class
                }
        
        # Rangkuman hasil
        duration = time.time() - start_time
        summary_message = f"🎉 Augmentasi dataset dengan balancing kelas selesai: {total_generated} file dari {total_augmented_count} gambar dalam {duration:.1f} detik"
        self.logger.success(summary_message)
        self.report_progress(
            len(classes_to_augment), len(classes_to_augment),
            summary_message,
            status="success",
            step=2  # Tahap finalisasi
        )
        
        return {
            "status": "success",
            "original": sum(class_counts.values()),
            "generated": total_generated,
            "errors": total_errors,
            "total_files": sum(class_counts.values()) + total_generated,
            "augmentation_types": augmentation_types,
            "duration": duration,
            "output_dir": output_dir,
            "final_output_dir": final_output_dir,
            "target_count": target_count,
            "results_by_class": result_by_class
        }
    
    def _augment_without_class_tracking(
        self,
        image_files: List[str],
        augmentation_params: Dict[str, Any],
        num_workers: Optional[int],
        move_to_preprocessed: bool,
        final_output_dir: str,
        preprocessed_dir: str,
        output_dir: str,
        images_output_dir: str,
        labels_output_dir: str,
        split: str,
        augmentation_types: List[str],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Proses augmentasi tanpa tracking per kelas (metode standar).
        
        Args:
            image_files: List path file gambar
            augmentation_params: Parameter augmentasi
            num_workers: Jumlah worker
            move_to_preprocessed: Apakah hasil dipindahkan ke direktori preprocessed
            final_output_dir: Direktori akhir hasil
            preprocessed_dir: Direktori preprocessed
            output_dir: Direktori output sementara
            images_output_dir: Direktori output gambar
            labels_output_dir: Direktori output label
            split: Split dataset
            augmentation_types: Jenis augmentasi
            start_time: Waktu mulai proses
            
        Returns:
            Dictionary hasil augmentasi
        """
        total_files = len(image_files)
        self.logger.info(f"🚀 Memulai augmentasi {total_files} file dengan {len(augmentation_types)} jenis augmentasi")
        self.report_progress(0, total_files, f"Memulai augmentasi {total_files} file", "info")
        
        # Simpan statistik jumlah file yang berhasil diaugmentasi
        augmented_count = 0
        total_generated = 0
        total_errors = 0
        
        # Gunakan multiprocessing untuk augmentasi
        use_workers = num_workers if num_workers is not None else self.num_workers
        
        try:
            # Batasi worker agar tidak terlalu banyak
            max_workers = min(use_workers, multiprocessing.cpu_count(), 16)
            self.logger.info(f"🔄 Menggunakan {max_workers} workers dari {use_workers} yang dikonfigurasi")
            
            # Proses dengan executor
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit semua task
                futures = {
                    executor.submit(
                        process_single_file, 
                        image_path=img_path, 
                        **augmentation_params
                    ): img_path for img_path in image_files
                }
                
                # Progress tracking
                processed = 0
                for future in as_completed(futures):
                    if self._stop_signal:
                        self.logger.warning("⚠️ Augmentasi dihentikan oleh pengguna")
                        executor.shutdown(wait=False)
                        return {"status": "cancelled", "message": "Augmentasi dihentikan oleh pengguna"}
                    
                    # Get result
                    img_path = futures[future]
                    try:
                        result = future.result()
                        if result['status'] == 'success':
                            augmented_count += 1
                            total_generated += result.get('generated', 0)
                        else:
                            total_errors += 1
                            self.logger.warning(f"⚠️ Error augmentasi {img_path}: {result.get('message')}")
                    except Exception as e:
                        total_errors += 1
                        self.logger.warning(f"⚠️ Error task augmentasi {img_path}: {str(e)}")
                    
                    # Update progress
                    processed += 1
                    
                    # Throttled progress updates (setiap 5% atau 10 file)
                    report_interval = max(1, min(total_files // 20, 10))
                    if processed % report_interval == 0 or processed == total_files:
                        self.report_progress(
                            processed, total_files, 
                            f"Augmentasi {processed}/{total_files} file ({int(processed/total_files*100)}%)", 
                            "info" if total_errors == 0 else "warning"
                        )
        except Exception as e:
            message = f"Error saat proses multiprocessing: {str(e)}"
            self.logger.error(f"❌ {message}")
            return {"status": "error", "message": message}
            
        # Pindahkan hasil ke direktori preprocessed jika diminta
        if move_to_preprocessed and total_generated > 0:
            self.report_progress(
                message=f"Memindahkan {total_generated} file ke {final_output_dir}",
                status="info"
            )
            
            try:
                self._move_files_to_preprocessed(
                    images_output_dir, 
                    labels_output_dir, 
                    augmentation_params['output_prefix'], 
                    final_output_dir,
                    split
                )
                self.logger.info(f"✅ Hasil augmentasi berhasil dipindahkan ke {final_output_dir}")
            except Exception as e:
                message = f"Error saat memindahkan file: {str(e)}"
                self.logger.error(f"❌ {message}")
                return {
                    "status": "partial_error", 
                    "message": message,
                    "original": total_files,
                    "generated": total_generated,
                    "augmentation_types": augmentation_types,
                    "duration": time.time() - start_time
                }
        
        # Final progress update
        duration = time.time() - start_time
        summary_message = f"🎉 Augmentasi selesai: {total_generated} file dihasilkan dari {total_files} gambar dalam {duration:.1f} detik"
        self.report_progress(
            total_files, total_files, 
            summary_message,
            "success"
        )
        
        # Berhasil
        self.logger.success(summary_message)
        
        return {
            "status": "success",
            "original": total_files,
            "generated": total_generated,
            "errors": total_errors,
            "total_files": total_files + total_generated,
            "augmentation_types": augmentation_types,
            "duration": duration,
            "output_dir": output_dir,
            "final_output_dir": final_output_dir
        }