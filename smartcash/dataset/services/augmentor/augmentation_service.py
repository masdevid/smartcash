"""
File: smartcash/dataset/services/augmentor/augmentation_service.py
Deskripsi: Layanan utama untuk augmentasi dataset dengan dukungan multi-processing dan balancing class
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

from smartcash.common.logger import get_logger
from smartcash.dataset.services.augmentor.pipeline_factory import AugmentationPipelineFactory
from smartcash.dataset.services.augmentor.bbox_augmentor import BBoxAugmentor
from smartcash.dataset.services.augmentor.class_balancer import ClassBalancer
from smartcash.dataset.services.augmentor.augmentation_worker import process_single_file

class AugmentationService:
    """
    Layanan augmentasi dataset dengan dukungan multiprocessing dan balancing class.
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
        Augmentasi dataset dengan penggunaan multiprocessing dan balancing class.
        
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
        
        # Filter file berdasarkan kebutuhan balancing jika diminta
        if target_balance:
            try:
                # Persiapkan balancing untuk class target
                message = f"🎯 Mempersiapkan balancing class dengan target {target_count} sampel per kelas"
                self.report_progress(message=message)
                self.logger.info(message)
                
                # Gunakan class balancer untuk mendapatkan file yang perlu diaugmentasi
                result = self.class_balancer.prepare_balanced_dataset(
                    image_files=image_files,
                    labels_dir=labels_input_dir,
                    target_count=target_count,
                    filter_single_class=True  # Filter untuk file yang hanya memiliki 1 class
                )
                
                # Gunakan file yang direkomendasikan untuk balancing
                balanced_files = result.get('selected_files', [])
                class_stats = result.get('class_counts', {})
                augmentation_needs = result.get('augmentation_needs', {})
                
                # Log kebutuhan augmentasi per kelas dengan lebih detail
                for class_id, needed in augmentation_needs.items():
                    current_count = class_stats.get(class_id, 0)
                    if needed > 0:
                        self.logger.info(f"🏷️ Kelas {class_id}: perlu tambahan {needed} sampel (dari {current_count} → {target_count})")
                    else:
                        self.logger.info(f"✅ Kelas {class_id}: sudah cukup ({current_count} ≥ {target_count})")
                
                # Update file yang akan diproses
                if balanced_files:
                    self.logger.info(f"📊 Menggunakan {len(balanced_files)} dari {len(image_files)} untuk balancing class")
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
            # TODO: Implementasi resume dengan tracking file yang sudah diproses
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
        
        # Proses dengan multiprocessing
        self.logger.info(f"🚀 Memulai augmentasi dengan {self.num_workers if num_workers is None else num_workers} workers")
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
                    self.report_progress(
                        processed, total_files, 
                        f"Augmentasi {processed}/{total_files} file", 
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
                    output_prefix, 
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
        self.report_progress(
            total_files, total_files, 
            f"Augmentasi selesai: {total_generated} file dihasilkan dalam {duration:.1f} detik", 
            "success"
        )
        
        # Berhasil
        self.logger.success(f"✅ Augmentasi dataset selesai: {total_generated} file dari {total_files} gambar dalam {duration:.1f} detik")
        
        return {
            "status": "success",
            "original": total_files,
            "generated": total_generated,
            "errors": total_errors,
            "total_files": total_files + total_generated,
            "augmentation_types": augmentation_types,
            "duration": duration,
            "output_dir": output_dir,
            "final_output_dir": final_output_dir,
            "target_count": target_count
        }
    
    def _move_files_to_preprocessed(
        self, 
        images_source_dir: str, 
        labels_source_dir: str,
        file_prefix: str,
        preprocessed_dir: str,
        split: str = 'train'
    ) -> None:
        """
        Pindahkan file hasil augmentasi ke direktori preprocessed.
        
        Args:
            images_source_dir: Direktori sumber gambar
            labels_source_dir: Direktori sumber label
            file_prefix: Prefix file yang akan dipindahkan
            preprocessed_dir: Path direktori preprocessed
            split: Split dataset ('train', 'valid', 'test')
        """
        # Siapkan path tujuan
        images_target_dir = os.path.join(preprocessed_dir, split, 'images')
        labels_target_dir = os.path.join(preprocessed_dir, split, 'labels')
        
        # Pastikan direktori ada
        os.makedirs(images_target_dir, exist_ok=True)
        os.makedirs(labels_target_dir, exist_ok=True)
        
        # Temukan semua file gambar augmentasi
        image_pattern = os.path.join(images_source_dir, f"{file_prefix}_*.jpg")
        image_files = glob.glob(image_pattern)
        
        # Jika tidak ada file, kembalikan
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada file dengan pola {image_pattern} untuk dipindahkan")
            return
        
        # Log jumlah file
        self.logger.info(f"🔄 Memindahkan {len(image_files)} file gambar ke {images_target_dir}")
        
        # Pindahkan gambar
        for img_path in tqdm(image_files, desc="Memindahkan gambar", unit="file"):
            filename = os.path.basename(img_path)
            target_path = os.path.join(images_target_dir, filename)
            
            # Copy file (tidak gunakan move untuk menghindari kehilangan file jika error)
            shutil.copy2(img_path, target_path)
            
            # Jika label ada, pindahkan juga
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_source = os.path.join(labels_source_dir, label_filename)
            
            if os.path.exists(label_source):
                label_target = os.path.join(labels_target_dir, label_filename)
                shutil.copy2(label_source, label_target)
    
    def stop(self) -> None:
        """Berhenti augmentasi yang sedang berjalan."""
        self._stop_signal = True
        self.logger.warning("⚠️ Mengirim sinyal berhenti ke augmentation service")