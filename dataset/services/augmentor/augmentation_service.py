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