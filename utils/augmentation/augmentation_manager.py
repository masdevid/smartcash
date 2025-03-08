"""
File: smartcash/utils/augmentation/augmentation_manager.py
Author: Alfrida Sabar
Deskripsi: Kelas utama yang mengelola proses augmentasi dataset dengan dukungan multithreading
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import time
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase
from smartcash.utils.augmentation.augmentation_pipeline import AugmentationPipeline
from smartcash.utils.augmentation.augmentation_processor import AugmentationProcessor
from smartcash.utils.augmentation.augmentation_validator import AugmentationValidator
from smartcash.utils.augmentation.augmentation_checkpoint import AugmentationCheckpoint

class AugmentationManager(AugmentationBase):
    """
    Kelas utama yang mengelola seluruh proses augmentasi dataset.
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4,
        checkpoint_interval: int = 50
    ):
        """
        Inisialisasi manager augmentasi.
        
        Args:
            config: Konfigurasi aplikasi
            output_dir: Direktori output
            logger: Logger kustom
            num_workers: Jumlah worker untuk paralelisasi
            checkpoint_interval: Interval checkpoint dalam jumlah gambar
        """
        super().__init__(config, output_dir, logger)
        self.num_workers = num_workers
        
        # Inisialisasi komponen
        self.pipeline = AugmentationPipeline(config, output_dir, logger)
        self.processor = AugmentationProcessor(config, self.pipeline, output_dir, logger)
        self.validator = AugmentationValidator(config, output_dir, logger)
        self.checkpoint_manager = AugmentationCheckpoint(
            config, output_dir, logger, checkpoint_interval
        )
        
        # Inisialisasi statistics
        self.stats = self.reset_stats()
        
    def augment_dataset(
        self,
        split: str = 'train',
        augmentation_types: List[str] = ['combined'],
        num_variations: int = 2,
        output_prefix: str = 'aug',
        resume: bool = True,
        validate_results: bool = True
    ) -> Dict[str, int]:
        """
        Augmentasi dataset dengan validasi layer label.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            augmentation_types: List tipe augmentasi yang akan digunakan
            num_variations: Jumlah variasi per tipe augmentasi
            output_prefix: Prefix untuk nama file output
            resume: Lanjutkan dari checkpoint jika ada
            validate_results: Validasi hasil augmentasi
            
        Returns:
            Dict statistik augmentasi
        """
        # Validasi tipe augmentasi
        valid_types = ['position', 'lighting', 'combined', 'extreme_rotation']
        augmentation_types = [t for t in augmentation_types if t in valid_types]
        
        if not augmentation_types:
            self.logger.warning("⚠️ Tidak ada tipe augmentasi valid yang ditentukan, menggunakan 'combined'")
            augmentation_types = ['combined']
        
        # Setup direktori
        input_dir = self.data_dir / split
        output_dir = self.output_dir / split
        
        images_dir = input_dir / 'images'
        labels_dir = input_dir / 'labels'
        output_images_dir = output_dir / 'images'
        output_labels_dir = output_dir / 'labels'
        
        # Pastikan direktori output ada
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset statistik jika tidak melanjutkan
        if not resume:
            self.stats = self.reset_stats()
        
        # Periksa file gambar yang tersedia
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return self.stats
        
        self.logger.info(
            f"🔍 Menemukan {len(image_files)} gambar di {split}\n"
            f"   Menggunakan {len(augmentation_types)} tipe augmentasi: {', '.join(augmentation_types)}\n"
            f"   Variasi per tipe: {num_variations}"
        )
        
        # Process gambar yang sudah diproses dari checkpoint jika resume
        processed_files = self.checkpoint_manager.load_checkpoint(split) if resume else []
        processed_files_set = set(processed_files)
        
        # Filter gambar yang belum diproses
        unprocessed_images = [img for img in image_files if str(img) not in processed_files_set]
        
        self.logger.info(
            f"🔄 {len(processed_files)} gambar telah diproses sebelumnya\n"
            f"   {len(unprocessed_images)} gambar akan diproses"
        )
        
        if not unprocessed_images:
            self.logger.success("✅ Semua gambar telah diproses!")
            return self.stats
        
        # Fungsi untuk memproses satu gambar
        def process_image(img_path):
            # Simpan file paths yang telah berhasil diproses
            successful_files = []
            
            # Cari label yang sesuai
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            for aug_type in augmentation_types:
                # Augmentasi
                aug_images, aug_layer_labels, out_paths = self.processor.augment_image(
                    img_path,
                    label_path if label_path.exists() else None,
                    aug_type,
                    output_prefix,
                    variations=num_variations
                )
                
                # Simpan hasil
                for idx, (img, out_path) in enumerate(zip(aug_images, out_paths)):
                    # Ubah path ke direktori output
                    save_path = output_images_dir / out_path.name
                    
                    # Ambil label layer yang sesuai dengan gambar ini
                    layer_labels_to_save = aug_layer_labels[idx] if aug_layer_labels and idx < len(aug_layer_labels) else {}
                    
                    # Simpan hasil
                    if self.processor.save_augmented_data(img, layer_labels_to_save, save_path, output_labels_dir):
                        successful_files.append(aug_type + "_" + save_path.name)
            
            # Jika berhasil, tambahkan path gambar asli ke daftar yang sudah diproses
            if successful_files:
                return str(img_path), successful_files
            else:
                return None, []
        
        # Proses menggunakan ThreadPoolExecutor dengan progress bar
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit semua tugas
            futures = {executor.submit(process_image, img_path): img_path for img_path in unprocessed_images}
            
            # Inisialisasi progress bar
            pbar = tqdm(
                total=len(unprocessed_images),
                desc=f"🎨 Augmentasi {split}",
                unit="img"
            )
            
            # Track waktu untuk checkpoint
            last_checkpoint_time = time.time()
            
            # Proses hasil setiap future yang selesai
            for future in futures:
                img_path, successful_files = future.result()
                pbar.update(1)
                
                if img_path:
                    processed_files.append(img_path)
                
                # Checkpoint setiap interval atau 5 menit
                current_time = time.time()
                if self.checkpoint_manager.should_checkpoint(len(processed_files), last_checkpoint_time):
                    self.checkpoint_manager.save_checkpoint(processed_files, self.stats)
                    last_checkpoint_time = current_time
                    
                    # Update statistik dalam progress bar
                    with self._stats_lock:
                        success_rate = (self.stats['augmented'] / max(1, self.stats['processed'] * len(augmentation_types) * num_variations)) * 100
                        pbar.set_postfix({
                            'augmented': self.stats['augmented'],
                            'success_rate': f"{success_rate:.1f}%"
                        })
            
            pbar.close()
        
        # Validasi hasil jika diminta
        if validate_results:
            self.logger.info("🔍 Memvalidasi hasil augmentasi...")
            validation_stats = self.validator.validate_augmentation_results(output_dir)
            self.stats['validation'] = validation_stats
            
        # Simpan checkpoint final
        self.checkpoint_manager.save_checkpoint(processed_files, self.stats)
        
        # Perbarui statistik durasi
        with self._stats_lock:
            self.stats['duration'] = time.time() - self.stats['start_time']
            
            # Hitung dan tampilkan statistik
            total_images = len(image_files)
            total_augmented = self.stats['augmented']
            expected_augmented = total_images * len(augmentation_types) * num_variations
            completion_rate = (total_augmented / max(1, expected_augmented)) * 100
            
            self.logger.success(
                f"✨ Augmentasi {split} selesai dalam {self.stats['duration']:.1f} detik\n"
                f"   Gambar asli: {total_images}\n"
                f"   Total hasil augmentasi: {total_augmented}\n"
                f"   Rasio penyelesaian: {completion_rate:.1f}%\n"
                f"   Gagal: {self.stats['failed']}\n"
                f"   Dilewati (tidak valid): {self.stats['skipped_invalid']}"
            )
            
            # Log statistik per tipe augmentasi
            self.logger.info("📊 Statistik per tipe augmentasi:")
            for aug_type, count in self.stats['per_type'].items():
                if aug_type in augmentation_types:
                    pct = (count / max(1, total_augmented)) * 100
                    self.logger.info(f"   • {aug_type}: {count} ({pct:.1f}%)")
            
            # Log statistik layer
            self.logger.info("📊 Statistik per layer:")
            for layer, count in self.stats['layer_stats'].items():
                if layer in self.active_layers:
                    self.logger.info(f"   • {layer}: {count} objek")
        
        return self.stats