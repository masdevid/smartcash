"""
File: smartcash/utils/augmentation/augmentation_manager.py
Author: Alfrida Sabar
Deskripsi: Kelas utama yang mengelola proses augmentasi dataset dengan dukungan multithreading
"""

import time
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase
from smartcash.utils.augmentation.augmentation_pipeline import AugmentationPipeline
from smartcash.utils.augmentation.augmentation_processor import AugmentationProcessor
from smartcash.utils.augmentation.augmentation_validator import AugmentationValidator
from smartcash.utils.augmentation.augmentation_checkpoint import AugmentationCheckpoint

class AugmentationManager(AugmentationBase):
    """Kelas utama yang mengelola seluruh proses augmentasi dataset."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4,
        checkpoint_interval: int = 50
    ):
        """Inisialisasi manager augmentasi."""
        super().__init__(config, output_dir, logger)
        
        # Inisialisasi komponen
        self.pipeline = AugmentationPipeline(config, output_dir, logger)
        self.processor = AugmentationProcessor(config, self.pipeline, output_dir, logger)
        self.validator = AugmentationValidator(config, output_dir, logger)
        self.checkpoint_manager = AugmentationCheckpoint(
            config, output_dir, logger, checkpoint_interval
        )
        
        # Konfigurasi tambahan
        self.num_workers = num_workers
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
        """Augmentasi dataset dengan validasi layer label."""
        # Validasi dan normalisasi tipe augmentasi
        valid_types = ['position', 'lighting', 'combined', 'extreme_rotation']
        augmentation_types = [t for t in augmentation_types if t in valid_types] or ['combined']
        
        # Setup direktori
        input_dir = self.data_dir / split
        output_dir = self.output_dir / split
        images_dir, labels_dir = input_dir / 'images', input_dir / 'labels'
        output_images_dir, output_labels_dir = output_dir / 'images', output_dir / 'labels'
        
        # Buat direktori output
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset atau lanjutkan statistik
        self.stats = self.reset_stats() if not resume else self.stats
        
        # Temukan gambar yang tersedia
        image_files = list(images_dir.glob('*.[jJ][pP]*[gG]'))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return self.stats
        
        # Lanjutkan dari checkpoint jika diminta
        processed_files = self.checkpoint_manager.load_checkpoint(split) if resume else []
        processed_files_set = set(processed_files)
        unprocessed_images = [img for img in image_files if str(img) not in processed_files_set]
        
        self.logger.info(
            f"🔍 {len(image_files)} gambar ditemukan\n"
            f"   Tipe augmentasi: {', '.join(augmentation_types)}\n"
            f"   Variasi per tipe: {num_variations}\n"
            f"   Telah diproses: {len(processed_files)}"
        )
        
        if not unprocessed_images:
            self.logger.success("✅ Semua gambar telah diproses!")
            return self.stats
        
        # Fungsi untuk memproses gambar
        def process_image(img_path):
            label_path = labels_dir / f"{img_path.stem}.txt"
            successful_files = []
            
            for aug_type in augmentation_types:
                aug_images, aug_layer_labels, out_paths = self.processor.augment_image(
                    img_path,
                    label_path if label_path.exists() else None,
                    aug_type,
                    output_prefix,
                    variations=num_variations
                )
                
                for idx, (img, out_path) in enumerate(zip(aug_images, out_paths)):
                    save_path = output_images_dir / out_path.name
                    layer_labels = aug_layer_labels[idx] if aug_layer_labels and idx < len(aug_layer_labels) else {}
                    
                    if self.processor.save_augmented_data(img, layer_labels, save_path, output_labels_dir):
                        successful_files.append(aug_type + "_" + save_path.name)
            
            return str(img_path), successful_files
        
        # Proses gambar menggunakan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_image, img_path): img_path for img_path in unprocessed_images}
            
            pbar = tqdm(total=len(unprocessed_images), desc=f"🎨 Augmentasi {split}", unit="img")
            last_checkpoint_time = time.time()
            
            for future in as_completed(futures):
                img_path, successful_files = future.result()
                pbar.update(1)
                
                if successful_files:
                    processed_files.append(img_path)
                
                # Checkpoint
                current_time = time.time()
                if self.checkpoint_manager.should_checkpoint(len(processed_files), last_checkpoint_time):
                    self.checkpoint_manager.save_checkpoint(processed_files, self.stats)
                    last_checkpoint_time = current_time
            
            pbar.close()
        
        # Validasi hasil
        if validate_results:
            validation_stats = self.validator.validate_augmentation_results(output_dir)
            self.stats['validation'] = validation_stats
        
        # Simpan checkpoint akhir
        self.checkpoint_manager.save_checkpoint(processed_files, self.stats)
        
        # Perbarui dan log statistik
        with self._stats_lock:
            self.stats['duration'] = time.time() - self.stats['start_time']
            total_images, total_augmented = len(image_files), self.stats['augmented']
            expected_augmented = total_images * len(augmentation_types) * num_variations
            completion_rate = (total_augmented / max(1, expected_augmented)) * 100
            
            self.logger.success(
                f"✨ Augmentasi {split} selesai dalam {self.stats['duration']:.1f} detik\n"
                f"   Gambar asli: {total_images}\n"
                f"   Total augmentasi: {total_augmented}\n"
                f"   Rasio penyelesaian: {completion_rate:.1f}%\n"
                f"   Gagal: {self.stats['failed']}"
            )
        
        return self.stats