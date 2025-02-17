# File: utils/preprocessing.py
# Author: Alfrida Sabar
# Deskripsi: Modul preprocessing untuk dataset uang kertas Rupiah, 
# menangani resize, normalisasi, dan augmentasi data

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import albumentations as A
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import yaml

from utils.logger import SmartCashLogger

class ImagePreprocessor:
    """Preprocessor untuk dataset gambar uang kertas Rupiah"""
    
    def __init__(
        self,
        config_path: str,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.target_size = tuple(self.config['model']['img_size'])
        
        # Setup augmentasi berdasarkan config
        self.augmentor = self._setup_augmentations()
        
        # Hitung jumlah worker berdasarkan CPU dan memory limit
        self.n_workers = self._calculate_workers()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi preprocessing"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _calculate_workers(self) -> int:
        """Hitung jumlah optimal worker berdasarkan resource limit"""
        cpu_count = mp.cpu_count()
        memory_limit = self.config['model']['memory_limit']
        
        # Gunakan 60% dari CPU yang tersedia (sesuai memory_limit)
        suggested_workers = max(1, int(cpu_count * memory_limit))
        
        self.logger.info(
            f"🧮 Worker count: {suggested_workers} "
            f"(dari {cpu_count} CPU dengan limit {memory_limit*100}%)"
        )
        
        return suggested_workers
    
    def _setup_augmentations(self) -> A.Compose:
        """Setup pipeline augmentasi data"""
        aug_config = self.config['training']
        
        return A.Compose([
            A.RandomResizedCrop(
                *self.target_size,
                scale=(0.8, 1.0),
                p=1.0
            ),
            A.HorizontalFlip(p=aug_config['fliplr']),
            A.VerticalFlip(p=aug_config['flipud']),
            A.HueSaturationValue(
                hue_shift_limit=aug_config['hsv_h'],
                sat_shift_limit=aug_config['hsv_s'],
                val_shift_limit=aug_config['hsv_v'],
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=aug_config['translate'],
                scale_limit=aug_config['scale'],
                rotate_limit=aug_config['degrees'],
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ])
    
    def preprocess_image(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        augment: bool = True
    ) -> np.ndarray:
        """
        Preprocess satu gambar
        Args:
            image_path: Path ke file gambar
            save_path: Path untuk menyimpan hasil (optional)
            augment: Apakah perlu augmentasi
        Returns:
            Gambar yang sudah dipreprocess
        """
        try:
            # Baca gambar
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize
            image = cv2.resize(image, self.target_size)
            
            # Augmentasi jika diperlukan
            if augment:
                transformed = self.augmentor(image=image)
                image = transformed['image']
            
            # Simpan jika diperlukan
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(
                    save_path,
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
            
            return image
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memproses {image_path}: {str(e)}")
            raise e
    
    def preprocess_dataset(
        self,
        input_dir: str,
        output_dir: str,
        augment: bool = True
    ) -> None:
        """
        Preprocess seluruh dataset
        Args:
            input_dir: Direktori dataset mentah
            output_dir: Direktori output
            augment: Apakah perlu augmentasi
        """
        self.logger.start(
            f"🎬 Memulai preprocessing dataset di {input_dir}"
        )
        
        try:
            # Siapkan list file
            image_files = list(Path(input_dir).rglob("*.jpg"))
            self.logger.info(f"📁 Total gambar: {len(image_files)}")
            
            # Buat output dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Process dengan multiprocessing
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                list(tqdm(
                    executor.map(
                        lambda x: self.preprocess_image(
                            str(x),
                            str(Path(output_dir) / x.name),
                            augment
                        ),
                        image_files
                    ),
                    total=len(image_files),
                    desc="💫 Processing"
                ))
            
            self.logger.success(
                f"✨ Preprocessing selesai! Output: {output_dir}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing gagal: {str(e)}")
            raise e