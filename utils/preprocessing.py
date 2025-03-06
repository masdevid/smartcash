# File: smartcash/utils/preprocessing.py
# Author: Alfrida Sabar
# Deskripsi: Modul preprocessing yang diperbarui untuk dataset uang kertas Rupiah, 
# menggunakan enhanced cache dan layer config manager

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import albumentations as A
from tqdm.auto import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import yaml

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.coordinate_normalizer import CoordinateNormalizer
from smartcash.utils.enhanced_cache import EnhancedCache  # Menggunakan enhanced cache
from smartcash.utils.layer_config_manager import get_layer_config  # Menggunakan layer config manager

class ImagePreprocessor:
    """Preprocessor untuk dataset gambar uang kertas Rupiah dengan komponen yang diperbarui"""
    
    def __init__(
        self,
        config_path: str,
        logger: Optional[SmartCashLogger] = None,
        cache_size_gb: float = 1.0
    ):
        """
        Inisialisasi image preprocessor.
        
        Args:
            config_path: Path ke file konfigurasi
            logger: Custom logger
            cache_size_gb: Ukuran cache dalam GB
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.target_size = tuple(self.config['model']['img_size'])
        
        # Setup komponen
        self.augmentor = self._setup_augmentations()
        self.coord_normalizer = CoordinateNormalizer(logger=self.logger)
        
        # Gunakan EnhancedCache dengan fitur yang lebih baik
        self.cache = EnhancedCache(
            cache_dir=".cache/preprocessing",
            max_size_gb=cache_size_gb,
            logger=self.logger
        )
        
        # Dapatkan konfigurasi layer dari layer config manager
        self.layer_config = get_layer_config()
        self.active_layers = self.config.get('layers', ['banknote'])
        
        # Hitung jumlah worker berdasarkan CPU dan memory limit
        self.n_workers = self._calculate_workers()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi preprocessing"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _calculate_workers(self) -> int:
        """Hitung jumlah optimal worker berdasarkan resource limit"""
        cpu_count = mp.cpu_count()
        memory_limit = self.config['model'].get('memory_limit', 0.6)
        suggested_workers = max(1, int(cpu_count * memory_limit))
        
        self.logger.info(
            f"🧮 Worker count: {suggested_workers} "
            f"(dari {cpu_count} CPU dengan limit {memory_limit*100}%)"
        )
        
        return suggested_workers
    
    def _setup_augmentations(self) -> A.Compose:
        """Setup pipeline augmentasi data berdasarkan konfigurasi"""
        aug_config = self.config['training']
        
        return A.Compose([
            A.RandomResizedCrop(
                *self.target_size,
                scale=(0.8, 1.0),
                p=1.0
            ),
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.5)),
            A.VerticalFlip(p=aug_config.get('flipud', 0.0)),
            A.HueSaturationValue(
                hue_shift_limit=aug_config.get('hsv_h', 0.015),
                sat_shift_limit=aug_config.get('hsv_s', 0.7),
                val_shift_limit=aug_config.get('hsv_v', 0.4),
                p=0.5
            ),
            A.ShiftScaleRotate(
                shift_limit=aug_config.get('translate', 0.1),
                scale_limit=aug_config.get('scale', 0.5),
                rotate_limit=aug_config.get('degrees', 45),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                p=1.0
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def process_image_and_label(
        self,
        image_path: str,
        label_path: Optional[str] = None,
        save_dir: Optional[str] = None,
        augment: bool = True
    ) -> Tuple[np.ndarray, Optional[str]]:
        """
        Preprocess gambar dan labelnya dengan enhanced cache
        Args:
            image_path: Path ke file gambar
            label_path: Path ke file label (optional)
            save_dir: Direktori untuk menyimpan hasil
            augment: Apakah perlu augmentasi
        Returns:
            Tuple (gambar yang sudah dipreprocess, path label yang dinormalisasi)
        """
        try:
            # Gunakan enhanced cache untuk get/put
            cache_params = {
                'target_size': self.target_size,
                'augment': augment,
                'label_path': label_path,
                'save_dir': save_dir
            }
            cache_key = self.cache.get_cache_key(image_path, cache_params)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                return cached_result['image'], cached_result['label_path']
            
            # Process gambar
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = (image.shape[1], image.shape[0])
            
            # Resize
            image = cv2.resize(image, self.target_size)
            
            # Augmentasi jika diperlukan
            if augment:
                transformed = self.augmentor(image=image)
                image = transformed['image']
            
            # Process label jika ada
            normalized_label_path = None
            if label_path and Path(label_path).exists():
                if save_dir:
                    normalized_label_path = str(
                        Path(save_dir) / 'labels' / Path(label_path).name
                    )
                
                self.coord_normalizer.process_label_file(
                    label_path,
                    original_size,
                    normalized_label_path
                )
            
            # Simpan gambar jika diperlukan
            if save_dir:
                img_save_path = str(
                    Path(save_dir) / 'images' / Path(image_path).name
                )
                os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
                cv2.imwrite(
                    img_save_path,
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                )
            
            # Simpan ke enhanced cache
            result = {
                'image': image,
                'label_path': normalized_label_path
            }
            self.cache.put(cache_key, result, image.nbytes)
            
            return image, normalized_label_path
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memproses {image_path}: {str(e)}")
            raise e
    
    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        augment: bool = True
    ) -> None:
        """
        Preprocess seluruh dataset dengan enhanced cache
        Args:
            input_dir: Direktori dataset mentah
            output_dir: Direktori output
            augment: Apakah perlu augmentasi
        """
        self.logger.start(
            f"🎬 Memulai preprocessing dataset di {input_dir}"
        )
        
        try:
            input_dir = Path(input_dir)
            output_dir = Path(output_dir)
            
            # Siapkan list file
            image_files = list(Path(input_dir).rglob("*.jpg"))
            self.logger.info(f"📁 Total gambar: {len(image_files)}")
            
            # Buat output dirs
            os.makedirs(output_dir / 'images', exist_ok=True)
            os.makedirs(output_dir / 'labels', exist_ok=True)
            
            def process_pair(image_path: Path):
                try:
                    # Cari label yang sesuai
                    label_path = input_dir / 'labels' / f"{image_path.stem}.txt"
                    
                    self.process_image_and_label(
                        str(image_path),
                        str(label_path) if label_path.exists() else None,
                        str(output_dir),
                        augment
                    )
                    
                except Exception as e:
                    self.logger.error(
                        f"❌ Gagal memproses {image_path.name}: {str(e)}"
                    )
            
            # Process dengan multiprocessing
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                list(tqdm(
                    executor.map(process_pair, image_files),
                    total=len(image_files),
                    desc="💫 Processing"
                ))
                
            # Log cache stats dari enhanced cache
            stats = self.cache.get_stats()
            self.logger.success(
                f"✨ Preprocessing selesai!\n"
                f"📊 Cache stats:\n"
                f"   Hit rate: {stats['hit_ratio']:.1f}%\n"
                f"   Cache size: {stats['cache_size_mb']:.1f} MB\n"
                f"   Files cached: {stats['file_count']}\n"
                f"💾 Output: {output_dir}"
            )
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing gagal: {str(e)}")
            raise e
            
    def clear_cache(self) -> None:
        """Bersihkan enhanced cache"""
        try:
            self.cache.clear()
            self.logger.success("🧹 Cache berhasil dibersihkan")
        except Exception as e:
            self.logger.error(f"❌ Gagal membersihkan cache: {str(e)}")
            raise e
            
    def validate_preprocessing(
        self,
        output_dir: str
    ) -> Dict[str, int]:
        """
        Validasi hasil preprocessing
        Args:
            output_dir: Direktori output preprocessing
        Returns:
            Dict statistik validasi
        """
        self.logger.info(f"🔍 Memvalidasi hasil preprocessing di {output_dir}")
        
        try:
            output_dir = Path(output_dir)
            stats = {
                'total_images': 0,
                'total_labels': 0,
                'invalid_images': 0,
                'invalid_labels': 0,
                'layer_stats': {layer: 0 for layer in self.active_layers}
            }
            
            # Check gambar
            image_dir = output_dir / 'images'
            if image_dir.exists():
                for img_path in image_dir.glob("*.jpg"):
                    stats['total_images'] += 1
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None or img.shape[:2] != self.target_size[::-1]:
                            stats['invalid_images'] += 1
                            self.logger.warning(
                                f"⚠️ Invalid image: {img_path.name}"
                            )
                    except:
                        stats['invalid_images'] += 1
                        
            # Check label
            label_dir = output_dir / 'labels'
            if label_dir.exists():
                # Get class-to-layer mapping
                class_to_layer = {}
                for layer, layer_config in self.layer_config.get_layer_config().items():
                    for cls_id in layer_config['class_ids']:
                        class_to_layer[cls_id] = layer
                
                for label_path in label_dir.glob("*.txt"):
                    stats['total_labels'] += 1
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            if not lines:
                                stats['invalid_labels'] += 1
                                self.logger.warning(
                                    f"⚠️ Empty label: {label_path.name}"
                                )
                                continue
                                
                            # Count per layer
                            for line in lines:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    try:
                                        cls_id = int(float(parts[0]))
                                        if cls_id in class_to_layer:
                                            layer = class_to_layer[cls_id]
                                            if layer in stats['layer_stats']:
                                                stats['layer_stats'][layer] += 1
                                    except:
                                        pass
                    except:
                        stats['invalid_labels'] += 1
                        
            # Log hasil
            self.logger.success(
                f"✅ Validasi selesai:\n"
                f"📊 Statistik:\n"
                f"   Total images: {stats['total_images']}\n"
                f"   Invalid images: {stats['invalid_images']}\n"
                f"   Total labels: {stats['total_labels']}\n"
                f"   Invalid labels: {stats['invalid_labels']}"
            )
            
            # Log statistik per layer
            for layer, count in stats['layer_stats'].items():
                if count > 0:
                    self.logger.info(f"📊 Layer '{layer}': {count} anotasi")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Validasi gagal: {str(e)}")
            raise e