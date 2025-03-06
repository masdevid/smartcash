# File: smartcash/utils/multilayer_augmentation.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk augmentasi dataset multilayer dengan validasi label

import os
import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import json
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import random

from smartcash.utils.logger import SmartCashLogger

class MultilayerAugmentation:
    """Kelas untuk augmentasi data yang menjaga konsistensi layer label."""
    
    def __init__(
        self, 
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi augmentasi multilayer.
        
        Args:
            config: Konfigurasi aplikasi
            output_dir: Direktori output (jika None, gunakan path dari config)
            logger: Logger kustom
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup path
        self.data_dir = Path(config.get('data_dir', 'data'))
        self.output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Konfigurasi layer
        self.layer_config = {
            'banknote': {
                'classes': ['001', '002', '005', '010', '020', '050', '100'],
                'class_ids': list(range(7))  # 0-6
            },
            'nominal': {
                'classes': ['l2_001', 'l2_002', 'l2_005', 'l2_010', 'l2_020', 'l2_050', 'l2_100'],
                'class_ids': list(range(7, 14))  # 7-13
            },
            'security': {
                'classes': ['l3_sign', 'l3_text', 'l3_thread'],
                'class_ids': list(range(14, 17))  # 14-16
            }
        }
        
        # Layer yang diaktifkan
        self.active_layers = config.get('layers', ['banknote'])
        
        # Setup pipeline augmentasi
        self._setup_augmentation_pipelines()
        
        # Track statistik
        self.stats = {
            'processed': 0,
            'augmented': 0,
            'failed': 0,
            'skipped_invalid': 0,
            'layer_stats': {layer: 0 for layer in self.layer_config}
        }
    
    def _setup_augmentation_pipelines(self):
        """Setup pipeline augmentasi untuk berbagai kondisi."""
        # Ekstrak parameter dari config
        aug_config = self.config.get('training', {})
        
        # Pipeline posisi - variasi posisi/orientasi uang
        self.position_aug = A.Compose([
            A.SafeRotate(limit=30, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.5),
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.5)),
            A.ShiftScaleRotate(
                shift_limit=aug_config.get('translate', 0.1),
                scale_limit=aug_config.get('scale', 0.5),
                rotate_limit=aug_config.get('degrees', 45),
                p=0.5
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3  # Pastikan objek masih terlihat setidaknya 30%
        ))
        
        # Pipeline pencahayaan - variasi cahaya dan kontras
        self.lighting_aug = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.RandomShadow(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=aug_config.get('hsv_h', 0.015),
                sat_shift_limit=aug_config.get('hsv_s', 0.7),
                val_shift_limit=aug_config.get('hsv_v', 0.4),
                p=0.3
            ),
            A.ImageCompression(
                quality_lower=80,
                quality_upper=100,
                p=0.2
            )
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Pipeline kombinasi - gabungan posisi dan pencahayaan
        self.combined_aug = A.Compose([
            A.OneOf([
                A.SafeRotate(limit=30, p=0.7),
                A.Perspective(scale=(0.05, 0.1), p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=aug_config.get('translate', 0.1),
                    scale_limit=aug_config.get('scale', 0.5),
                    rotate_limit=aug_config.get('degrees', 45),
                    p=0.5
                )
            ], p=0.7),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomShadow(p=0.5),
                A.HueSaturationValue(p=0.5)
            ], p=0.7),
            A.HorizontalFlip(p=aug_config.get('fliplr', 0.3)),
        ], bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
    
    def augment_image(
        self,
        image_path: Path,
        label_path: Optional[Path] = None,
        augmentation_type: str = 'combined',
        output_prefix: str = 'aug'
    ) -> Tuple[List[np.ndarray], List[Dict[str, List[Tuple[int, List[float]]]]], List[Path]]:
        """
        Augmentasi satu gambar dengan validasi label.
        
        Args:
            image_path: Path ke file gambar
            label_path: Path ke file label (opsional)
            augmentation_type: Tipe augmentasi ('position', 'lighting', atau 'combined')
            output_prefix: Prefix untuk nama file output
            
        Returns:
            Tuple of (augmented_images, augmented_layer_labels, output_paths)
        """
        # Baca gambar
        image = cv2.imread(str(image_path))
        if image is None:
            self.logger.error(f"❌ Tidak dapat membaca gambar: {image_path}")
            self.stats['failed'] += 1
            return [], [], []
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Baca dan validasi label
        all_bboxes = []
        all_class_labels = []
        layer_labels = {layer: [] for layer in self.active_layers}
        
        # Mapping class ID ke layer
        class_to_layer = {}
        for layer, config in self.layer_config.items():
            for cls_id in config['class_ids']:
                class_to_layer[cls_id] = layer
        
        if label_path and label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id, x, y, w, h
                        try:
                            cls_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            
                            # Validasi nilai koordinat (harus antara 0-1)
                            if any(not (0 <= coord <= 1) for coord in bbox):
                                continue
                                
                            # Periksa apakah class ID termasuk dalam layer yang diaktifkan
                            if cls_id in class_to_layer:
                                layer = class_to_layer[cls_id]
                                if layer in self.active_layers:
                                    # Tambahkan ke list semua bboxes untuk augmentasi
                                    all_bboxes.append(bbox)
                                    all_class_labels.append(cls_id)
                                    
                                    # Tambahkan ke layer yang sesuai
                                    layer_labels[layer].append((cls_id, bbox))
                                    
                                    # Update statistik layer
                                    self.stats['layer_stats'][layer] += 1
                                    
                        except (ValueError, IndexError):
                            continue
        
        # Periksa apakah ada label yang valid
        if not all_bboxes and augmentation_type != 'lighting':
            # Untuk augmentasi position dan combined, perlu ada bbox yang valid
            # Untuk lighting, boleh tidak ada bbox
            self.stats['skipped_invalid'] += 1
            return [], [], []
        
        # Pilih pipeline augmentasi
        if augmentation_type == 'position':
            augmentor = self.position_aug
        elif augmentation_type == 'lighting':
            augmentor = self.lighting_aug
        else:
            augmentor = self.combined_aug
        
        # Generate 3 variasi
        augmented_images = []
        augmented_layer_labels = []
        output_paths = []
        
        for i in range(3):
            # Apply augmentation
            try:
                if all_bboxes:
                    # Augmentasi dengan bounding box
                    augmented = augmentor(
                        image=image,
                        bboxes=all_bboxes,
                        class_labels=all_class_labels
                    )
                    
                    # Reorganisasi hasil augmentasi ke format per layer
                    augmented_labels = {layer: [] for layer in self.active_layers}
                    
                    for bbox, cls_id in zip(augmented['bboxes'], augmented['class_labels']):
                        if cls_id in class_to_layer:
                            layer = class_to_layer[cls_id]
                            if layer in self.active_layers:
                                augmented_labels[layer].append((cls_id, bbox))
                    
                    augmented_layer_labels.append(augmented_labels)
                else:
                    # Augmentasi hanya gambar
                    augmented = augmentor(image=image)
                    
                    # Gunakan label kosong
                    augmented_labels = {layer: [] for layer in self.active_layers}
                    augmented_layer_labels.append(augmented_labels)
                
                augmented_images.append(augmented['image'])
                
                # Generate output paths
                suffix = f"{output_prefix}_{augmentation_type}_{i+1}"
                img_output_path = image_path.parent / f"{image_path.stem}_{suffix}{image_path.suffix}"
                
                output_paths.append(img_output_path)
                self.stats['augmented'] += 1
                
            except Exception as e:
                self.logger.error(f"❌ Augmentasi gagal untuk {image_path.name}: {str(e)}")
                continue
                
        self.stats['processed'] += 1
        return augmented_images, augmented_layer_labels, output_paths
    
    def save_augmented_data(
        self,
        image: np.ndarray,
        layer_labels: Dict[str, List[Tuple[int, List[float]]]],
        image_path: Path,
        labels_dir: Path
    ) -> bool:
        """
        Simpan hasil augmentasi ke file dengan format multilayer.
        
        Args:
            image: Gambar hasil augmentasi
            layer_labels: Label per layer hasil augmentasi
            image_path: Path untuk menyimpan gambar
            labels_dir: Direktori untuk menyimpan label
            
        Returns:
            Boolean sukses/gagal
        """
        try:
            # Simpan gambar
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            
            # Simpan label
            has_labels = any(len(labels) > 0 for labels in layer_labels.values())
            
            if has_labels:
                label_path = labels_dir / f"{image_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    for layer, labels in layer_labels.items():
                        for cls_id, bbox in labels:
                            if len(bbox) == 4:  # x, y, w, h
                                line = f"{int(cls_id)} {' '.join(map(str, bbox))}\n"
                                f.write(line)
                            
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan hasil augmentasi: {str(e)}")
            return False
    
    def augment_dataset(
        self,
        split: str = 'train',
        augmentation_type: str = 'combined',
        num_workers: int = 4,
        output_prefix: str = 'aug',
        validate_results: bool = True
    ) -> Dict[str, int]:
        """
        Augmentasi dataset dengan validasi layer label.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            augmentation_type: Tipe augmentasi ('position', 'lighting', atau 'combined')
            num_workers: Jumlah worker untuk multiprocessing
            output_prefix: Prefix untuk nama file output
            validate_results: Validasi hasil augmentasi
            
        Returns:
            Dict statistik augmentasi
        """
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
        
        # Periksa file gambar yang tersedia
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {images_dir}")
            return {
                'processed': 0, 
                'augmented': 0, 
                'failed': 0, 
                'skipped_invalid': 0
            }
        
        self.logger.info(f"🔍 Menemukan {len(image_files)} gambar di {split}")
        
        # Reset statistik
        self.stats = {
            'processed': 0,
            'augmented': 0,
            'failed': 0,
            'skipped_invalid': 0,
            'layer_stats': {layer: 0 for layer in self.layer_config}
        }
        
        # Fungsi untuk memproses satu gambar
        def process_image(img_path):
            local_stats = {'augmented': 0, 'failed': 0}
            
            # Cari label yang sesuai
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # Augmentasi
            aug_images, aug_layer_labels, out_paths = self.augment_image(
                img_path,
                label_path if label_path.exists() else None,
                augmentation_type,
                output_prefix
            )
            
            # Simpan hasil
            for idx, (img, out_path) in enumerate(zip(aug_images, out_paths)):
                # Ubah path ke direktori output
                save_path = output_images_dir / out_path.name
                
                # Ambil label layer yang sesuai dengan gambar ini
                layer_labels_to_save = aug_layer_labels[idx] if aug_layer_labels and idx < len(aug_layer_labels) else {}
                
                # Simpan hasil
                if self.save_augmented_data(img, layer_labels_to_save, save_path, output_labels_dir):
                    local_stats['augmented'] += 1
                else:
                    local_stats['failed'] += 1
            
            return local_stats
        
        # Proses menggunakan ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(process_image, image_files),
                total=len(image_files),
                desc=f"Augmentasi {split}"
            ))
            
        # Agregasi hasil
        for result in results:
            self.stats['augmented'] += result['augmented']
            self.stats['failed'] += result['failed']
            
        # Validasi hasil jika diminta
        if validate_results:
            self.logger.info("🔍 Memvalidasi hasil augmentasi...")
            self._validate_augmentation_results(output_dir)
            
        self.logger.success(
            f"✨ Augmentasi {split} selesai: "
            f"{self.stats['augmented']} gambar dihasilkan, "
            f"{self.stats['failed']} error, "
            f"{self.stats['skipped_invalid']} dilewati (tidak valid)"
        )
        
        # Log statistik layer
        for layer, count in self.stats['layer_stats'].items():
            if layer in self.active_layers:
                self.logger.info(f"📊 Layer '{layer}': {count} objek")
        
        return self.stats
    
    def _validate_augmentation_results(self, output_dir: Path) -> Dict[str, int]:
        """
        Validasi hasil augmentasi untuk memastikan konsistensi label.
        
        Args:
            output_dir: Direktori output augmentasi
            
        Returns:
            Dict statistik validasi
        """
        validation_stats = {
            'valid_images': 0,
            'valid_labels': 0,
            'invalid_images': 0,
            'invalid_labels': 0,
            'layer_consistency': {layer: 0 for layer in self.active_layers}
        }
        
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        
        # Validasi gambar
        augmented_images = [f for f in images_dir.glob('*.*') if 'aug' in f.name]
        
        for img_path in augmented_images:
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                validation_stats['invalid_images'] += 1
                continue
                
            validation_stats['valid_images'] += 1
            
            # Periksa label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                # Tidak ada label, tetapi gambar valid, ini mungkin baik-baik saja
                continue
                
            # Validasi label
            try:
                layer_present = {layer: False for layer in self.active_layers}
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                bbox = [float(x) for x in parts[1:5]]
                                
                                # Validasi nilai koordinat (harus antara 0-1)
                                if any(not (0 <= coord <= 1) for coord in bbox):
                                    validation_stats['invalid_labels'] += 1
                                    continue
                                    
                                # Periksa layer
                                for layer, config in self.layer_config.items():
                                    if cls_id in config['class_ids'] and layer in self.active_layers:
                                        layer_present[layer] = True
                                        break
                                
                            except (ValueError, IndexError):
                                validation_stats['invalid_labels'] += 1
                                continue
                
                validation_stats['valid_labels'] += 1
                
                # Update statistik konsistensi layer
                for layer, present in layer_present.items():
                    if present:
                        validation_stats['layer_consistency'][layer] += 1
                
            except Exception:
                validation_stats['invalid_labels'] += 1
        
        # Log hasil validasi
        self.logger.info(
            f"✅ Validasi augmentasi:\n"
            f"   Gambar valid: {validation_stats['valid_images']}/{len(augmented_images)}\n"
            f"   Label valid: {validation_stats['valid_labels']}\n"
            f"   Gambar tidak valid: {validation_stats['invalid_images']}\n"
            f"   Label tidak valid: {validation_stats['invalid_labels']}"
        )
        
        # Log konsistensi layer
        for layer, count in validation_stats['layer_consistency'].items():
            if layer in self.active_layers:
                percentage = (count / max(1, validation_stats['valid_images'])) * 100
                self.logger.info(f"📊 Konsistensi layer '{layer}': {count} ({percentage:.1f}%)")
        
        return validation_stats
    
    def clean_augmented_data(
        self,
        splits: List[str] = ['train', 'valid', 'test'],
        prefix: str = 'aug'
    ) -> Dict[str, int]:
        """
        Bersihkan file hasil augmentasi.
        
        Args:
            splits: List split yang akan dibersihkan
            prefix: Prefix file augmentasi
            
        Returns:
            Dict statistik pembersihan
        """
        stats = {'removed_images': 0, 'removed_labels': 0}
        
        for split in splits:
            # Setup direktori
            images_dir = self.data_dir / split / 'images'
            labels_dir = self.data_dir / split / 'labels'
            
            # Cari file augmentasi gambar
            aug_images = list(images_dir.glob(f'*{prefix}*.*'))
            
            # Hapus gambar
            for img_path in tqdm(aug_images, desc=f"Membersihkan {split} images"):
                try:
                    img_path.unlink()
                    stats['removed_images'] += 1
                except Exception as e:
                    self.logger.error(f"❌ Gagal menghapus {img_path}: {str(e)}")
            
            # Hapus label yang sesuai
            for img_path in aug_images:
                label_path = labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    try:
                        label_path.unlink()
                        stats['removed_labels'] += 1
                    except Exception as e:
                        self.logger.error(f"❌ Gagal menghapus {label_path}: {str(e)}")
        
        self.logger.success(
            f"✨ Pembersihan selesai: "
            f"{stats['removed_images']} gambar, "
            f"{stats['removed_labels']} label"
        )
        
        return stats