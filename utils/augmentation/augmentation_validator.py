"""
File: smartcash/utils/augmentation/augmentation_validator.py
Author: Alfrida Sabar
Deskripsi: Validator hasil augmentasi untuk memastikan konsistensi label dan kualitas gambar
"""

import cv2
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from tqdm.auto import tqdm
import threading

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase

class AugmentationValidator(AugmentationBase):
    """
    Validator hasil augmentasi untuk memastikan konsistensi label dan kualitas gambar.
    
    Menyediakan kemampuan untuk:
    - Memvalidasi hasil augmentasi secara otomatis
    - Memeriksa integritas gambar
    - Memvalidasi konsistensi label antar layer
    - Menghasilkan statistik validasi komprehensif
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi validator augmentasi.
        
        Args:
            config: Konfigurasi aplikasi
            output_dir: Direktori output
            logger: Logger kustom
        """
        super().__init__(config, output_dir, logger)
        self._lock = threading.Lock()
    
    def validate_augmentation_results(
        self, 
        output_dir: Path,
        sample_size: int = 100,
        check_image_quality: bool = True
    ) -> Dict[str, Any]:
        """
        Validasi hasil augmentasi untuk memastikan konsistensi label dan kualitas gambar.
        
        Args:
            output_dir: Direktori output augmentasi
            sample_size: Jumlah maksimal sampel untuk validasi
            check_image_quality: Flag untuk memeriksa kualitas gambar
            
        Returns:
            Dict statistik validasi
        """
        validation_stats = {
            'valid_images': 0,
            'valid_labels': 0,
            'invalid_images': 0,
            'invalid_labels': 0,
            'low_quality_images': 0,
            'layer_consistency': {layer: 0 for layer in self.active_layers},
            'class_distribution': {},
            'bbox_stats': {
                'avg_size': 0,
                'avg_ratio': 0,
                'empty_labels': 0
            }
        }
        
        images_dir = output_dir / 'images'
        labels_dir = output_dir / 'labels'
        
        if not self._validate_directories(images_dir, labels_dir):
            return validation_stats
        
        # Validasi gambar yang diaugmentasi (berisi 'aug' di nama file)
        augmented_images = self._get_augmented_images(images_dir)
        
        if not augmented_images:
            self.logger.warning(f"⚠️ Tidak ada gambar hasil augmentasi ditemukan")
            return validation_stats
        
        # Sampel untuk validasi (max sample_size gambar untuk performa)
        validation_sample = self._get_validation_sample(augmented_images, sample_size)
        
        # Validasi sampel
        self._validate_samples(
            validation_sample, 
            labels_dir, 
            validation_stats, 
            check_image_quality
        )
        
        # Ekstrapolasi hasil untuk seluruh dataset
        if len(validation_sample) < len(augmented_images):
            self._extrapolate_results(validation_stats, len(validation_sample), len(augmented_images))
        
        # Finalisasi statistik
        self._finalize_statistics(validation_stats, len(augmented_images))
        
        # Log hasil validasi
        self._log_validation_results(validation_stats, len(augmented_images))
        
        return validation_stats
    
    def _validate_directories(self, images_dir: Path, labels_dir: Path) -> bool:
        """Validasi keberadaan direktori output."""
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.warning(f"⚠️ Direktori output tidak lengkap untuk validasi: {images_dir}")
            return False
        return True
    
    def _get_augmented_images(self, images_dir: Path) -> List[Path]:
        """Dapatkan daftar gambar hasil augmentasi."""
        return [f for f in images_dir.glob('*.*') if 'aug' in f.name]
    
    def _get_validation_sample(self, augmented_images: List[Path], sample_size: int) -> List[Path]:
        """Ambil sampel untuk validasi."""
        if len(augmented_images) > sample_size:
            validation_sample = random.sample(augmented_images, sample_size)
            self.logger.info(f"   Menggunakan sampel {sample_size} gambar untuk validasi")
        else:
            validation_sample = augmented_images
        
        self.logger.info(f"🔍 Memvalidasi {len(validation_sample)} gambar hasil augmentasi...")
        return validation_sample
    
    def _validate_samples(
        self, 
        validation_sample: List[Path], 
        labels_dir: Path, 
        validation_stats: Dict[str, Any],
        check_image_quality: bool
    ) -> None:
        """Validasi sampel gambar dan label."""
        bbox_sizes = []
        bbox_ratios = []
        
        for img_path in tqdm(validation_sample, desc="Validasi Augmentasi", ncols=80):
            # Validasi gambar
            img_validation = self._validate_image(img_path, check_image_quality)
            
            # Update statistik gambar
            if img_validation['valid']:
                validation_stats['valid_images'] += 1
                if img_validation.get('low_quality', False):
                    validation_stats['low_quality_images'] += 1
            else:
                validation_stats['invalid_images'] += 1
                continue
            
            # Validasi label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                validation_stats['bbox_stats']['empty_labels'] += 1
                continue
            
            # Proses validasi label
            label_result = self._validate_label(
                label_path, 
                img_validation['shape'] if 'shape' in img_validation else None
            )
            
            # Update statistik label
            if label_result['valid']:
                validation_stats['valid_labels'] += 1
                
                # Update statistik bbox
                bbox_sizes.extend(label_result.get('bbox_sizes', []))
                bbox_ratios.extend(label_result.get('bbox_ratios', []))
                
                # Update statistik layer
                for layer, present in label_result.get('layers_present', {}).items():
                    if present and layer in validation_stats['layer_consistency']:
                        validation_stats['layer_consistency'][layer] += 1
                
                # Update distribusi kelas
                for cls, count in label_result.get('classes', {}).items():
                    if cls not in validation_stats['class_distribution']:
                        validation_stats['class_distribution'][cls] = 0
                    validation_stats['class_distribution'][cls] += count
            else:
                validation_stats['invalid_labels'] += 1
        
        # Hitung statistik bbox
        if bbox_sizes:
            validation_stats['bbox_stats']['avg_size'] = sum(bbox_sizes) / len(bbox_sizes)
        if bbox_ratios:
            validation_stats['bbox_stats']['avg_ratio'] = sum(bbox_ratios) / len(bbox_ratios)
    
    def _validate_image(self, img_path: Path, check_quality: bool) -> Dict[str, Any]:
        """
        Validasi kualitas dan integritas gambar.
        
        Args:
            img_path: Path ke file gambar
            check_quality: Flag untuk memeriksa kualitas gambar
            
        Returns:
            Dict dengan status validasi dan informasi tambahan
        """
        result = {'valid': False}
        
        try:
            # Baca gambar
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                return result
            
            result['valid'] = True
            result['shape'] = img.shape
            
            # Periksa kualitas gambar jika diminta
            if check_quality:
                # Periksa blur
                laplacian_var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                if laplacian_var < 100:  # Threshold untuk blur
                    result['low_quality'] = True
                
                # Periksa kontras
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                contrast = gray.std()
                if contrast < 20:  # Threshold untuk kontras rendah
                    result['low_quality'] = True
            
            return result
        except Exception as e:
            self.logger.error(f"❌ Error validasi gambar {img_path}: {str(e)}")
            return result
    
    def _validate_label(self, label_path: Path, img_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """
        Validasi label dan konsistensi layer.
        
        Args:
            label_path: Path ke file label
            img_shape: Bentuk gambar (height, width, channels)
            
        Returns:
            Dict dengan status validasi dan informasi tambahan
        """
        result = {
            'valid': False,
            'layers_present': {layer: False for layer in self.active_layers},
            'bbox_sizes': [],
            'bbox_ratios': [],
            'classes': {}
        }
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return result
            
            result['valid'] = True
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(float(parts[0]))
                        bbox = [float(x) for x in parts[1:5]]
                        
                        # Validasi nilai koordinat
                        if any(not (0 <= coord <= 1) for coord in bbox):
                            result['valid'] = False
                            break
                        
                        # Track class distribution
                        if cls_id not in result['classes']:
                            result['classes'][cls_id] = 0
                        result['classes'][cls_id] += 1
                        
                        # Periksa layer
                        layer_name = self.layer_config_manager.get_layer_for_class_id(cls_id)
                        if layer_name in self.active_layers:
                            result['layers_present'][layer_name] = True
                        
                        # Hitung statistik bbox
                        if img_shape:
                            # bbox dalam format YOLO: x_center, y_center, width, height
                            width_px = bbox[2] * img_shape[1]
                            height_px = bbox[3] * img_shape[0]
                            result['bbox_sizes'].append(width_px * height_px)
                            
                            # Rasio aspek
                            if height_px > 0:
                                result['bbox_ratios'].append(width_px / height_px)
                        
                    except (ValueError, IndexError, ZeroDivisionError):
                        result['valid'] = False
                        break
                else:
                    result['valid'] = False
                    break
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error validasi label {label_path}: {str(e)}")
            return result
    
    def _extrapolate_results(
        self, 
        validation_stats: Dict[str, Any], 
        sample_size: int, 
        total_size: int
    ) -> None:
        """Ekstrapolasi hasil validasi sampel ke seluruh dataset."""
        scaling_factor = total_size / max(1, sample_size)
        
        for key in ['valid_images', 'valid_labels', 'invalid_images', 'invalid_labels', 'low_quality_images']:
            validation_stats[key] = int(validation_stats[key] * scaling_factor)
        
        for layer in validation_stats['layer_consistency']:
            validation_stats['layer_consistency'][layer] = int(
                validation_stats['layer_consistency'][layer] * scaling_factor
            )
        
        for cls in validation_stats['class_distribution']:
            validation_stats['class_distribution'][cls] = int(
                validation_stats['class_distribution'][cls] * scaling_factor
            )
    
    def _finalize_statistics(self, validation_stats: Dict[str, Any], total_images: int) -> None:
        """Finalisasi perhitungan statistik."""
        # Persentase gambar valid
        validation_stats['valid_images_percent'] = (validation_stats['valid_images'] / max(1, total_images)) * 100
        
        # Persentase label valid
        if validation_stats['valid_images'] > 0:
            validation_stats['valid_labels_percent'] = (validation_stats['valid_labels'] / max(1, validation_stats['valid_images'])) * 100
        else:
            validation_stats['valid_labels_percent'] = 0
        
        # Layer coverage
        layer_coverage = {}
        for layer, count in validation_stats['layer_consistency'].items():
            layer_coverage[layer] = (count / max(1, validation_stats['valid_images'])) * 100
        validation_stats['layer_coverage'] = layer_coverage
    
    def _log_validation_results(
        self, 
        validation_stats: Dict[str, Any], 
        total_images: int
    ) -> None:
        """Log hasil validasi dengan format yang informatif."""
        # Log statistik dasar
        self.logger.info(
            f"✅ Validasi augmentasi:\n"
            f"   Gambar valid: {validation_stats['valid_images']}/{total_images} "
            f"({validation_stats['valid_images_percent']:.1f}%)\n"
            f"   Label valid: {validation_stats['valid_labels']} "
            f"({validation_stats['valid_labels_percent']:.1f}%)\n"
            f"   Gambar tidak valid: {validation_stats['invalid_images']}\n"
            f"   Label tidak valid: {validation_stats['invalid_labels']}\n"
            f"   Gambar kualitas rendah: {validation_stats['low_quality_images']}"
        )
        
        # Log konsistensi layer
        self.logger.info("📊 Konsistensi layer:")
        for layer, percent in validation_stats['layer_coverage'].items():
            count = validation_stats['layer_consistency'][layer]
            self.logger.info(f"   • {layer}: {count} gambar ({percent:.1f}%)")
        
        # Log distribusi kelas (top 5)
        if validation_stats['class_distribution']:
            top_classes = sorted(
                validation_stats['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            self.logger.info("📊 Distribusi kelas (top 5):")
            for cls, count in top_classes:
                class_name = self.layer_config_manager.get_class_name(cls) or f"Class {cls}"
                percent = (count / sum(validation_stats['class_distribution'].values())) * 100
                self.logger.info(f"   • {class_name}: {count} objek ({percent:.1f}%)")
        
        # Log statistik bbox
        if 'avg_size' in validation_stats['bbox_stats'] and validation_stats['bbox_stats']['avg_size'] > 0:
            self.logger.info(
                f"📏 Statistik bounding box:\n"
                f"   • Ukuran rata-rata: {validation_stats['bbox_stats']['avg_size']:.0f} pixel²\n"
                f"   • Rasio aspek rata-rata: {validation_stats['bbox_stats']['avg_ratio']:.2f}\n"
                f"   • Label kosong: {validation_stats['bbox_stats']['empty_labels']}"
            )