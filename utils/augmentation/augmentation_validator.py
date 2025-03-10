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
    """Validator hasil augmentasi untuk memastikan konsistensi label dan kualitas gambar."""
    
    def __init__(
        self,
        config: Dict,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi validator augmentasi."""
        super().__init__(config, output_dir, logger)
        self._lock = threading.Lock()
    
    def validate_augmentation_results(
        self, 
        output_dir: Path,
        sample_size: int = 100,
        check_image_quality: bool = True
    ) -> Dict[str, Any]:
        """Validasi hasil augmentasi untuk memastikan konsistensi label dan kualitas gambar."""
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
        
        augmented_images = self._get_augmented_images(images_dir)
        
        if not augmented_images:
            self.logger.warning(f"⚠️ Tidak ada gambar hasil augmentasi ditemukan")
            return validation_stats
        
        validation_sample = self._get_validation_sample(augmented_images, sample_size)
        
        bbox_sizes, bbox_ratios = [], []
        
        for img_path in tqdm(validation_sample, desc="Validasi Augmentasi", ncols=80):
            # Validasi gambar
            img_result = self._validate_image(img_path, check_image_quality)
            
            if not img_result['valid']:
                validation_stats['invalid_images'] += 1
                continue
            
            validation_stats['valid_images'] += 1
            if img_result.get('low_quality', False):
                validation_stats['low_quality_images'] += 1
            
            # Validasi label
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                validation_stats['bbox_stats']['empty_labels'] += 1
                continue
            
            label_result = self._validate_label(label_path, img_result['shape'])
            
            if label_result['valid']:
                validation_stats['valid_labels'] += 1
                
                # Update statistik layer dan kelas
                for layer, present in label_result.get('layers_present', {}).items():
                    if present and layer in validation_stats['layer_consistency']:
                        validation_stats['layer_consistency'][layer] += 1
                
                for cls, count in label_result.get('classes', {}).items():
                    validation_stats['class_distribution'][cls] = \
                        validation_stats['class_distribution'].get(cls, 0) + count
                
                # Kumpulkan statistik bbox
                bbox_sizes.extend(label_result.get('bbox_sizes', []))
                bbox_ratios.extend(label_result.get('bbox_ratios', []))
            else:
                validation_stats['invalid_labels'] += 1
        
        # Finalisasi statistik
        if bbox_sizes:
            validation_stats['bbox_stats']['avg_size'] = sum(bbox_sizes) / len(bbox_sizes)
        if bbox_ratios:
            validation_stats['bbox_stats']['avg_ratio'] = sum(bbox_ratios) / len(bbox_ratios)
        
        # Hitung persentase
        total_images = len(validation_sample)
        validation_stats['valid_images_percent'] = (validation_stats['valid_images'] / total_images) * 100
        validation_stats['valid_labels_percent'] = (validation_stats['valid_labels'] / total_images) * 100
        
        # Log hasil
        self._log_validation_results(validation_stats, total_images)
        
        return validation_stats
    
    def _validate_directories(self, images_dir: Path, labels_dir: Path) -> bool:
        """Validasi keberadaan direktori output."""
        if not images_dir.exists() or not labels_dir.exists():
            self.logger.warning(f"⚠️ Direktori output tidak lengkap")
            return False
        return True
    
    def _get_augmented_images(self, images_dir: Path) -> List[Path]:
        """Dapatkan daftar gambar hasil augmentasi."""
        return [f for f in images_dir.glob('*.*') if 'aug' in f.name]
    
    def _get_validation_sample(self, augmented_images: List[Path], sample_size: int) -> List[Path]:
        """Ambil sampel untuk validasi."""
        sample = random.sample(augmented_images, min(len(augmented_images), sample_size))
        self.logger.info(f"🔍 Memvalidasi {len(sample)} gambar hasil augmentasi")
        return sample
    
    def _validate_image(self, img_path: Path, check_quality: bool) -> Dict[str, Any]:
        """Validasi kualitas dan integritas gambar."""
        try:
            img = cv2.imread(str(img_path))
            if img is None or img.size == 0:
                return {'valid': False}
            
            result = {'valid': True, 'shape': img.shape}
            
            if check_quality:
                # Periksa blur
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                contrast = gray.std()
                
                if laplacian_var < 100 or contrast < 20:
                    result['low_quality'] = True
            
            return result
        except Exception as e:
            self.logger.error(f"❌ Error validasi gambar {img_path}: {str(e)}")
            return {'valid': False}
    
    def _validate_label(self, label_path: Path, img_shape: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        """Validasi label dan konsistensi layer."""
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
                        
                        # Validasi koordinat
                        if any(not (0 <= coord <= 1) for coord in bbox):
                            result['valid'] = False
                            break
                        
                        # Statistik kelas
                        result['classes'][cls_id] = result['classes'].get(cls_id, 0) + 1
                        
                        # Periksa layer
                        layer_name = self.layer_config_manager.get_layer_for_class_id(cls_id)
                        if layer_name in self.active_layers:
                            result['layers_present'][layer_name] = True
                        
                        # Statistik bbox
                        if img_shape:
                            width_px = bbox[2] * img_shape[1]
                            height_px = bbox[3] * img_shape[0]
                            result['bbox_sizes'].append(width_px * height_px)
                            
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
    
    def _log_validation_results(self, validation_stats: Dict[str, Any], total_images: int) -> None:
        """Log hasil validasi dengan format informatif."""
        self.logger.info(
            f"✅ Validasi augmentasi:\n"
            f"   Gambar valid: {validation_stats['valid_images']}/{total_images} "
            f"({validation_stats['valid_images_percent']:.1f}%)\n"
            f"   Label valid: {validation_stats['valid_labels']} "
            f"({validation_stats['valid_labels_percent']:.1f}%)\n"
            f"   Gambar kualitas rendah: {validation_stats['low_quality_images']}"
        )
        
        # Log konsistensi layer
        self.logger.info("📊 Konsistensi layer:")
        for layer, count in validation_stats['layer_consistency'].items():
            if count > 0:
                percent = (count / total_images) * 100
                self.logger.info(f"   • {layer}: {count} gambar ({percent:.1f}%)")
        
        # Log distribusi kelas
        if validation_stats['class_distribution']:
            top_classes = sorted(
                validation_stats['class_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            self.logger.info("📊 Distribusi kelas (top 5):")
            total_cls_count = sum(validation_stats['class_distribution'].values())
            for cls, count in top_classes:
                class_name = self.layer_config_manager.get_class_name(cls) or f"Class {cls}"
                percent = (count / total_cls_count) * 100
                self.logger.info(f"   • {class_name}: {count} objek ({percent:.1f}%)")
        
        # Log statistik bbox
        if validation_stats['bbox_stats']['avg_size'] > 0:
            self.logger.info(
                f"📏 Statistik bounding box:\n"
                f"   • Ukuran rata-rata: {validation_stats['bbox_stats']['avg_size']:.0f} pixel²\n"
                f"   • Rasio aspek rata-rata: {validation_stats['bbox_stats']['avg_ratio']:.2f}\n"
                f"   • Label kosong: {validation_stats['bbox_stats']['empty_labels']}"
            )