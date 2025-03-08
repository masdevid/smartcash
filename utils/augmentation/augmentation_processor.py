"""
File: smartcash/utils/augmentation/augmentation_processor.py
Author: Alfrida Sabar
Deskripsi: Prosesor augmentasi untuk memproses satu gambar dengan berbagai transformasi
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase
from smartcash.utils.augmentation.augmentation_pipeline import AugmentationPipeline

class AugmentationProcessor(AugmentationBase):
    """
    Prosesor untuk memproses satu gambar dengan berbagai transformasi augmentasi.
    """
    
    def __init__(
        self,
        config: Dict,
        pipeline: AugmentationPipeline,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi prosesor augmentasi.
        
        Args:
            config: Konfigurasi aplikasi
            pipeline: Pipeline augmentasi yang akan digunakan
            output_dir: Direktori output
            logger: Logger kustom
        """
        super().__init__(config, output_dir, logger)
        self.pipeline = pipeline
        
        # Mapping class ID ke layer name untuk performa
        self.class_to_layer_map = self._create_class_layer_mapping()
    
    def _create_class_layer_mapping(self) -> Dict[int, str]:
        """
        Buat mapping dari class ID ke nama layer.
        
        Returns:
            Dictionary mapping class ID ke nama layer
        """
        mapping = {}
        for layer_name in self.active_layers:
            layer_config = self.layer_config_manager.get_layer_config(layer_name)
            for cls_id in layer_config.get('class_ids', []):
                mapping[cls_id] = layer_name
        return mapping
    
    def augment_image(
        self,
        image_path: Path,
        label_path: Optional[Path] = None,
        augmentation_type: str = 'combined',
        output_prefix: str = 'aug',
        variations: int = 3
    ) -> Tuple[List[np.ndarray], List[Dict[str, List[Tuple[int, List[float]]]]], List[Path]]:
        """
        Augmentasi satu gambar dengan berbagai variasi.
        
        Args:
            image_path: Path ke file gambar
            label_path: Path ke file label (opsional)
            augmentation_type: Tipe augmentasi 
            output_prefix: Prefix untuk nama file output
            variations: Jumlah variasi yang dihasilkan
            
        Returns:
            Tuple of (augmented_images, augmented_layer_labels, output_paths)
        """
        try:
            # Baca gambar
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.warning(f"⚠️ Tidak dapat membaca gambar: {image_path}")
                with self._stats_lock:
                    self.stats['failed'] += 1
                return [], [], []
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Baca dan validasi label
            all_bboxes = []
            all_class_labels = []
            layer_labels = {layer: [] for layer in self.active_layers}
            
            if label_path and label_path.exists():
                try:
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
                                    if cls_id in self.class_to_layer_map:
                                        layer_name = self.class_to_layer_map[cls_id]
                                        
                                        # Tambahkan ke list semua bboxes untuk augmentasi
                                        all_bboxes.append(bbox)
                                        all_class_labels.append(cls_id)
                                        
                                        # Tambahkan ke layer yang sesuai
                                        layer_labels[layer_name].append((cls_id, bbox))
                                        
                                        # Update statistik layer
                                        with self._stats_lock:
                                            self.stats['layer_stats'][layer_name] += 1
                                        
                                except (ValueError, IndexError):
                                    continue
                except Exception as e:
                    self.logger.warning(f"⚠️ Error membaca label {label_path}: {str(e)}")
            
            # Periksa apakah ada label yang valid
            if not all_bboxes and augmentation_type != 'lighting':
                # Untuk augmentasi selain lighting, perlu ada bbox
                with self._stats_lock:
                    self.stats['skipped_invalid'] += 1
                return [], [], []
            
            # Pilih pipeline augmentasi
            augmentor = self.pipeline.get_pipeline(augmentation_type)
            
            # Generate variasi
            augmented_images = []
            augmented_layer_labels = []
            output_paths = []
            
            for i in range(variations):
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
                            if cls_id in self.class_to_layer_map:
                                layer_name = self.class_to_layer_map[cls_id]
                                augmented_labels[layer_name].append((cls_id, bbox))
                        
                        augmented_layer_labels.append(augmented_labels)
                    else:
                        # Augmentasi hanya gambar
                        augmented = augmentor(image=image)
                        
                        # Gunakan label kosong
                        augmented_layer_labels.append({layer: [] for layer in self.active_layers})
                    
                    augmented_images.append(augmented['image'])
                    
                    # Generate output paths
                    suffix = f"{output_prefix}_{augmentation_type}_{i+1}"
                    img_output_path = image_path.parent / f"{image_path.stem}_{suffix}{image_path.suffix}"
                    
                    output_paths.append(img_output_path)
                    with self._stats_lock:
                        self.stats['augmented'] += 1
                        self.stats['per_type'][augmentation_type] += 1
                    
                except Exception as e:
                    self.logger.error(f"❌ Augmentasi gagal untuk {image_path.name} ({augmentation_type}): {str(e)}")
                    continue
                    
            with self._stats_lock:
                self.stats['processed'] += 1
                
            return augmented_images, augmented_layer_labels, output_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengaugmentasi {image_path}: {str(e)}")
            with self._stats_lock:
                self.stats['failed'] += 1
            return [], [], []
    
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
            img_dir = image_path.parent
            img_dir.mkdir(parents=True, exist_ok=True)
            
            cv2.imwrite(
                str(image_path),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            
            # Simpan label jika ada
            has_labels = any(len(labels) > 0 for labels in layer_labels.values())
            
            if has_labels:
                labels_dir.mkdir(parents=True, exist_ok=True)
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