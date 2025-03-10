"""
File: smartcash/utils/augmentation/augmentation_processor.py
Author: Alfrida Sabar
Deskripsi: Prosesor augmentasi untuk memproses satu gambar dengan berbagai transformasi
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.augmentation.augmentation_base import AugmentationBase
from smartcash.utils.augmentation.augmentation_pipeline import AugmentationPipeline

class AugmentationProcessor(AugmentationBase):
    """Prosesor untuk memproses satu gambar dengan berbagai transformasi augmentasi."""
    
    def __init__(
        self,
        config: Dict,
        pipeline: AugmentationPipeline,
        output_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """Inisialisasi prosesor augmentasi."""
        super().__init__(config, output_dir, logger)
        self.pipeline = pipeline
        self.class_to_layer_map = self._create_class_layer_mapping()
    
    def _create_class_layer_mapping(self) -> Dict[int, str]:
        """Buat mapping dari class ID ke nama layer."""
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
        """Augmentasi satu gambar dengan berbagai variasi."""
        try:
            # Baca gambar
            image = cv2.imread(str(image_path))
            if image is None:
                with self._stats_lock:
                    self.stats['failed'] += 1
                return [], [], []
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Proses label
            layer_labels, all_bboxes, all_class_labels = self._extract_labels(label_path)
            
            # Validasi label untuk augmentasi non-lighting
            if not all_bboxes and augmentation_type != 'lighting':
                with self._stats_lock:
                    self.stats['skipped_invalid'] += 1
                return [], [], []
            
            # Pilih augmentator
            augmentor = self.pipeline.get_pipeline(augmentation_type)
            
            # Generate variasi
            augmented_images, augmented_layer_labels, output_paths = [], [], []
            
            for i in range(variations):
                aug_result = self._apply_augmentation(
                    augmentor, image, all_bboxes, all_class_labels, 
                    image_path, output_prefix, augmentation_type, i
                )
                
                if aug_result:
                    img, labels, out_path = aug_result
                    augmented_images.append(img)
                    augmented_layer_labels.append(labels)
                    output_paths.append(out_path)
            
            with self._stats_lock:
                self.stats['processed'] += 1
            
            return augmented_images, augmented_layer_labels, output_paths
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengaugmentasi {image_path}: {str(e)}")
            with self._stats_lock:
                self.stats['failed'] += 1
            return [], [], []
    
    def _extract_labels(self, label_path: Optional[Path]) -> Tuple[Dict, List, List]:
        """Ekstrak label dari file untuk augmentasi."""
        layer_labels = {layer: [] for layer in self.active_layers}
        all_bboxes, all_class_labels = [], []
        
        if label_path and label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                bbox = [float(x) for x in parts[1:5]]
                                
                                # Validasi koordinat
                                if any(not (0 <= coord <= 1) for coord in bbox):
                                    continue
                                
                                # Periksa layer
                                if cls_id in self.class_to_layer_map:
                                    layer_name = self.class_to_layer_map[cls_id]
                                    
                                    all_bboxes.append(bbox)
                                    all_class_labels.append(cls_id)
                                    layer_labels[layer_name].append((cls_id, bbox))
                                    
                                    with self._stats_lock:
                                        self.stats['layer_stats'][layer_name] += 1
                                    
                            except (ValueError, IndexError):
                                continue
            except Exception as e:
                self.logger.warning(f"⚠️ Error membaca label {label_path}: {str(e)}")
        
        return layer_labels, all_bboxes, all_class_labels
    
    def _apply_augmentation(
        self, 
        augmentor, 
        image: np.ndarray, 
        bboxes: List, 
        class_labels: List, 
        original_path: Path, 
        prefix: str, 
        aug_type: str, 
        variation: int
    ) -> Optional[Tuple[np.ndarray, Dict, Path]]:
        """Terapkan augmentasi dan kembalikan hasilnya."""
        try:
            if bboxes:
                augmented = augmentor(
                    image=image,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                layer_labels = {layer: [] for layer in self.active_layers}
                for bbox, cls_id in zip(augmented['bboxes'], augmented['class_labels']):
                    if cls_id in self.class_to_layer_map:
                        layer_name = self.class_to_layer_map[cls_id]
                        layer_labels[layer_name].append((cls_id, bbox))
            else:
                augmented = augmentor(image=image)
                layer_labels = {layer: [] for layer in self.active_layers}
            
            # Buat path output
            suffix = f"{prefix}_{aug_type}_{variation+1}"
            out_path = original_path.parent / f"{original_path.stem}_{suffix}{original_path.suffix}"
            
            with self._stats_lock:
                self.stats['augmented'] += 1
                self.stats['per_type'][aug_type] += 1
            
            return augmented['image'], layer_labels, out_path
        
        except Exception as e:
            self.logger.error(f"❌ Augmentasi gagal: {str(e)}")
            return None
    
    def save_augmented_data(
        self,
        image: np.ndarray,
        layer_labels: Dict[str, List[Tuple[int, List[float]]]],
        image_path: Path,
        labels_dir: Path
    ) -> bool:
        """Simpan hasil augmentasi ke file dengan format multilayer."""
        try:
            # Simpan gambar
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Simpan label jika ada
            if any(layer_labels.values()):
                labels_dir.mkdir(parents=True, exist_ok=True)
                label_path = labels_dir / f"{image_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    for layer_labels_list in layer_labels.values():
                        for cls_id, bbox in layer_labels_list:
                            if len(bbox) == 4:
                                f.write(f"{int(cls_id)} {' '.join(map(str, bbox))}\n")
            
            return True
        except Exception as e:
            self.logger.error(f"❌ Gagal menyimpan hasil augmentasi: {str(e)}")
            return False