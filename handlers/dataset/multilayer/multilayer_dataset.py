# File: smartcash/handlers/dataset/multilayer/multilayer_dataset.py
# Author: Alfrida Sabar
# Deskripsi: Dataset multilayer yang terintegrasi dengan utils/dataset

import torch
import numpy as np
import traceback
import cv2
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.observer import EventTopics, notify
from smartcash.handlers.dataset.multilayer.multilayer_dataset_base import MultilayerDatasetBase
from smartcash.handlers.dataset.multilayer.multilayer_label_handler import MultilayerLabelHandler
from smartcash.handlers.dataset.dataset_utils_adapter import DatasetUtilsAdapter


class MultilayerDataset(MultilayerDatasetBase):
    """Dataset multilayer dengan integrasi utils/dataset melalui adapter."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        layers: Optional[List[str]] = None,
        require_all_layers: bool = False,
        logger: Optional[SmartCashLogger] = None,
        config: Optional[Dict] = None
    ):
        """Inisialisasi MultilayerDataset."""
        super().__init__(data_path=data_path, img_size=img_size, mode=mode, 
                         transform=transform, layers=layers, logger=logger)
        
        self.require_all_layers = require_all_layers
        self.config = config or {}
        
        # Notifikasi inisialisasi
        notify(EventTopics.PREPROCESSING, self, action="dataset_init", 
               data_path=str(self.data_path), mode=mode, layers=self.layers)
        
        # Init komponen
        self.label_handler = MultilayerLabelHandler(layers=self.layers, logger=self.logger)
        self.utils_adapter = DatasetUtilsAdapter(
            config=self.config if self.config else {'layers': self.layers},
            data_dir=str(self.data_path),
            logger=self.logger
        )
        
        # Validasi dataset
        self.image_files = self._find_image_files()
        self.valid_samples = self._validate_dataset()
        
        # Notifikasi selesai
        notify(EventTopics.PREPROCESSING, self, action="dataset_init_complete",
               data_path=str(self.data_path), valid_samples=len(self.valid_samples))
        
        if len(self.valid_samples) == 0:
            self.logger.warning(f"⚠️ Tidak ada sampel valid di {self.data_path}")
    
    def _validate_dataset(self) -> List[Dict]:
        """Validasi dataset menggunakan utils_adapter."""
        try:
            # Dapatkan file valid
            valid_files = self.utils_adapter.validator.get_valid_files(
                data_dir=str(self.data_path),
                split=self.data_path.name,
                check_images=True,
                check_labels=True
            )
            
            # Filter berdasarkan layer
            result = []
            for file_info in valid_files:
                label_path = Path(file_info['label_path'])
                available_layers = self.label_handler.get_available_layers(label_path)
                
                # Cek apakah file mempunyai layer yang diperlukan
                if self.require_all_layers:
                    # Harus ada semua layer
                    if all(layer in available_layers for layer in self.layers):
                        file_info['available_layers'] = available_layers
                        result.append(file_info)
                else:
                    # Cukup ada satu layer
                    if any(layer in available_layers for layer in self.layers):
                        file_info['available_layers'] = available_layers
                        result.append(file_info)
            
            return result
        except Exception as e:
            self.logger.error(f"❌ Validasi multilayer gagal: {str(e)}")
            return []
    
    def __getitem__(self, idx: int) -> Dict:
        """Ambil item dataset."""
        if idx >= len(self.valid_samples):
            raise IndexError(f"Indeks {idx} di luar batas dataset (ukuran: {len(self.valid_samples)})")
            
        sample = self.valid_samples[idx]
        
        # Load gambar dan label
        try:
            img = self._load_image(sample['image_path'])
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal baca gambar {sample['image_path']}: {str(e)}")
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # Parse label per layer
        layer_labels = self.label_handler.parse_label_by_layer(sample['label_path'])
        
        # Flatten bbox dan class untuk augmentasi
        all_bboxes, all_classes = [], []
        for layer in self.layers:
            for cls_id, bbox in layer_labels[layer]:
                all_bboxes.append(bbox)
                all_classes.append(cls_id)
        
        # Augmentasi jika diperlukan
        if self.transform:
            try:
                transformed = self.transform(
                    image=img,
                    bboxes=all_bboxes,
                    class_labels=all_classes
                )
                img = transformed['image']
                
                # Reorganisasi hasil augmentasi ke layer
                if 'class_labels' in transformed and 'bboxes' in transformed:
                    augmented_layer_labels = {layer: [] for layer in self.layers}
                    for cls_id, bbox in zip(transformed['class_labels'], transformed['bboxes']):
                        if cls_id in self.class_to_layer and self.class_to_layer[cls_id] in self.layers:
                            layer = self.class_to_layer[cls_id]
                            augmented_layer_labels[layer].append((cls_id, bbox))
                    layer_labels = augmented_layer_labels
            except Exception as e:
                self.logger.debug(f"⚠️ Augmentasi gagal: {traceback.format_exc()}")
        elif img.shape[:2] != self.img_size:
            # Resize jika tidak ada transform
            img = cv2.resize(img, self.img_size)
        
        # Konversi gambar ke tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Buat tensor target per layer
        targets = {}
        for layer in self.layers:
            if layer in sample['available_layers']:
                targets[layer] = self.label_handler.create_layer_tensor(layer, layer_labels[layer])
            else:
                targets[layer] = torch.zeros((self.get_layer_dimension(layer), 5))
        
        # Return hasil dengan metadata
        return {
            'image': img_tensor,
            'targets': targets,
            'metadata': {
                'image_path': str(sample['image_path']),
                'label_path': str(sample['label_path']),
                'available_layers': sample['available_layers']
            }
        }