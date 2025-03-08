# File: smartcash/handlers/dataset/multilayer/multilayer_dataset.py
# Author: Alfrida Sabar
# Deskripsi: Dataset multilayer yang terintegrasi dengan utils/dataset

import torch
import numpy as np
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset import EnhancedDatasetValidator
from smartcash.handlers.dataset.multilayer.multilayer_dataset_base import MultilayerDatasetBase
from smartcash.handlers.dataset.multilayer.multilayer_label_handler import MultilayerLabelHandler
from smartcash.handlers.dataset.integration.validator_adapter import DatasetValidatorAdapter
from smartcash.factories.dataset_component_factory import DatasetComponentFactory


class MultilayerDataset(MultilayerDatasetBase):
    """
    Dataset untuk deteksi multilayer dengan integrasi validator dari utils/dataset.
    
    Versi refaktor yang menghindari duplikasi dengan EnhancedDatasetValidator.
    """
    
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
        """
        Inisialisasi MultilayerDataset.
        
        Args:
            data_path: Path ke direktori dataset
            img_size: Ukuran target gambar
            mode: Mode dataset ('train', 'val', 'test')
            transform: Transformasi kustom
            layers: Daftar layer yang akan diaktifkan
            require_all_layers: Jika True, hanya gambar dengan semua layer yang akan digunakan
            logger: Logger kustom
            config: Konfigurasi tambahan (opsional)
        """
        super().__init__(
            data_path=data_path,
            img_size=img_size,
            mode=mode,
            transform=transform,
            layers=layers,
            logger=logger
        )
        
        self.require_all_layers = require_all_layers
        self.config = config or {}
        
        # Inisialisasi label handler
        self.label_handler = MultilayerLabelHandler(
            layers=self.layers,
            logger=self.logger
        )
        
        # Inisialisasi validator dari utils/dataset dengan adapter pattern
        validator = EnhancedDatasetValidator(
            config={'layers': self.layers},
            data_dir=str(self.data_path),
            logger=self.logger,
            num_workers=1  # Gunakan 1 worker untuk validasi dalam proses ini
        )
        
        self.validator_adapter = DatasetValidatorAdapter(
            validator=validator,
            layers=self.layers,
            logger=self.logger
        )
        
        # Cari file gambar dan validasi label
        self.image_files = self._find_image_files()
        self.valid_samples = self._validate_dataset()
        
        if len(self.valid_samples) == 0:
            self.logger.warning(f"⚠️ Tidak ada sampel valid yang ditemukan di {self.data_path}")
    
    def _validate_dataset(self) -> List[Dict]:
        """
        Validasi dataset menggunakan validator adapter.
        
        Returns:
            List info sampel valid
        """
        # Gunakan validator adapter dari integrasi utils/dataset
        return self.validator_adapter.validate_dataset_for_multilayer(
            data_path=self.data_path,
            require_all_layers=self.require_all_layers
        )
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Mendapatkan item dataset dengan format multilayer.
        
        Args:
            idx: Indeks sampel
            
        Returns:
            Dict berisi gambar dan label per layer
        """
        if idx >= len(self.valid_samples):
            raise IndexError(f"Indeks {idx} di luar batas dataset (ukuran: {len(self.valid_samples)})")
            
        sample = self.valid_samples[idx]
        
        # Load gambar
        try:
            img = self._load_image(sample['image_path'])
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca gambar {sample['image_path']}: {str(e)}")
            # Return dummy image sebagai fallback
            img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
        
        # Parse label per layer
        layer_labels = self.label_handler.parse_label_by_layer(sample['label_path'])
        
        # Siapkan data untuk augmentasi
        all_bboxes = []
        all_classes = []
        for layer in self.layers:
            for cls_id, bbox in layer_labels[layer]:
                all_bboxes.append(bbox)
                all_classes.append(cls_id)
        
        # Augmentasi
        if self.transform:
            try:
                transformed = self.transform(
                    image=img,
                    bboxes=all_bboxes,
                    class_labels=all_classes
                )
                img = transformed['image']
                
                # Reorganisasi hasil augmentasi ke format per layer
                augmented_layer_labels = {layer: [] for layer in self.layers}
                
                # Pastikan transformed memiliki class_labels dan bboxes
                if 'class_labels' in transformed and 'bboxes' in transformed:
                    for i, (cls_id, bbox) in enumerate(zip(transformed['class_labels'], transformed['bboxes'])):
                        if cls_id in self.class_to_layer:
                            layer = self.class_to_layer[cls_id]
                            augmented_layer_labels[layer].append((cls_id, bbox))
                
                    layer_labels = augmented_layer_labels
                
            except Exception as e:
                self.logger.warning(f"⚠️ Augmentasi gagal untuk {sample['image_path']}: {str(e)}")
                # Error details untuk debugging
                self.logger.debug(f"⚠️ Detail error: {traceback.format_exc()}")
        else:
            # Resize gambar jika tidak ada augmentasi
            img = cv2.resize(img, self.img_size)
        
        # Konversi gambar ke tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Buat tensor target untuk setiap layer
        targets = {}
        for layer in self.layers:
            if layer in sample['available_layers']:
                # Layer ini tersedia untuk gambar ini
                targets[layer] = self.label_handler.create_layer_tensor(layer, layer_labels[layer])
            else:
                # Layer tidak tersedia, buat tensor kosong
                num_classes = self.get_layer_dimension(layer)
                targets[layer] = torch.zeros((num_classes, 5))
        
        # Tambahkan informasi tambahan untuk debugging
        metadata = {
            'image_path': str(sample['image_path']),
            'label_path': str(sample['label_path']),
            'available_layers': sample['available_layers']
        }
        
        # Return hasil
        return {
            'image': img_tensor,
            'targets': targets,
            'metadata': metadata
        }
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        data_path: str,
        mode: str = 'train',
        transform = None,
        logger: Optional[SmartCashLogger] = None
    ) -> 'MultilayerDataset':
        """
        Membuat dataset dari konfigurasi menggunakan factory.
        
        Args:
            config: Konfigurasi dataset
            data_path: Path ke direktori dataset
            mode: Mode dataset ('train', 'val', 'test')
            transform: Transformasi kustom
            logger: Logger kustom
            
        Returns:
            Instance MultilayerDataset
        """
        return DatasetComponentFactory.create_multilayer_dataset(
            config=config,
            data_path=data_path,
            mode=mode,
            transform=transform,
            logger=logger
        )