# File: smartcash/handlers/dataset/multilayer_dataset_base.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk dataset multilayer

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class MultilayerDatasetBase(Dataset):
    """Kelas dasar untuk dataset multilayer dengan fungsionalitas umum."""
    
    def __init__(
        self,
        data_path: str,
        img_size: Tuple[int, int] = (640, 640),
        mode: str = 'train',
        transform = None,
        layers: Optional[List[str]] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi MultilayerDatasetBase.
        
        Args:
            data_path: Path ke direktori dataset
            img_size: Ukuran target gambar
            mode: Mode dataset ('train', 'val', 'test')
            transform: Transformasi kustom
            layers: Daftar layer yang akan diaktifkan
            logger: Logger kustom
        """
        self.data_path = Path(data_path)
        self.img_size = img_size
        self.mode = mode
        self.transform = transform
        self.logger = logger or SmartCashLogger("multilayer_dataset")
        
        # Dapatkan konfigurasi layer dari layer config manager
        self.layer_config_manager = get_layer_config()
        self.layers = layers or self.layer_config_manager.get_layer_names()
        
        # Setup jalur direktori
        self.images_dir = self.data_path / 'images'
        self.labels_dir = self.data_path / 'labels'
        
        # Validasi direktori
        if not self.images_dir.exists():
            self.logger.warning(f"⚠️ Direktori gambar tidak ditemukan: {self.images_dir}")
            self.images_dir.mkdir(parents=True, exist_ok=True)
            
        if not self.labels_dir.exists():
            self.logger.warning(f"⚠️ Direktori label tidak ditemukan: {self.labels_dir}")
            self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Class ID to layer mapping untuk lookup cepat
        self.class_to_layer = {}
        for layer in self.layers:
            layer_config = self.layer_config_manager.get_layer_config(layer)
            for cls_id in layer_config['class_ids']:
                self.class_to_layer[cls_id] = layer
        
        # Initialized di kelas turunan
        self.valid_samples = []
        
    def __len__(self) -> int:
        """Mendapatkan jumlah sampel valid."""
        return len(self.valid_samples)
        
    def _load_image(self, image_path: Path) -> np.ndarray:
        """
        Membaca gambar dari file.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Array gambar (RGB)
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Gagal membaca gambar: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca gambar {image_path}: {str(e)}")
            # Return dummy image sebagai fallback
            return np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
    
    def _find_image_files(self) -> List[Path]:
        """
        Menemukan semua file gambar dalam direktori.
        
        Returns:
            List path file gambar
        """
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(self.images_dir.glob(ext)))
        
        self.logger.info(f"🔍 Menemukan {len(image_files)} file gambar di {self.images_dir}")
        return image_files
    
    def get_layer_dimension(self, layer: str) -> int:
        """
        Mendapatkan dimensi layer (jumlah kelas).
        
        Args:
            layer: Nama layer
            
        Returns:
            Jumlah kelas dalam layer
        """
        layer_config = self.layer_config_manager.get_layer_config(layer)
        return len(layer_config['classes'])
    
    def get_layer_class_ids(self, layer: str) -> List[int]:
        """
        Mendapatkan class ID untuk layer tertentu.
        
        Args:
            layer: Nama layer
            
        Returns:
            List class ID
        """
        layer_config = self.layer_config_manager.get_layer_config(layer)
        return layer_config['class_ids']