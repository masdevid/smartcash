# File: smartcash/handlers/detection/core/preprocessor.py
# Author: Alfrida Sabar
# Deskripsi: Preprocessor gambar untuk deteksi mata uang

import os
from pathlib import Path
from typing import Dict, Any, Union, Tuple, Optional, List
import torch
import numpy as np
import cv2
from PIL import Image

from smartcash.exceptions.base import DataError

class ImagePreprocessor:
    """
    Preprocessor gambar untuk deteksi mata uang.
    Mengubah gambar menjadi tensor yang siap digunakan oleh model deteksi.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        img_size: Union[Tuple[int, int], int] = (640, 640)
    ):
        """
        Inisialisasi preprocessor.
        
        Args:
            config: Konfigurasi
            img_size: Ukuran gambar target (w, h) atau single int
        """
        self.config = config
        
        # Konversi img_size ke tuple jika integer
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        
        # Parameter preprocessing dari konfigurasi
        self.normalize = config.get('data', {}).get('preprocessing', {}).get('normalize_enabled', True)
        
        # Cache untuk gambar original untuk postprocessing
        self.original_shapes = {}
        
    def process(
        self,
        image_source: Union[str, Path, np.ndarray],
        return_original: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Preprocess gambar untuk model deteksi.
        
        Args:
            image_source: Path ke gambar atau array numpy
            return_original: Flag untuk mengembalikan gambar original
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary dengan:
                - 'tensor': Tensor gambar yang siap untuk deteksi
                - 'original_image': Gambar original (optional)
                - 'original_shape': Shape gambar original
                - 'source': Path gambar jika disediakan
        """
        # Baca gambar
        try:
            # Jika sumber adalah path file
            if isinstance(image_source, (str, Path)):
                image_path = Path(image_source)
                if not image_path.exists():
                    raise DataError(f"File gambar tidak ditemukan: {image_path}")
                    
                # Baca gambar dengan OpenCV
                img = cv2.imread(str(image_path))
                if img is None:
                    raise DataError(f"Gagal membaca gambar: {image_path}")
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                source = str(image_path)
            else:
                # Asumsikan sumber adalah array numpy
                img = image_source
                if not isinstance(img, np.ndarray):
                    raise DataError(f"Format gambar tidak didukung: {type(image_source)}")
                source = "array"
                
            # Simpan shape original untuk rescaling hasil
            original_shape = img.shape[:2]  # (height, width)
            
            # Simpan gambar original jika diminta
            original_image = img.copy() if return_original else None
            
            # Resize gambar
            img_resized = cv2.resize(img, self.img_size)
            
            # Normalisasi (asumsi range 0-255)
            if self.normalize:
                img_resized = img_resized.astype(np.float32) / 255.0
            
            # HWC ke CHW (untuk PyTorch)
            img_chw = np.transpose(img_resized, (2, 0, 1))
            
            # Tambahkan dimensi batch
            img_batch = np.expand_dims(img_chw, 0)
            
            # Konversi ke tensor PyTorch
            img_tensor = torch.from_numpy(img_batch).float()
            
            # Siapkan output
            result = {
                'tensor': img_tensor,
                'original_shape': original_shape,
                'source': source
            }
            
            # Tambahkan gambar original jika diminta
            if return_original:
                result['original_image'] = original_image
                
            # Cache original shape dengan path sebagai key
            self.original_shapes[source] = original_shape
            
            return result
            
        except DataError:
            # Re-raise DataError
            raise
        except Exception as e:
            raise DataError(f"Error saat preprocessing gambar: {str(e)}")
            
    def get_original_shape(self, source: str) -> Optional[Tuple[int, int]]:
        """
        Dapatkan shape original dari cache.
        
        Args:
            source: Path atau identifier gambar
            
        Returns:
            Tuple (height, width) atau None jika tidak ditemukan
        """
        return self.original_shapes.get(source)