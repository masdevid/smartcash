"""
File: smartcash/dataset/preprocessor/utils/file_processor.py
Deskripsi: Modul untuk menangani pemrosesan file gambar dan label
"""
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from smartcash.common.logger import get_logger

class FileProcessor:
    """Kelas untuk menangani pemrosesan file gambar dan label"""
    
    def __init__(self, config: Dict[str, Any]):
        """Inisialisasi FileProcessor
        
        Args:
            config: Konfigurasi pemrosesan file
        """
        self.config = config
        self.logger = get_logger()
    
    def read_image(self, image_path: Path) -> Optional[np.ndarray]:
        """Baca gambar dari path
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Array numpy berisi data gambar atau None jika gagal
        """
        try:
            with Image.open(image_path) as img:
                return np.array(img.convert('RGB'))
        except Exception as e:
            self.logger.error(f"Gagal membaca gambar {image_path}: {str(e)}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: Path) -> bool:
        """Simpan gambar ke path tujuan
        
        Args:
            image: Array numpy berisi data gambar
            output_path: Path tujuan untuk menyimpan gambar
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            os.makedirs(output_path.parent, exist_ok=True)
            img = Image.fromarray(image)
            img.save(output_path)
            return True
        except Exception as e:
            self.logger.error(f"Gagal menyimpan gambar ke {output_path}: {str(e)}")
            return False
    
    def read_label_file(self, label_path: Path) -> List[List[float]]:
        """Baca file label YOLO format
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List bounding box dalam format [class_id, x_center, y_center, width, height]
        """
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:  # Format YOLO: class x_center y_center width height
                    bbox = list(map(float, parts))
                    bboxes.append(bbox)
            
            return bboxes
        except Exception as e:
            self.logger.error(f"Gagal membaca file label {label_path}: {str(e)}")
            return []
    
    def save_label_file(self, bboxes: List[List[float]], output_path: Path) -> bool:
        """Simpan bounding box ke file label YOLO format
        
        Args:
            bboxes: List bounding box dalam format [class_id, x_center, y_center, width, height]
            output_path: Path tujuan untuk menyimpan file label
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            os.makedirs(output_path.parent, exist_ok=True)
            with open(output_path, 'w') as f:
                for bbox in bboxes:
                    line = ' '.join(map(str, bbox)) + '\n'
                    f.write(line)
            return True
        except Exception as e:
            self.logger.error(f"Gagal menyimpan file label ke {output_path}: {str(e)}")
            return False
    
    def copy_file(self, src_path: Path, dst_path: Path) -> bool:
        """Salin file dari src ke dst
        
        Args:
            src_path: Path sumber
            dst_path: Path tujuan
            
        Returns:
            True jika berhasil, False jika gagal
        """
        try:
            os.makedirs(dst_path.parent, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return True
        except Exception as e:
            self.logger.error(f"Gagal menyalin file dari {src_path} ke {dst_path}: {str(e)}")
            return False
