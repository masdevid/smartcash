# File: utils/coordinate_normalizer.py
# Author: Alfrida Sabar
# Deskripsi: Modul untuk normalisasi koordinat polygon dan bounding box pada label anotasi

import numpy as np
from typing import List, Optional, Tuple, Union, Dict
from pathlib import Path
import cv2
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger

class CoordinateNormalizer:
    """Handler untuk normalisasi koordinat pada label anotasi"""
    
    def __init__(
        self,
        logger: Optional[SmartCashLogger] = None,
        num_workers: int = 4
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.num_workers = num_workers
        
    def normalize_polygon(
        self,
        points: List[Tuple[float, float]],
        image_size: Tuple[int, int]
    ) -> List[float]:
        """
        Normalisasi koordinat polygon
        Args:
            points: List koordinat [(x1,y1), (x2,y2), ...]
            image_size: (width, height) gambar
        Returns:
            List koordinat ternormalisasi [x1,y1,x2,y2,...]
        """
        width, height = image_size
        
        # Flatten dan normalisasi koordinat
        normalized = []
        for x, y in points:
            normalized.extend([
                max(0.0, min(1.0, x / width)),
                max(0.0, min(1.0, y / height))
            ])
            
        return normalized
        
    def denormalize_polygon(
        self,
        normalized: List[float],
        image_size: Tuple[int, int]
    ) -> List[Tuple[float, float]]:
        """
        Denormalisasi koordinat polygon
        Args:
            normalized: List koordinat ternormalisasi [x1,y1,x2,y2,...]
            image_size: (width, height) gambar
        Returns:
            List koordinat asli [(x1,y1), (x2,y2), ...]
        """
        width, height = image_size
        points = []
        
        # Reconstruct koordinat
        for i in range(0, len(normalized), 2):
            points.append((
                normalized[i] * width,
                normalized[i+1] * height
            ))
            
        return points
        
    def normalize_bbox(
        self,
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """
        Normalisasi koordinat bounding box (format YOLO)
        Args:
            bbox: (x_center, y_center, width, height)
            image_size: (width, height) gambar
        Returns:
            Tuple koordinat ternormalisasi
        """
        width, height = image_size
        x_center, y_center, box_width, box_height = bbox
        
        return (
            x_center / width,
            y_center / height,
            box_width / width,
            box_height / height
        )
        
    def process_label_file(
        self,
        label_path: Union[str, Path],
        image_size: Tuple[int, int],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Proses file label untuk normalisasi koordinat
        Args:
            label_path: Path ke file label
            image_size: (width, height) gambar
            save_path: Path untuk menyimpan hasil (optional)
        """
        label_path = Path(label_path)
        if save_path:
            save_path = Path(save_path)
            
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            normalized_lines = []
            for line in lines:
                parts = line.strip().split()
                class_id = parts[0]
                
                # Convert koordinat ke float
                coords = [float(x) for x in parts[1:]]
                
                # Normalisasi berdasarkan format
                if len(coords) == 4:  # bbox
                    normalized = self.normalize_bbox(
                        tuple(coords),
                        image_size
                    )
                else:  # polygon
                    points = list(zip(coords[::2], coords[1::2]))
                    normalized = self.normalize_polygon(points, image_size)
                    
                # Format output
                normalized_line = f"{class_id} {' '.join(map(str, normalized))}\n"
                normalized_lines.append(normalized_line)
                
            # Simpan hasil
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    f.writelines(normalized_lines)
                    
            return normalized_lines
            
        except Exception as e:
            self.logger.error(f"❌ Gagal memproses {label_path}: {str(e)}")
            raise e
            
    def process_dataset(
        self,
        label_dir: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Proses seluruh dataset untuk normalisasi koordinat
        Args:
            label_dir: Direktori file label
            image_dir: Direktori gambar
            output_dir: Direktori output (optional)
        """
        self.logger.start(f"🎯 Memulai normalisasi koordinat di {label_dir}")
        
        try:
            label_dir = Path(label_dir)
            image_dir = Path(image_dir)
            if output_dir:
                output_dir = Path(output_dir)
                
            # Collect file paths
            label_files = list(label_dir.glob("*.txt"))
            
            # Setup progress bar
            pbar = tqdm(
                total=len(label_files),
                desc="💫 Normalisasi koordinat"
            )
            
            def process_file(label_file: Path):
                try:
                    # Get corresponding image size
                    image_file = image_dir / label_file.with_suffix('.jpg').name
                    if not image_file.exists():
                        image_file = image_file.with_suffix('.png')
                        
                    img = cv2.imread(str(image_file))
                    image_size = (img.shape[1], img.shape[0])
                    
                    # Process label
                    if output_dir:
                        save_path = output_dir / label_file.name
                    else:
                        save_path = None
                        
                    self.process_label_file(label_file, image_size, save_path)
                    pbar.update(1)
                    
                except Exception as e:
                    self.logger.error(
                        f"❌ Gagal memproses {label_file.name}: {str(e)}"
                    )
                    
            # Process dengan multi-threading
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                executor.map(process_file, label_files)
                
            pbar.close()
            self.logger.success("✨ Normalisasi koordinat selesai!")
            
        except Exception as e:
            self.logger.error(f"❌ Normalisasi dataset gagal: {str(e)}")
            raise e