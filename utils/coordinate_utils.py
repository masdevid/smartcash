"""
File: smartcash/utils/coordinate_utils.py
Author: Alfrida Sabar
Deskripsi: Utilitas terintegrasi untuk operasi koordinat, bounding box, dan polygon.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import threading

try:
    from shapely.geometry import Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY, Polygon = False, None

from smartcash.utils.logger import get_logger
from smartcash.utils.dataset.dataset_utils import IMG_EXTENSIONS

class CoordinateUtils:
    def __init__(self, logger=None, num_workers: int = 4):
        self.logger = logger or get_logger("coord_utils")
        self.num_workers = num_workers
        self._lock = threading.Lock()
    
    def normalize_bbox(self, bbox, image_size, format='xyxy'):
        """Normalisasi koordinat bounding box."""
        width, height = image_size
        transforms = {
            'xyxy': lambda b: [max(0, min(1, x/width)) for x in b],
            'xywh': lambda b: [max(0, min(1, x/width)) for x in b],
            'yolo': lambda b: [max(0, min(1, x/width)) for x in b]
        }
        return transforms.get(format, lambda x: x)(bbox)
    
    def convert_bbox_format(self, bbox, from_format, to_format):
        """Konversi antar format bounding box."""
        if from_format == to_format:
            return bbox
        
        def to_xyxy(b, fmt):
            if fmt == 'xyxy': return b
            elif fmt == 'xywh': return [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            elif fmt == 'yolo': 
                x, y, w, h = b
                return [x - w/2, y - h/2, x + w/2, y + h/2]
        
        def from_xyxy(b, fmt):
            if fmt == 'xyxy': return b
            elif fmt == 'xywh': return [b[0], b[1], b[2] - b[0], b[3] - b[1]]
            elif fmt == 'yolo': 
                x1, y1, x2, y2 = b
                return [(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1]
        
        return from_xyxy(to_xyxy(bbox, from_format), to_format)
    
    def calculate_iou(self, box1, box2, format='xyxy'):
        """Hitung Intersection over Union."""
        def prepare_boxes(b1, b2, fmt):
            b1 = self.convert_bbox_format(b1, fmt, 'xyxy')
            b2 = self.convert_bbox_format(b2, fmt, 'xyxy')
            return b1, b2
        
        box1, box2 = prepare_boxes(box1, box2, format)
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
                     (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def calculate_poly_iou(self, poly1, poly2):
        """Hitung IoU antar polygon."""
        if not HAS_SHAPELY:
            return self._fallback_poly_iou(poly1, poly2)
        
        try:
            p1, p2 = Polygon(poly1), Polygon(poly2)
            intersection = p1.intersection(p2).area
            union = p1.union(p2).area
            return intersection / union if union > 0 else 0.0
        except Exception:
            return self._fallback_poly_iou(poly1, poly2)
    
    def _fallback_poly_iou(self, poly1, poly2):
        """Fallback sederhana IoU polygon."""
        def bbox(poly):
            return [min(p[0] for p in poly), min(p[1] for p in poly),
                    max(p[0] for p in poly), max(p[1] for p in poly)]
        
        return self.calculate_iou(bbox(poly1), bbox(poly2))
    
    def process_label_file(self, label_path, image_size, save_path=None, normalize=True):
        """Proses file label untuk normalisasi/denormalisasi."""
        label_path = Path(label_path)
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            processed_lines = []
            processed_labels = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:5]]
                
                # Normalisasi atau denormalisasi
                if normalize and all(0 <= c <= 1 for c in coords):
                    processed_coords = coords
                elif not normalize:
                    x_center = coords[0] * image_size[0]
                    y_center = coords[1] * image_size[1]
                    width = coords[2] * image_size[0]
                    height = coords[3] * image_size[1]
                    processed_coords = [x_center, y_center, width, height]
                else:
                    processed_coords = coords
                
                processed_line = f"{class_id} {' '.join(map(str, processed_coords))}\n"
                processed_lines.append(processed_line)
                
                processed_labels.append({
                    'class_id': class_id,
                    'coordinates': processed_coords
                })
            
            # Simpan hasil jika ada save_path
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w') as f:
                    f.writelines(processed_lines)
            
            return processed_labels
        
        except Exception as e:
            self.logger.error(f"❌ Gagal memproses label {label_path}: {str(e)}")
            return []
    
    def normalize_polygon(self, points, image_size):
        """Normalisasi koordinat polygon."""
        width, height = image_size
        normalized = []
        
        for x, y in points:
            normalized.extend([
                max(0.0, min(1.0, x / width)),
                max(0.0, min(1.0, y / height))
            ])
        
        return normalized
    
    def box_to_polygon(self, box, format='xyxy'):
        """Konversi bounding box ke polygon."""
        # Konversi ke format xyxy jika belum
        if format != 'xyxy':
            box = self.convert_bbox_format(box, format, 'xyxy')
        
        x1, y1, x2, y2 = box
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    
    def process_dataset(self, label_dir, image_dir, output_dir=None, normalize=True):
        """Proses dataset untuk normalisasi/denormalisasi koordinat."""
        label_files = list(Path(label_dir).glob("*.txt"))
        if not label_files:
            self.logger.warning("⚠️ Tidak ada file label ditemukan")
            return {'processed': 0, 'failed': 0, 'skipped': 0}
        
        def process_file(label_file):
            try:
                image_file = next(
                    (Path(image_dir) / (label_file.stem + ext) 
                     for ext in IMG_EXTENSIONS 
                     if (Path(image_dir) / (label_file.stem + ext)).exists()), 
                    None
                )
                
                if not image_file or cv2.imread(str(image_file)) is None:
                    return {'skipped': 1}
                
                img = cv2.imread(str(image_file))
                image_size = (img.shape[1], img.shape[0])
                save_path = Path(output_dir) / label_file.name if output_dir else None
                
                self.process_label_file(label_file, image_size, save_path, normalize)
                return {'processed': 1}
            
            except Exception as e:
                self.logger.error(f"❌ Gagal memproses {label_file.name}: {str(e)}")
                return {'failed': 1}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(tqdm(
                executor.map(process_file, label_files), 
                total=len(label_files), 
                desc="💫 Memproses koordinat"
            ))
        
        stats = {'processed': 0, 'failed': 0, 'skipped': 0}
        for result in results:
            for key, value in result.items():
                stats[key] += value
        
        return stats

def calculate_iou(box1, box2, format='xyxy'):
    """Fungsi helper global untuk menghitung IoU."""
    return CoordinateUtils().calculate_iou(box1, box2, format)