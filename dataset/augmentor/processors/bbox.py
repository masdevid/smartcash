"""
File: smartcash/dataset/augmentor/processors/bbox.py
Deskripsi: BBox processing operations dengan one-liner optimized untuk YOLO format transformations
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from smartcash.common.logger import get_logger

# One-liner helper functions untuk YOLO format
parse_yolo = lambda line: [float(x) for x in line.strip().split()] if line.strip() else []
format_yolo = lambda bbox: f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}"
validate_bbox = lambda bbox: len(bbox) >= 5 and all(0 <= x <= 1 for x in bbox[1:5])
scale_bbox = lambda bbox, w_ratio, h_ratio: [bbox[0], bbox[1]*w_ratio, bbox[2]*h_ratio, bbox[3]*w_ratio, bbox[4]*h_ratio] if len(bbox) >= 5 else bbox
bbox_to_xyxy = lambda bbox, w, h: (int((bbox[1] - bbox[3]/2) * w), int((bbox[2] - bbox[4]/2) * h), int((bbox[1] + bbox[3]/2) * w), int((bbox[2] + bbox[4]/2) * h))
xyxy_to_bbox = lambda x1, y1, x2, y2, w, h, cls: [cls, (x1+x2)/(2*w), (y1+y2)/(2*h), (x2-x1)/w, (y2-y1)/h]

class BBoxProcessor:
    """Processor untuk transformasi bounding box dengan optimized YOLO operations."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi BBoxProcessor.
        
        Args:
            logger: Logger untuk logging operations
        """
        self.logger = logger or get_logger(__name__)
        self.processed_count = 0
        self.error_count = 0
    
    def read_yolo_labels(self, label_path: str) -> List[List[float]]:
        """
        Baca file label YOLO format.
        
        Args:
            label_path: Path ke file label
            
        Returns:
            List bounding boxes dalam format YOLO
        """
        try:
            if not Path(label_path).exists():
                return []
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Parse semua line dengan filter validation
            bboxes = [parse_yolo(line) for line in lines]
            valid_bboxes = [bbox for bbox in bboxes if validate_bbox(bbox)]
            
            if len(valid_bboxes) != len(bboxes):
                self.logger.warning(f"⚠️ {len(bboxes) - len(valid_bboxes)} invalid bboxes dalam {label_path}")
            
            return valid_bboxes
            
        except Exception as e:
            self.logger.error(f"❌ Error membaca label {label_path}: {str(e)}")
            self.error_count += 1
            return []
    
    def save_yolo_labels(self, bboxes: List[List[float]], output_path: str) -> bool:
        """
        Simpan bounding boxes ke file YOLO format.
        
        Args:
            bboxes: List bounding boxes
            output_path: Path output file
            
        Returns:
            Boolean keberhasilan
        """
        try:
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Format dan save dengan validation
            valid_bboxes = [bbox for bbox in bboxes if validate_bbox(bbox)]
            
            with open(output_path, 'w') as f:
                for bbox in valid_bboxes:
                    f.write(format_yolo(bbox) + '\n')
            
            self.processed_count += 1
            if len(valid_bboxes) != len(bboxes):
                self.logger.warning(f"⚠️ {len(bboxes) - len(valid_bboxes)} invalid bboxes dilewati")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error menyimpan label {output_path}: {str(e)}")
            self.error_count += 1
            return False
    
    def transform_bboxes_by_ratio(
        self, 
        bboxes: List[List[float]], 
        width_ratio: float, 
        height_ratio: float
    ) -> List[List[float]]:
        """
        Transform bounding boxes berdasarkan rasio perubahan dimensi.
        
        Args:
            bboxes: List bounding boxes original
            width_ratio: Rasio perubahan lebar
            height_ratio: Rasio perubahan tinggi
            
        Returns:
            List bounding boxes yang sudah ditransform
        """
        try:
            # Transform dengan validation per bbox
            transformed = []
            for bbox in bboxes:
                if not validate_bbox(bbox):
                    continue
                
                # Scale coordinates dengan clipping
                scaled = scale_bbox(bbox, width_ratio, height_ratio)
                
                # Clip coordinates ke range [0, 1]
                clipped = [
                    scaled[0],  # class_id tidak berubah
                    max(0, min(1, scaled[1])),  # x_center
                    max(0, min(1, scaled[2])),  # y_center
                    max(0, min(1, scaled[3])),  # width
                    max(0, min(1, scaled[4]))   # height
                ]
                
                # Validasi hasil final
                if validate_bbox(clipped) and clipped[3] > 0 and clipped[4] > 0:
                    transformed.append(clipped)
            
            return transformed
            
        except Exception as e:
            self.logger.error(f"❌ Error transform bboxes: {str(e)}")
            return []
    
    def filter_valid_bboxes(
        self, 
        bboxes: List[List[float]], 
        min_area: float = 0.0001
    ) -> List[List[float]]:
        """
        Filter bounding boxes berdasarkan validitas dan area minimum.
        
        Args:
            bboxes: List bounding boxes
            min_area: Area minimum (width * height)
            
        Returns:
            List bounding boxes yang valid
        """
        valid_bboxes = []
        
        for bbox in bboxes:
            if not validate_bbox(bbox):
                continue
            
            # Hitung area
            area = bbox[3] * bbox[4]  # width * height
            
            if area >= min_area:
                valid_bboxes.append(bbox)
        
        return valid_bboxes
    
    def convert_to_absolute(
        self, 
        bboxes: List[List[float]], 
        image_width: int, 
        image_height: int
    ) -> List[Tuple[int, int, int, int, int]]:
        """
        Konversi YOLO format ke absolute coordinates (x1, y1, x2, y2, class).
        
        Args:
            bboxes: List bounding boxes YOLO format
            image_width: Lebar gambar
            image_height: Tinggi gambar
            
        Returns:
            List absolute coordinates
        """
        absolute_bboxes = []
        
        for bbox in bboxes:
            if not validate_bbox(bbox):
                continue
            
            try:
                x1, y1, x2, y2 = bbox_to_xyxy(bbox, image_width, image_height)
                
                # Clipping ke dimensi gambar
                x1 = max(0, min(image_width - 1, x1))
                y1 = max(0, min(image_height - 1, y1))
                x2 = max(x1 + 1, min(image_width, x2))
                y2 = max(y1 + 1, min(image_height, y2))
                
                absolute_bboxes.append((x1, y1, x2, y2, int(bbox[0])))
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error konversi bbox: {str(e)}")
                continue
        
        return absolute_bboxes
    
    def convert_from_absolute(
        self, 
        absolute_bboxes: List[Tuple[int, int, int, int, int]], 
        image_width: int, 
        image_height: int
    ) -> List[List[float]]:
        """
        Konversi absolute coordinates ke YOLO format.
        
        Args:
            absolute_bboxes: List absolute coordinates (x1, y1, x2, y2, class)
            image_width: Lebar gambar
            image_height: Tinggi gambar
            
        Returns:
            List bounding boxes YOLO format
        """
        yolo_bboxes = []
        
        for x1, y1, x2, y2, cls in absolute_bboxes:
            try:
                # Validasi coordinates
                if x2 <= x1 or y2 <= y1:
                    continue
                
                bbox = xyxy_to_bbox(x1, y1, x2, y2, image_width, image_height, cls)
                
                if validate_bbox(bbox):
                    yolo_bboxes.append(bbox)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Error konversi absolute: {str(e)}")
                continue
        
        return yolo_bboxes
    
    def get_bbox_statistics(self, bboxes: List[List[float]]) -> Dict[str, Any]:
        """
        Dapatkan statistik dari bounding boxes.
        
        Args:
            bboxes: List bounding boxes
            
        Returns:
            Dictionary statistik
        """
        if not bboxes:
            return {'count': 0, 'classes': [], 'areas': []}
        
        # Hitung statistik dengan one-liner optimizations
        valid_bboxes = [bbox for bbox in bboxes if validate_bbox(bbox)]
        classes = [int(bbox[0]) for bbox in valid_bboxes]
        areas = [bbox[3] * bbox[4] for bbox in valid_bboxes]
        
        return {
            'count': len(valid_bboxes),
            'total_input': len(bboxes),
            'classes': sorted(set(classes)),
            'class_counts': {cls: classes.count(cls) for cls in set(classes)},
            'areas': areas,
            'avg_area': sum(areas) / len(areas) if areas else 0,
            'min_area': min(areas) if areas else 0,
            'max_area': max(areas) if areas else 0
        }
    
    def get_processing_stats(self) -> Dict[str, int]:
        """
        Dapatkan statistik processing.
        
        Returns:
            Dictionary statistik
        """
        return {
            'processed_labels': self.processed_count,
            'error_count': self.error_count,
            'success_rate': (self.processed_count / (self.processed_count + self.error_count)) * 100 if (self.processed_count + self.error_count) > 0 else 0
        }
    
    def reset_stats(self) -> None:
        """Reset counter statistik."""
        self.processed_count = 0
        self.error_count = 0
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BBoxProcessor(processed={self.processed_count}, errors={self.error_count})"