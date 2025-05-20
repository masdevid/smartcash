"""
File: smartcash/detection/services/postprocessing/result_formatter.py
Description: Formatter untuk hasil deteksi.
"""

import json
import csv
import io
from typing import Dict, List, Optional, Union, Any

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection


class ResultFormatter:
    """Formatter untuk hasil deteksi"""
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Result Formatter
        
        Args:
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.logger = logger or get_logger()
    
    def to_json(self, 
               detections: List[Detection], 
               include_metadata: bool = True,
               pretty: bool = True) -> str:
        """
        Format deteksi ke JSON
        
        Args:
            detections: List hasil deteksi
            include_metadata: Flag untuk menyertakan metadata tambahan
            pretty: Flag untuk format JSON yang rapi
            
        Returns:
            String JSON hasil deteksi
        """
        result = []
        
        for detection in detections:
            item = {
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": float(detection.confidence),
                "bbox": [float(coord) for coord in detection.bbox]
            }
            
            # Tambahkan metadata lain jika ada dan diminta
            if include_metadata and hasattr(detection, 'metadata') and detection.metadata:
                item["metadata"] = detection.metadata
                
            result.append(item)
            
        # Tambahkan metadata umum jika diperlukan
        if include_metadata:
            metadata = {
                "count": len(detections),
                "format": "JSON"
            }
            
            wrapped_result = {
                "metadata": metadata,
                "detections": result
            }
        else:
            wrapped_result = result
            
        # Format JSON
        indent = 2 if pretty else None
        return json.dumps(wrapped_result, indent=indent)
    
    def to_csv(self, 
              detections: List[Detection],
              include_header: bool = True) -> str:
        """
        Format deteksi ke CSV
        
        Args:
            detections: List hasil deteksi
            include_header: Flag untuk menyertakan header
            
        Returns:
            String CSV hasil deteksi
        """
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Tulis header jika diminta
        if include_header:
            writer.writerow(['class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        
        # Tulis data deteksi
        for detection in detections:
            writer.writerow([
                detection.class_id,
                detection.class_name,
                f"{detection.confidence:.6f}",
                f"{detection.bbox[0]:.6f}",
                f"{detection.bbox[1]:.6f}",
                f"{detection.bbox[2]:.6f}",
                f"{detection.bbox[3]:.6f}"
            ])
        
        return output.getvalue()
    
    def to_yolo_format(self, 
                      detections: List[Detection]) -> str:
        """
        Format deteksi ke format YOLO
        
        Args:
            detections: List hasil deteksi
            
        Returns:
            String format YOLO hasil deteksi
        """
        lines = []
        
        for detection in detections:
            # Format YOLO: class_id x_center y_center width height
            x1, y1, x2, y2 = detection.bbox
            
            # Konversi koordinat ke format YOLO
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            line = f"{detection.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            lines.append(line)
        
        return "\n".join(lines)
    
    def to_coco_format(self, 
                      detections: List[Detection],
                      image_id: int = 1,
                      image_width: int = 1,
                      image_height: int = 1) -> Dict:
        """
        Format deteksi ke format COCO
        
        Args:
            detections: List hasil deteksi
            image_id: ID gambar untuk format COCO
            image_width: Lebar gambar asli
            image_height: Tinggi gambar asli
            
        Returns:
            Dictionary format COCO hasil deteksi
        """
        annotations = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection.bbox
            
            # Konversi koordinat relatif ke absolut
            x1_abs = x1 * image_width
            y1_abs = y1 * image_height
            x2_abs = x2 * image_width
            y2_abs = y2 * image_height
            
            # Hitung width dan height
            width = x2_abs - x1_abs
            height = y2_abs - y1_abs
            
            # Format COCO annotation
            annotation = {
                "id": i + 1,
                "image_id": image_id,
                "category_id": detection.class_id,
                "bbox": [float(x1_abs), float(y1_abs), float(width), float(height)],
                "area": float(width * height),
                "segmentation": [],
                "iscrowd": 0,
                "score": float(detection.confidence)
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def format_detections(self, 
                         detections: List[Detection],
                         format: str = 'json',
                         **kwargs) -> Union[str, Dict]:
        """
        Format deteksi ke format yang ditentukan
        
        Args:
            detections: List hasil deteksi
            format: Format output ('json', 'csv', 'yolo', 'coco')
            **kwargs: Parameter tambahan untuk formatter spesifik
            
        Returns:
            Hasil deteksi dalam format yang ditentukan
        """
        if format.lower() == 'json':
            return self.to_json(
                detections=detections,
                include_metadata=kwargs.get('include_metadata', True),
                pretty=kwargs.get('pretty', True)
            )
            
        elif format.lower() == 'csv':
            return self.to_csv(
                detections=detections,
                include_header=kwargs.get('include_header', True)
            )
            
        elif format.lower() == 'yolo':
            return self.to_yolo_format(
                detections=detections
            )
            
        elif format.lower() == 'coco':
            return self.to_coco_format(
                detections=detections,
                image_id=kwargs.get('image_id', 1),
                image_width=kwargs.get('image_width', 1),
                image_height=kwargs.get('image_height', 1)
            )
            
        else:
            self.logger.warning(f"⚠️ Format tidak didukung: {format}, menggunakan JSON")
            return self.to_json(detections)