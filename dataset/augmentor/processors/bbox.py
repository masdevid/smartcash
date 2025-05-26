"""
File: smartcash/dataset/augmentor/processors/bbox.py
Deskripsi: Fixed BBox processing dengan one-liner validation untuk handle float class_id errors
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from smartcash.common.logger import get_logger

# Fixed one-liner helpers dengan int conversion
parse_yolo_safe = lambda line: [int(float(parts[0]))] + [float(x) for x in parts[1:]] if (parts := line.strip().split()) and len(parts) >= 5 else []
format_yolo_safe = lambda bbox: f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}"
validate_bbox_safe = lambda bbox: len(bbox) >= 5 and isinstance(bbox[0], (int, float)) and all(0 <= x <= 1 for x in bbox[1:5])

class BBoxProcessor:
    """Fixed BBox processor dengan proper class_id handling"""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
        self.processed_count = 0
        self.error_count = 0
    
    def read_yolo_labels(self, label_path: str) -> List[List[float]]:
        """Fixed YOLO label reader dengan proper class_id conversion"""
        try:
            if not Path(label_path).exists():
                return []
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Parse dengan safe conversion
            bboxes = [parse_yolo_safe(line) for line in lines]
            valid_bboxes = [bbox for bbox in bboxes if bbox and validate_bbox_safe(bbox)]
            
            # Silent logging untuk prevent flooding
            error_count = len(bboxes) - len(valid_bboxes)
            error_count > 0 and self.logger.debug(f"⚠️ {error_count} invalid bboxes di {Path(label_path).name}")
            
            return valid_bboxes
            
        except Exception as e:
            self.logger.debug(f"❌ Error reading label {Path(label_path).name}: {str(e)}")
            self.error_count += 1
            return []
    
    def save_yolo_labels(self, bboxes: List[List[float]], output_path: str) -> bool:
        """Fixed YOLO label saver dengan validation"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            valid_bboxes = [bbox for bbox in bboxes if validate_bbox_safe(bbox)]
            
            with open(output_path, 'w') as f:
                [f.write(format_yolo_safe(bbox) + '\n') for bbox in valid_bboxes]
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            self.logger.debug(f"❌ Error saving label {Path(output_path).name}: {str(e)}")
            self.error_count += 1
            return False
    
    def get_processing_stats(self) -> Dict[str, int]:
        return {
            'processed_labels': self.processed_count,
            'error_count': self.error_count,
            'success_rate': (self.processed_count / max(1, self.processed_count + self.error_count)) * 100
        }
    
    def reset_stats(self) -> None:
        self.processed_count = self.error_count = 0