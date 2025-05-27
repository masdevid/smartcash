"""
File: smartcash/dataset/augmentor/utils/bbox_operations.py
Deskripsi: SRP module untuk operasi YOLO bounding box dengan validation
"""

import os
from pathlib import Path
from typing import List

# Direct import untuk avoid circular dependency
def resolve_drive_path(path: str) -> str:
    """Inline resolve_drive_path untuk avoid circular import"""
    if os.path.isabs(path):
        return path
    for base in ['/content/drive/MyDrive/SmartCash', '/content/drive/MyDrive', '/content/SmartCash', '/content', os.getcwd()]:
        full_path = os.path.join(base, path)
        if os.path.exists(full_path):
            return full_path
    return path

def get_parent(path: str) -> str:
    """Inline get_parent untuk avoid circular import"""
    return str(Path(path).parent)

def _safe_execute(operation, fallback_result=None):
    """Inline safe_execute untuk avoid circular import"""
    try:
        return operation()
    except Exception:
        return fallback_result

# =============================================================================
# BBOX PARSING - One-liner utilities
# =============================================================================

parse_yolo_line = lambda line: [int(float(parts[0]))] + [float(x) for x in parts[1:]] if (parts := line.strip().split()) and len(parts) >= 5 else []
format_yolo_line = lambda bbox: f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}"
validate_bbox = lambda bbox: len(bbox) >= 5 and isinstance(bbox[0], (int, float)) and all(0 <= x <= 1 for x in bbox[1:5])

# =============================================================================
# BBOX FILE OPERATIONS
# =============================================================================

def read_yolo_labels(label_path: str) -> List[List[float]]:
    """Read YOLO labels dengan validation"""
    def _read_labels():
        with open(resolve_drive_path(label_path), 'r') as f:
            return [bbox for line in f if (bbox := parse_yolo_line(line)) and validate_bbox(bbox)]
    
    return _safe_execute(_read_labels, [])

def save_yolo_labels(bboxes: List[List[float]], output_path: str) -> bool:
    """Save YOLO labels dengan validation"""
    def _save_labels():
        from smartcash.dataset.augmentor.utils.path_operations import ensure_dirs
        ensure_dirs(get_parent(output_path))
        
        with open(resolve_drive_path(output_path), 'w') as f:
            [f.write(format_yolo_line(bbox) + '\n') for bbox in bboxes if validate_bbox(bbox)]
        return True
    
    return _safe_execute(_save_labels, False)

def load_yolo_labels(label_path: str) -> tuple:
    """Load YOLO labels untuk augmentation - return (bboxes, class_labels)"""
    if not label_path:
        return [], []
    
    def _load_labels():
        bboxes, class_labels = [], []
        with open(resolve_drive_path(label_path), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = int(float(parts[0]))
                        bbox = [float(x) for x in parts[1:5]]
                        if all(0 <= x <= 1 for x in bbox) and bbox[2] > 0.001 and bbox[3] > 0.001:
                            bboxes.append(bbox)
                            class_labels.append(class_id)
                    except (ValueError, IndexError):
                        continue
        return bboxes, class_labels
    
    return _safe_execute(_load_labels, ([], []))

def save_validated_labels(bboxes: List, class_labels: List, output_path: str) -> bool:
    """Save labels dengan validation untuk augmentation output"""
    def _save_validated():
        from smartcash.dataset.augmentor.utils.path_operations import ensure_dirs
        ensure_dirs(get_parent(output_path))
        
        with open(resolve_drive_path(output_path), 'w') as f:
            for bbox, class_label in zip(bboxes, class_labels):
                if len(bbox) >= 4:
                    coords = [max(0.0, min(1.0, float(x))) for x in bbox[:4]]
                    if coords[2] > 0.001 and coords[3] > 0.001:
                        f.write(f"{int(class_label)} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
        return True
    
    return _safe_execute(_save_validated, False)