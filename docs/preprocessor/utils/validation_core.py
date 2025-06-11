"""
File: smartcash/dataset/preprocessor/utils/validation_core.py
Deskripsi: Konsolidasi core validation logic dengan unified validation interface
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set, Union
from dataclasses import dataclass

from smartcash.common.logger import get_logger

@dataclass
class ValidationResult:
    """📊 Unified validation result structure"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]
    
    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0

class ValidationCore:
    """🔍 Konsolidasi core validation logic dengan unified interface"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        # Image validation settings
        self.min_image_size = self.config.get('min_image_size', 32)
        self.max_image_size = self.config.get('max_image_size', 4096)
        self.max_file_size_mb = self.config.get('max_file_size_mb', 10)
        self.allowed_image_formats = set(self.config.get('allowed_image_formats', ['.jpg', '.jpeg', '.png', '.bmp']))
        
        # Label validation settings
        self.min_objects = self.config.get('min_objects_per_image', 1)
        self.max_objects = self.config.get('max_objects_per_image', 100)
        self.allowed_classes = set(self.config.get('allowed_classes', [])) if self.config.get('allowed_classes') else None
        self.min_bbox_size = self.config.get('min_bbox_size', 2)  # pixels
        
        # Quality settings
        self.check_blur = self.config.get('check_image_quality', True)
        self.blur_threshold = self.config.get('blur_threshold', 100)
        self.aspect_ratio_limits = self.config.get('aspect_ratio_limits', (0.1, 10.0))
    
    # === IMAGE VALIDATION ===
    
    def validate_image(self, image_path: Union[str, Path], 
                      check_quality: bool = None) -> ValidationResult:
        """🖼️ Comprehensive image validation dengan quality checks"""
        errors, warnings, stats = [], [], {}
        path = Path(image_path)
        
        try:
            # Basic file checks
            if not path.exists():
                errors.append(f"File tidak ditemukan: {path.name}")
                return ValidationResult(False, errors, warnings, stats)
            
            # Extension check
            if path.suffix.lower() not in self.allowed_image_formats:
                errors.append(f"Format tidak didukung: {path.suffix}")
            
            # File size check
            file_size_mb = path.stat().st_size / (1024 * 1024)
            stats['file_size_mb'] = round(file_size_mb, 2)
            
            if file_size_mb > self.max_file_size_mb:
                errors.append(f"File terlalu besar: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)")
            
            # Load and validate image content
            image = cv2.imread(str(path))
            if image is None:
                errors.append("File gambar corrupt atau tidak dapat dibaca")
                stats['is_corrupt'] = True
                return ValidationResult(False, errors, warnings, stats)
            
            # Dimension validation
            h, w = image.shape[:2]
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            
            stats.update({
                'width': w, 'height': h, 'channels': channels,
                'aspect_ratio': round(w / h, 2) if h > 0 else 0,
                'total_pixels': w * h
            })
            
            # Size validation
            if w < self.min_image_size or h < self.min_image_size:
                errors.append(f"Gambar terlalu kecil: {w}x{h} (min: {self.min_image_size}x{self.min_image_size})")
            
            if w > self.max_image_size or h > self.max_image_size:
                errors.append(f"Gambar terlalu besar: {w}x{h} (max: {self.max_image_size}x{self.max_image_size})")
            
            # Aspect ratio validation
            aspect_ratio = w / h if h > 0 else 0
            min_ratio, max_ratio = self.aspect_ratio_limits
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                warnings.append(f"Rasio aspek ekstrim: {aspect_ratio:.2f}")
            
            # Quality checks
            check_quality = check_quality if check_quality is not None else self.check_blur
            if check_quality and channels >= 3:
                blur_score = self._calculate_blur_score(image)
                stats['blur_score'] = round(blur_score, 2)
                
                if blur_score < self.blur_threshold:
                    warnings.append(f"Gambar blur terdeteksi (score: {blur_score:.1f})")
            
            return ValidationResult(len(errors) == 0, errors, warnings, stats)
            
        except Exception as e:
            errors.append(f"Error validasi: {str(e)}")
            return ValidationResult(False, errors, warnings, stats)
    
    def validate_label(self, label_path: Union[str, Path], 
                      image_size: Optional[Tuple[int, int]] = None) -> ValidationResult:
        """📋 Comprehensive YOLO label validation"""
        errors, warnings, stats = [], [], {}
        path = Path(label_path)
        
        try:
            # Basic file checks
            if not path.exists():
                errors.append(f"File label tidak ditemukan: {path.name}")
                return ValidationResult(False, errors, warnings, {'is_empty': True})
            
            if path.stat().st_size == 0:
                warnings.append("File label kosong")
                return ValidationResult(True, errors, warnings, {'is_empty': True, 'num_objects': 0})
            
            # Parse label content
            bboxes_data = self._parse_yolo_label(path)
            valid_objects = len([b for b in bboxes_data if b['valid']])
            
            stats.update({
                'num_objects': valid_objects,
                'total_lines': len(bboxes_data),
                'invalid_lines': len([b for b in bboxes_data if not b['valid']]),
                'class_distribution': self._calculate_class_distribution(bboxes_data),
                'bbox_sizes': [b.get('bbox_area', 0) for b in bboxes_data if b['valid']],
                'is_empty': valid_objects == 0
            })
            
            # Validation checks
            if valid_objects < self.min_objects:
                errors.append(f"Terlalu sedikit objek: {valid_objects} (min: {self.min_objects})")
            
            if valid_objects > self.max_objects:
                warnings.append(f"Banyak objek: {valid_objects} (max: {self.max_objects})")
            
            # Check class validation
            if self.allowed_classes:
                found_classes = set(stats['class_distribution'].keys())
                invalid_classes = found_classes - self.allowed_classes
                if invalid_classes:
                    errors.append(f"Kelas tidak valid: {list(invalid_classes)}")
            
            # Check bbox sizes jika image_size disediakan
            if image_size and valid_objects > 0:
                small_bboxes = self._check_bbox_sizes(bboxes_data, image_size)
                if small_bboxes:
                    warnings.append(f"{len(small_bboxes)} bbox terlalu kecil")
            
            return ValidationResult(len(errors) == 0, errors, warnings, stats)
            
        except Exception as e:
            errors.append(f"Error validasi label: {str(e)}")
            return ValidationResult(False, errors, warnings, stats)
    
    def validate_pair(self, image_path: Union[str, Path], 
                     label_path: Optional[Union[str, Path]] = None) -> ValidationResult:
        """🔗 Validate image-label pair consistency"""
        errors, warnings, stats = [], [], {}
        img_path = Path(image_path)
        
        try:
            # Auto-detect label path jika tidak disediakan
            if label_path is None:
                label_path = img_path.with_suffix('.txt')
                if not img_path.parent.name == 'images':
                    # Try labels directory
                    label_dir = img_path.parent.parent / 'labels'
                    if label_dir.exists():
                        label_path = label_dir / f"{img_path.stem}.txt"
            else:
                label_path = Path(label_path)
            
            # Validate individual files
            img_result = self.validate_image(img_path, check_quality=False)  # Quick validation
            label_result = self.validate_label(label_path, 
                                             (img_result.stats.get('width'), img_result.stats.get('height')))
            
            # Combine results
            errors.extend(img_result.errors)
            errors.extend(label_result.errors)
            warnings.extend(img_result.warnings)
            warnings.extend(label_result.warnings)
            
            # Consistency checks
            if img_path.stem != label_path.stem:
                errors.append(f"Nama file tidak konsisten: {img_path.stem} vs {label_path.stem}")
            
            # Combine stats
            stats.update({
                'image_stats': img_result.stats,
                'label_stats': label_result.stats,
                'is_consistent': img_path.stem == label_path.stem,
                'both_exist': img_path.exists() and label_path.exists()
            })
            
            return ValidationResult(len(errors) == 0, errors, warnings, stats)
            
        except Exception as e:
            errors.append(f"Error validasi pair: {str(e)}")
            return ValidationResult(False, errors, warnings, stats)
    
    # === BATCH VALIDATION ===
    
    def batch_validate_images(self, image_paths: List[Path], 
                             progress_callback: Optional[callable] = None) -> Dict[Path, ValidationResult]:
        """📦 Batch image validation dengan progress"""
        results = {}
        total = len(image_paths)
        
        for i, img_path in enumerate(image_paths):
            results[img_path] = self.validate_image(img_path)
            
            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback('current', i + 1, total, f"Validating {img_path.name}")
        
        return results
    
    def batch_validate_pairs(self, image_paths: List[Path], 
                           progress_callback: Optional[callable] = None) -> Dict[Path, ValidationResult]:
        """📦 Batch pair validation dengan progress"""
        results = {}
        total = len(image_paths)
        
        for i, img_path in enumerate(image_paths):
            results[img_path] = self.validate_pair(img_path)
            
            if progress_callback and i % max(1, total // 20) == 0:
                progress_callback('current', i + 1, total, f"Validating pair {img_path.name}")
        
        return results
    
    # === UTILITY METHODS ===
    
    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """📊 Calculate blur score menggunakan Laplacian variance"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            return cv2.Laplacian(gray, cv2.CV_64F).var()
        except Exception:
            return 0.0
    
    def _parse_yolo_label(self, label_path: Path) -> List[Dict[str, Any]]:
        """📋 Parse YOLO label file dengan validation"""
        bboxes_data = []
        
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    bbox_data = {'line_num': line_num, 'valid': False, 'errors': []}
                    
                    try:
                        parts = line.split()
                        if len(parts) < 5:
                            bbox_data['errors'].append(f"Format tidak lengkap (line {line_num})")
                            bboxes_data.append(bbox_data)
                            continue
                        
                        # Parse values
                        class_id = int(float(parts[0]))
                        x, y, w, h = map(float, parts[1:5])
                        
                        bbox_data.update({
                            'class_id': class_id,
                            'x_center': x,
                            'y_center': y,
                            'width': w,
                            'height': h,
                            'bbox_area': w * h
                        })
                        
                        # Validation
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            bbox_data['errors'].append(f"Koordinat tidak valid (line {line_num})")
                        else:
                            bbox_data['valid'] = True
                        
                    except (ValueError, IndexError) as e:
                        bbox_data['errors'].append(f"Parse error (line {line_num}): {str(e)}")
                    
                    bboxes_data.append(bbox_data)
        
        except Exception as e:
            self.logger.error(f"❌ Error parsing {label_path}: {str(e)}")
        
        return bboxes_data
    
    def _calculate_class_distribution(self, bboxes_data: List[Dict[str, Any]]) -> Dict[int, int]:
        """📊 Calculate class distribution"""
        distribution = {}
        for bbox in bboxes_data:
            if bbox['valid']:
                class_id = bbox['class_id']
                distribution[class_id] = distribution.get(class_id, 0) + 1
        return distribution
    
    def _check_bbox_sizes(self, bboxes_data: List[Dict[str, Any]], 
                         image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
        """📏 Check bbox sizes dalam pixels"""
        img_w, img_h = image_size
        small_bboxes = []
        
        for bbox in bboxes_data:
            if bbox['valid']:
                pixel_w = bbox['width'] * img_w
                pixel_h = bbox['height'] * img_h
                
                if pixel_w < self.min_bbox_size or pixel_h < self.min_bbox_size:
                    small_bboxes.append({
                        'bbox': bbox,
                        'pixel_size': (pixel_w, pixel_h)
                    })
        
        return small_bboxes

# === FACTORY FUNCTIONS ===

def create_validation_core(config: Dict[str, Any] = None) -> ValidationCore:
    """🏭 Factory untuk create ValidationCore"""
    return ValidationCore(config)

# === CONVENIENCE FUNCTIONS ===

def validate_image_safe(image_path: Union[str, Path]) -> bool:
    """🖼️ One-liner safe image validation"""
    try:
        validator = create_validation_core()
        result = validator.validate_image(image_path)
        return result.is_valid
    except Exception:
        return False

def validate_label_safe(label_path: Union[str, Path], image_size: Optional[Tuple[int, int]] = None) -> bool:
    """📋 One-liner safe label validation"""
    try:
        validator = create_validation_core()
        result = validator.validate_label(label_path, image_size)
        return result.is_valid
    except Exception:
        return False

def validate_pair_safe(image_path: Union[str, Path], label_path: Optional[Union[str, Path]] = None) -> bool:
    """🔗 One-liner safe pair validation"""
    try:
        validator = create_validation_core()
        result = validator.validate_pair(image_path, label_path)
        return result.is_valid
    except Exception:
        return False