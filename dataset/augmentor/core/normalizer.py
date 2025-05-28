"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: One-liner normalizer dengan preprocessing logic konsisten dan default 640x640
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import time

from smartcash.dataset.augmentor.utils.file_operations import find_augmented_files_split_aware, copy_file_with_uuid_preservation
from smartcash.dataset.augmentor.utils.path_operations import ensure_split_dirs, resolve_drive_path
from smartcash.dataset.augmentor.utils.batch_processor import process_batch_split_aware
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker
from smartcash.dataset.augmentor.utils.bbox_operations import save_validated_labels

# One-liner constants
DEFAULT_IMG_SIZE = 640
get_preprocessing_config = lambda config: config.get('preprocessing', {})
get_normalization_method = lambda config: get_preprocessing_config(config).get('normalization_method', 'minmax')
get_img_size = lambda config: get_preprocessing_config(config).get('img_size', DEFAULT_IMG_SIZE)
get_preserve_aspect = lambda config: get_preprocessing_config(config).get('preserve_aspect_ratio', True)
get_normalize_flag = lambda config: get_preprocessing_config(config).get('normalize', True)

# One-liner image processing
normalize_to_float = lambda img: img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)
float_to_uint8 = lambda img: (img * 255.0).astype(np.uint8)
calc_resize_scale = lambda w, h, tw, th: min(tw / w, th / h)
create_canvas = lambda th, tw, channels: np.zeros((th, tw, channels), dtype=np.uint8) if channels > 1 else np.zeros((th, tw), dtype=np.uint8)

class NormalizationEngine:
    """One-liner normalizer dengan preprocessing logic konsisten"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config, self.progress, self.comm, self.stats = config, create_progress_tracker(communicator), communicator, defaultdict(int)
        
    def normalize_augmented_data(self, augmented_dir: str, preprocessed_dir: str, target_split: str = "train") -> Dict[str, Any]:
        """One-liner normalization dengan path handling"""
        start_time = time.time()
        self.progress.log_info(f"🔄 Normalisasi {target_split}: {augmented_dir} → {preprocessed_dir}")
        
        try:
            ensure_split_dirs(preprocessed_dir, target_split)
            aug_files = self._find_augmented_files(augmented_dir, target_split)
            
            if not aug_files:
                self.progress.log_warning(f"⚠️ Tidak ada file augmented: {augmented_dir}")
                return self._empty_result(preprocessed_dir, target_split)
            
            self.progress.log_info(f"📊 Processing {len(aug_files)} files")
            
            norm_results = process_batch_split_aware(
                aug_files, 
                lambda fp: self._normalize_file(fp, preprocessed_dir, target_split),
                progress_tracker=self.progress,
                operation_name="normalization", 
                split_context=target_split
            )
            
            return self._success_result(norm_results, time.time() - start_time, preprocessed_dir, target_split)
            
        except Exception as e:
            return self._error_result(f"Normalization error: {str(e)}")
    
    def _find_augmented_files(self, augmented_dir: str, target_split: str) -> List[str]:
        """One-liner file finder dengan path patterns"""
        resolved_dir = resolve_drive_path(augmented_dir)
        patterns = [f"{resolved_dir}/{target_split}/images", f"{resolved_dir}/{target_split}", f"{resolved_dir}/images", resolved_dir]
        
        for pattern_dir in patterns:
            if Path(pattern_dir).exists():
                files = [str(f) for f in Path(pattern_dir).glob('aug_*.jpg')]
                if files:
                    self.progress.log_info(f"📂 Found {len(files)} files in {pattern_dir}")
                    return files
        return []
    
    def _normalize_file(self, file_path: str, preprocessed_dir: str, target_split: str) -> Dict[str, Any]:
        """One-liner file normalization"""
        try:
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Cannot read image'}
            
            # One-liner preprocessing
            normalized_image = self._apply_preprocessing(image)
            file_stem = Path(file_path).stem
            
            # One-liner paths
            target_img_path = Path(preprocessed_dir) / target_split / 'images' / f"{file_stem}.jpg"
            target_label_path = Path(preprocessed_dir) / target_split / 'labels' / f"{file_stem}.txt"
            
            # One-liner save
            target_img_path.parent.mkdir(parents=True, exist_ok=True)
            target_label_path.parent.mkdir(parents=True, exist_ok=True)
            
            img_saved = cv2.imwrite(str(target_img_path), normalized_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            label_saved = self._copy_label(self._find_label(file_path), str(target_label_path))
            
            return {'status': 'success', 'file': file_path, 'img_saved': img_saved, 'label_saved': label_saved}
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """One-liner preprocessing pipeline dengan 640x640 default"""
        # One-liner resize
        resized = self._resize_image(image)
        # One-liner normalization
        return self._normalize_image(resized) if get_normalize_flag(self.config) else resized
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """One-liner resize dengan aspect ratio"""
        img_size = get_img_size(self.config)
        target_size = (img_size, img_size) if isinstance(img_size, int) else tuple(img_size[:2])
        
        return self._resize_with_padding(image, target_size) if get_preserve_aspect(self.config) else cv2.resize(image, target_size)
    
    def _resize_with_padding(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """One-liner resize dengan padding"""
        h, w = image.shape[:2]
        tw, th = target_size
        scale = calc_resize_scale(w, h, tw, th)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # One-liner resize dan padding
        resized = cv2.resize(image, (new_w, new_h))
        canvas = create_canvas(th, tw, len(image.shape))
        x_offset, y_offset = (tw - new_w) // 2, (th - new_h) // 2
        
        if len(image.shape) == 3:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = resized
        else:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
        return canvas
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """One-liner normalization dengan method selection"""
        method = get_normalization_method(self.config)
        
        if method == 'minmax':
            return float_to_uint8(normalize_to_float(image))
        elif method == 'standard':
            normalized = normalize_to_float(image)
            mean, std = normalized.mean(), normalized.std()
            if std > 0:
                standardized = (normalized - mean) / std
                return ((standardized - standardized.min()) / (standardized.max() - standardized.min()) * 255.0).astype(np.uint8)
        
        return image
    
    def _find_label(self, image_path: str) -> str:
        """One-liner label finder"""
        img_path = Path(image_path)
        label_paths = [img_path.parent.parent / 'labels' / f"{img_path.stem}.txt", img_path.parent / 'labels' / f"{img_path.stem}.txt", img_path.parent / f"{img_path.stem}.txt"]
        return next((str(lp) for lp in label_paths if lp.exists()), "")
    
    def _copy_label(self, source_label: str, target_label: str) -> bool:
        """One-liner label copy dengan validation"""
        if not source_label or not Path(source_label).exists():
            return False
        
        try:
            valid_lines = []
            with open(source_label, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            coords = [float(x) for x in parts[1:5]]
                            if all(0.0 <= x <= 1.0 for x in coords) and coords[2] > 0.001 and coords[3] > 0.001:
                                valid_lines.append(f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                        except (ValueError, IndexError):
                            continue
            
            Path(target_label).parent.mkdir(parents=True, exist_ok=True)
            with open(target_label, 'w') as f:
                f.writelines(valid_lines)
            return len(valid_lines) > 0
            
        except Exception:
            return copy_file_with_uuid_preservation(source_label, target_label) if source_label else False
    
    def _success_result(self, results: List[Dict], processing_time: float, preprocessed_dir: str, target_split: str) -> Dict[str, Any]:
        """One-liner success result"""
        successful = [r for r in results if r.get('status') == 'success']
        img_count, label_count = sum(1 for r in successful if r.get('img_saved')), sum(1 for r in successful if r.get('label_saved'))
        
        self.progress.log_success(f"✅ Normalisasi: {len(successful)}/{len(results)} files")
        self.progress.log_info(f"📊 Images: {img_count}, Labels: {label_count} → {preprocessed_dir}/{target_split}")
        self._log_config()
        
        return {
            'status': 'success', 'total_files_processed': len(results), 'total_normalized': len(successful),
            'images_saved': img_count, 'labels_saved': label_count, 'processing_time': processing_time,
            'target_split': target_split, 'target_dir': f"{preprocessed_dir}/{target_split}",
            'normalization_speed': len(results) / processing_time if processing_time > 0 else 0
        }
    
    def _log_config(self):
        """One-liner config logging"""
        method, size = get_normalization_method(self.config), get_img_size(self.config)
        preserve = get_preserve_aspect(self.config)
        self.progress.log_success(f"🔧 Config: {method} normalization, {size}x{size} resize, aspect={preserve}")
    
    def _empty_result(self, preprocessed_dir: str, target_split: str) -> Dict[str, Any]:
        """One-liner empty result"""
        return {'status': 'success', 'total_files_processed': 0, 'total_normalized': 0, 'images_saved': 0, 'labels_saved': 0, 'processing_time': 0.0, 'target_split': target_split, 'target_dir': f"{preprocessed_dir}/{target_split}", 'normalization_speed': 0.0}
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """One-liner error result"""
        self.progress.log_error(message)
        return {'status': 'error', 'message': message, 'total_normalized': 0}

# One-liner utilities
create_normalization_engine = lambda config, communicator=None: NormalizationEngine(config, communicator)
normalize_split_data = lambda config, aug_dir, prep_dir, target_split='train': create_normalization_engine(config).normalize_augmented_data(aug_dir, prep_dir, target_split)
apply_preprocessing_normalization = lambda image, config: create_normalization_engine(config)._apply_preprocessing(image)
normalize_with_config = lambda image, method='minmax', size=640: create_normalization_engine({'preprocessing': {'normalization_method': method, 'img_size': size}})._apply_preprocessing(image)