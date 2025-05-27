"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: Updated normalizer menggunakan SRP utils modules dengan one-liner style
"""

import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import time

# Updated imports dari SRP utils modules - menggantikan direct operations
from smartcash.dataset.augmentor.utils.file_operations import find_augmented_files_split_aware, copy_file_with_uuid_preservation
from smartcash.dataset.augmentor.utils.path_operations import ensure_split_dirs, resolve_drive_path
from smartcash.dataset.augmentor.utils.batch_processor import process_batch_split_aware
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker
from smartcash.dataset.augmentor.utils.bbox_operations import save_validated_labels

class NormalizationEngine:
    """Updated normalization engine menggunakan SRP utils modules"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.progress = create_progress_tracker(communicator)
        self.comm = communicator
        self.stats = defaultdict(int)
        
    def normalize_augmented_data(self, augmented_dir: str, preprocessed_dir: str, target_split: str = "train") -> Dict[str, Any]:
        """Normalisasi menggunakan SRP modules"""
        self.progress.log_info(f"🔄 Memulai normalisasi split-aware: {target_split}")
        start_time = time.time()
        
        try:
            # Setup directories menggunakan SRP path operations
            ensure_split_dirs(preprocessed_dir, target_split)
            
            # Get augmented files menggunakan SRP file operations
            aug_files = find_augmented_files_split_aware(augmented_dir, target_split)
            if not aug_files:
                return self._error_result("Tidak ada file augmented ditemukan untuk normalisasi")
            
            self.progress.progress("step", 20, 100, f"Ditemukan {len(aug_files)} file untuk normalisasi")
            
            # Process menggunakan SRP batch processor
            normalization_processor = lambda file_path: self._normalize_single_file(file_path, preprocessed_dir, target_split)
            norm_results = process_batch_split_aware(aug_files, normalization_processor,
                                                   progress_tracker=self.progress,
                                                   operation_name="normalization",
                                                   split_context=target_split)
            
            return self._create_success_result(norm_results, time.time() - start_time, preprocessed_dir, target_split)
            
        except Exception as e:
            error_msg = f"Normalization error: {str(e)}"
            self.progress.log_error(error_msg)
            return self._error_result(error_msg)
    
    def _normalize_single_file(self, file_path: str, preprocessed_dir: str, target_split: str) -> Dict[str, Any]:
        """Normalize single file menggunakan research-quality settings"""
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Cannot read image'}
            
            original_size = image.shape[:2]
            
            # Apply normalization menggunakan config
            normalized_image = self._apply_research_normalization(image)
            
            # Generate target paths
            file_stem = Path(file_path).stem
            target_img_path = Path(preprocessed_dir) / target_split / 'images' / f"{file_stem}.jpg"
            target_label_path = Path(preprocessed_dir) / target_split / 'labels' / f"{file_stem}.txt"
            
            # Save dengan research quality
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            img_saved = cv2.imwrite(str(target_img_path), normalized_image, save_params)
            
            # Copy dan validate label
            source_label_path = str(Path(file_path).parent.parent / 'labels' / f"{file_stem}.txt")
            label_saved = self._copy_validated_label(source_label_path, str(target_label_path))
            
            return {
                'status': 'success', 'file': file_path, 'normalized_name': file_stem,
                'original_size': original_size, 'size_changed': False, 'img_saved': img_saved, 'label_saved': label_saved
            }
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _apply_research_normalization(self, image):
        """Apply research-quality normalization"""
        try:
            # Get normalization settings
            norm_config = self.config.get('preprocessing', {}).get('normalization', {})
            
            # Optional pixel normalization
            if norm_config.get('normalize_pixel_values', False):
                normalized = image.astype('float32') / 255.0
                image = (normalized * 255.0).astype('uint8')
            
            # Optional resizing
            target_size = norm_config.get('target_size', None)
            if target_size and isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                if image.shape[:2] != tuple(target_size):
                    image = cv2.resize(image, tuple(target_size), interpolation=cv2.INTER_LANCZOS4)
            
            return image
            
        except Exception:
            return image  # Return original jika normalization gagal
    
    def _copy_validated_label(self, source_label: str, target_label: str) -> bool:
        """Copy dan validate label menggunakan SRP bbox operations"""
        try:
            if not Path(source_label).exists():
                return False
            
            # Read dan validate labels
            valid_lines = []
            with open(source_label, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            coords = [float(x) for x in parts[1:5]]
                            
                            # Validate coordinates
                            if all(0.0 <= x <= 1.0 for x in coords) and coords[2] > 0.001 and coords[3] > 0.001:
                                valid_lines.append(f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                        except (ValueError, IndexError):
                            continue
            
            # Write validated labels
            Path(target_label).parent.mkdir(parents=True, exist_ok=True)
            with open(target_label, 'w') as f:
                f.writelines(valid_lines)
            
            return len(valid_lines) > 0
            
        except Exception:
            # Fallback: copy original file
            try:
                return copy_file_with_uuid_preservation(source_label, target_label)
            except Exception:
                return False
    
    def _create_success_result(self, results: List[Dict], processing_time: float, 
                             preprocessed_dir: str, target_split: str) -> Dict[str, Any]:
        """Create success result dengan detailed statistics"""
        successful = [r for r in results if r.get('status') == 'success']
        img_saved_count = sum(1 for r in successful if r.get('img_saved', False))
        label_saved_count = sum(1 for r in successful if r.get('label_saved', False))
        
        return {
            'status': 'success', 'total_files_processed': len(results), 'total_normalized': len(successful),
            'images_saved': img_saved_count, 'labels_saved': label_saved_count,
            'processing_time': processing_time, 'target_split': target_split,
            'target_dir': f"{preprocessed_dir}/{target_split}",
            'normalization_speed': len(results) / processing_time if processing_time > 0 else 0
        }
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """Create error result"""
        return {'status': 'error', 'message': message, 'total_normalized': 0}

# One-liner utilities menggunakan SRP modules
create_normalization_engine = lambda config, communicator=None: NormalizationEngine(config, communicator)
normalize_split_data = lambda config, aug_dir, prep_dir, target_split='train': create_normalization_engine(config).normalize_augmented_data(aug_dir, prep_dir, target_split)
apply_research_normalization = lambda image, config: NormalizationEngine(config)._apply_research_normalization(image)