"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: Fixed normalizer dengan path handling dan progress tracking yang benar
"""

import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import time

from smartcash.dataset.augmentor.utils.file_operations import find_augmented_files_split_aware, copy_file_with_uuid_preservation
from smartcash.dataset.augmentor.utils.path_operations import ensure_split_dirs, resolve_drive_path
from smartcash.dataset.augmentor.utils.batch_processor import process_batch_split_aware
from smartcash.dataset.augmentor.utils.progress_tracker import create_progress_tracker
from smartcash.dataset.augmentor.utils.bbox_operations import save_validated_labels

class NormalizationEngine:
    """Fixed normalization engine dengan path handling dan progress yang benar"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.progress = create_progress_tracker(communicator)
        self.comm = communicator
        self.stats = defaultdict(int)
        
    def normalize_augmented_data(self, augmented_dir: str, preprocessed_dir: str, target_split: str = "train") -> Dict[str, Any]:
        """Normalisasi dengan path handling yang diperbaiki"""
        self.progress.log_info(f"🔄 Memulai normalisasi split-aware: {target_split}")
        self.progress.log_info(f"📁 Source: {augmented_dir}")
        self.progress.log_info(f"📁 Target: {preprocessed_dir}")
        
        start_time = time.time()
        
        try:
            # Setup directories
            ensure_split_dirs(preprocessed_dir, target_split)
            
            # Find augmented files dengan path yang benar
            aug_files = self._find_augmented_files_corrected(augmented_dir, target_split)
            if not aug_files:
                self.progress.log_warning(f"⚠️ Tidak ada file augmented ditemukan di {augmented_dir}")
                return self._create_empty_result(preprocessed_dir, target_split)
            
            self.progress.log_info(f"📊 Ditemukan {len(aug_files)} file untuk normalisasi")
            
            # Process dengan progress tracking
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
    
    def _find_augmented_files_corrected(self, augmented_dir: str, target_split: str) -> List[str]:
        """Find augmented files dengan path correction"""
        resolved_dir = resolve_drive_path(augmented_dir)
        
        # Try multiple path patterns
        search_patterns = [
            f"{resolved_dir}/{target_split}/images",
            f"{resolved_dir}/{target_split}",
            f"{resolved_dir}/images",
            resolved_dir
        ]
        
        aug_files = []
        for pattern_dir in search_patterns:
            try:
                if Path(pattern_dir).exists():
                    pattern_files = [str(f) for f in Path(pattern_dir).glob('aug_*.jpg')]
                    if pattern_files:
                        aug_files.extend(pattern_files)
                        self.progress.log_info(f"📂 Found {len(pattern_files)} files in {pattern_dir}")
                        break
            except Exception:
                continue
        
        return list(set(aug_files))  # Remove duplicates
    
    def _normalize_single_file(self, file_path: str, preprocessed_dir: str, target_split: str) -> Dict[str, Any]:
        """Normalize single file dengan validation"""
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Cannot read image'}
            
            original_size = image.shape[:2]
            
            # Apply normalization
            normalized_image = self._apply_research_normalization(image)
            
            # Generate target paths
            file_stem = Path(file_path).stem
            target_img_path = Path(preprocessed_dir) / target_split / 'images' / f"{file_stem}.jpg"
            target_label_path = Path(preprocessed_dir) / target_split / 'labels' / f"{file_stem}.txt"
            
            # Ensure directories exist
            target_img_path.parent.mkdir(parents=True, exist_ok=True)
            target_label_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save image dengan quality settings
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            img_saved = cv2.imwrite(str(target_img_path), normalized_image, save_params)
            
            # Copy dan validate label
            source_label_path = self._find_corresponding_label(file_path)
            label_saved = self._copy_validated_label(source_label_path, str(target_label_path))
            
            return {
                'status': 'success', 'file': file_path, 'normalized_name': file_stem,
                'target_image': str(target_img_path), 'target_label': str(target_label_path),
                'original_size': original_size, 'img_saved': img_saved, 'label_saved': label_saved
            }
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _find_corresponding_label(self, image_path: str) -> str:
        """Find corresponding label file untuk image"""
        image_path_obj = Path(image_path)
        
        # Try multiple label locations
        potential_label_paths = [
            image_path_obj.parent.parent / 'labels' / f"{image_path_obj.stem}.txt",
            image_path_obj.parent / 'labels' / f"{image_path_obj.stem}.txt",
            image_path_obj.parent / f"{image_path_obj.stem}.txt"
        ]
        
        for label_path in potential_label_paths:
            if label_path.exists():
                return str(label_path)
        
        return ""
    
    def _apply_research_normalization(self, image):
        """Apply normalization untuk kebutuhan training (float32 [0,1])"""
        try:
            norm_config = self.config.get('preprocessing', {}).get('normalization', {})
            scaler_type = norm_config.get('scaler', 'minmax')

            # Optional resizing dulu, sebelum normalisasi
            target_size = norm_config.get('target_size', None)
            if target_size and isinstance(target_size, (list, tuple)) and len(target_size) == 2:
                if image.shape[:2] != tuple(target_size):
                    image = cv2.resize(image, tuple(target_size), interpolation=cv2.INTER_LANCZOS4)

            # Konversi tipe dan normalisasi (sesuai scaler_type)
            image = image.astype('float32')

            if scaler_type == 'minmax' or scaler_type == 'none':
                image /= 255.0  # Normalisasi ke [0.0, 1.0]

            elif scaler_type == 'standard':
                mean = image.mean()
                std = image.std()
                if std > 0:
                    image = (image - mean) / std
                    # Tidak dikembalikan ke 0–255, biarkan standar distribusinya

            return image

        except Exception as e:
            return image

    
    def _copy_validated_label(self, source_label: str, target_label: str) -> bool:
        """Copy dan validate label"""
        if not source_label or not Path(source_label).exists():
            return False
        
        try:
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
        """Create success result dengan config summary di akhir"""
        successful = [r for r in results if r.get('status') == 'success']
        img_saved_count = sum(1 for r in successful if r.get('img_saved', False))
        label_saved_count = sum(1 for r in successful if r.get('label_saved', False))
        
        # Log summary results
        self.progress.log_success(f"✅ Normalisasi berhasil: {len(successful)}/{len(results)} file")
        self.progress.log_info(f"📊 Images: {img_saved_count}, Labels: {label_saved_count}")
        self.progress.log_info(f"📁 Saved to: {preprocessed_dir}/{target_split}")
        
        # Log config summary di akhir (tidak akan hilang karena reset)
        self._log_final_config_summary()
        
        return {
            'status': 'success', 'total_files_processed': len(results), 'total_normalized': len(successful),
            'images_saved': img_saved_count, 'labels_saved': label_saved_count,
            'processing_time': processing_time, 'target_split': target_split,
            'target_dir': f"{preprocessed_dir}/{target_split}",
            'normalization_speed': len(results) / processing_time if processing_time > 0 else 0
        }
    
    def _log_final_config_summary(self):
        """Log config summary di akhir proses"""
        norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        scaler = norm_config.get('scaler', 'minmax')
        target_size = norm_config.get('target_size', None)
        
        # Determine config sources
        scaler_source = "CONFIG" if 'preprocessing' in self.config and 'normalization' in self.config['preprocessing'] and 'scaler' in norm_config else "DEFAULT"
        size_source = "CONFIG" if target_size else "DEFAULT"
        
        self.progress.log_success(f"🔧 Normalization Applied:")
        self.progress.log_info(f"   • Scaler: {scaler} ({scaler_source})")
        self.progress.log_info(f"   • Resize: {target_size or 'None'} ({size_source})")
    
    def _create_empty_result(self, preprocessed_dir: str, target_split: str) -> Dict[str, Any]:
        """Create result untuk empty dataset"""
        return {
            'status': 'success', 'total_files_processed': 0, 'total_normalized': 0,
            'images_saved': 0, 'labels_saved': 0, 'processing_time': 0.0,
            'target_split': target_split, 'target_dir': f"{preprocessed_dir}/{target_split}",
            'normalization_speed': 0.0
        }
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """Create error result"""
        return {'status': 'error', 'message': message, 'total_normalized': 0}

# One-liner utilities
create_normalization_engine = lambda config, communicator=None: NormalizationEngine(config, communicator)
normalize_split_data = lambda config, aug_dir, prep_dir, target_split='train': create_normalization_engine(config).normalize_augmented_data(aug_dir, prep_dir, target_split)
apply_research_normalization = lambda image, config: NormalizationEngine(config)._apply_research_normalization(image)