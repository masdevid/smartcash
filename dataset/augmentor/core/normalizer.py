"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: Fixed normalizer dengan proper progress updates dan reduced logging flooding
"""

import os
import shutil
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count

class NormalizationEngine:
    """Fixed normalization engine dengan proper progress tracking"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
        self.stats = defaultdict(int)
        
    def normalize_augmented_data(self, augmented_dir: str, preprocessed_dir: str, target_split: str = "train") -> Dict[str, Any]:
        """Fixed normalisasi dengan proper progress reporting"""
        self.logger.info(f"🔄 Memulai normalisasi: {augmented_dir} → {preprocessed_dir}/{target_split}")
        
        # Progress: Step initialization
        self._update_progress("step", 5, "Inisialisasi normalization engine")
        
        start_time = time.time()
        
        try:
            if not self._validate_augmented_directory(augmented_dir):
                return self._create_error_result("Directory augmented tidak valid atau kosong")
            
            target_dir = os.path.join(preprocessed_dir, target_split)
            self._setup_target_directory(target_dir)
            
            aug_files = self._get_augmented_files(augmented_dir)
            if not aug_files:
                return self._create_error_result("Tidak ada file augmented ditemukan")
            
            # Progress: Files found
            self._update_progress("step", 20, f"Ditemukan {len(aug_files)} file untuk normalisasi")
            
            # Process dengan proper progress tracking
            norm_results = self._process_normalization_batch(aug_files, target_dir)
            
            processing_time = time.time() - start_time
            result = self._create_success_result(norm_results, processing_time, target_dir)
            
            # Final progress
            self._update_progress("step", 100, "Normalisasi selesai")
            self.logger.info(f"✅ Normalisasi selesai: {result['total_normalized']} file dalam {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error pada normalization engine: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.comm and hasattr(self.comm, 'log') and self.comm.log("error", error_msg)
            return self._create_error_result(error_msg)
    
    def _process_normalization_batch(self, aug_files: List[Tuple[str, str]], target_dir: str) -> List[Dict[str, Any]]:
        """Fixed batch processing dengan proper progress updates"""
        self.logger.info(f"⚡ Normalisasi {len(aug_files)} file")
        
        norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        target_size = norm_config.get('image_size', (640, 640))
        quality = norm_config.get('jpeg_quality', 95)
        
        results = []
        total_files = len(aug_files)
        max_workers = min(get_optimal_thread_count(), 6)
        
        # Progress tracking variables
        progress_interval = max(1, total_files // 20)  # Report every 5%
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._normalize_single_file, img_path, label_path, target_dir, target_size, quality): (img_path, label_path) for img_path, label_path in aug_files}
            
            processed = 0
            for future in as_completed(future_to_file):
                img_path, label_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed += 1
                    
                    # Update progress at intervals to avoid flooding
                    if processed % progress_interval == 0 or processed == total_files:
                        progress = 20 + int((processed / total_files) * 70)
                        self._update_progress("step", progress, f"Normalisasi: {processed}/{total_files}")
                        
                except Exception as e:
                    self.logger.debug(f"❌ Error normalizing {Path(img_path).name}: {str(e)}")
                    results.append({'status': 'error', 'file': img_path, 'error': str(e)})
                    processed += 1
        
        return results
    
    def _normalize_single_file(self, img_path: str, label_path: str, target_dir: str, target_size: Tuple[int, int], quality: int) -> Dict[str, Any]:
        """Normalize single file - reduced logging"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                return {'status': 'error', 'file': img_path, 'error': 'Cannot read image'}
            
            original_size = image.shape[:2]
            
            # Resize jika diperlukan
            if original_size != target_size:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Generate normalized filename (remove aug_ prefix)
            filename = Path(img_path).stem
            normalized_filename = filename[4:] if filename.startswith('aug_') else filename
                
            # Save normalized files
            target_img_path = os.path.join(target_dir, 'images', f"{normalized_filename}.jpg")
            cv2.imwrite(target_img_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Copy and validate label
            target_label_path = os.path.join(target_dir, 'labels', f"{normalized_filename}.txt")
            self._copy_and_validate_label(label_path, target_label_path)
            
            return {
                'status': 'success',
                'file': img_path,
                'normalized_name': normalized_filename,
                'original_size': original_size,
                'target_size': target_size,
                'size_changed': original_size != target_size
            }
            
        except Exception as e:
            return {'status': 'error', 'file': img_path, 'error': str(e)}
    
    def _copy_and_validate_label(self, source_label: str, target_label: str) -> None:
        """Copy dan validate label dengan reduced logging"""
        try:
            with open(source_label, 'r') as f:
                lines = f.readlines()
            
            # Validate dan filter lines
            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        # Fixed: proper class_id conversion
                        class_id = int(float(parts[0]))
                        coords = [float(x) for x in parts[1:5]]
                        if all(0.0 <= x <= 1.0 for x in coords):
                            valid_lines.append(f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
                    except (ValueError, IndexError):
                        continue
            
            # Write validated labels
            with open(target_label, 'w') as f:
                f.writelines(valid_lines)
                
        except Exception as e:
            self.logger.debug(f"⚠️ Error processing label {Path(source_label).name}: {str(e)}")
            try:
                shutil.copy2(source_label, target_label)
            except Exception:
                pass
    
    def _validate_augmented_directory(self, augmented_dir: str) -> bool:
        """One-liner directory validation"""
        return all(os.path.exists(path) for path in [augmented_dir, os.path.join(augmented_dir, 'images'), os.path.join(augmented_dir, 'labels')])
    
    def _setup_target_directory(self, target_dir: str) -> None:
        """One-liner target directory setup"""
        [os.makedirs(path, exist_ok=True) for path in [target_dir, os.path.join(target_dir, 'images'), os.path.join(target_dir, 'labels')]]
        self.logger.info(f"📁 Target directory siap: {target_dir}")
    
    def _get_augmented_files(self, augmented_dir: str) -> List[Tuple[str, str]]:
        """Get augmented files dengan reduced logging"""
        images_dir = os.path.join(augmented_dir, 'images')
        labels_dir = os.path.join(augmented_dir, 'labels')
        
        # Get augmented files with prefix filter
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        aug_files = [
            (os.path.join(images_dir, img_file), os.path.join(labels_dir, Path(img_file).stem + '.txt'))
            for img_file in os.listdir(images_dir)
            if img_file.startswith('aug_') and Path(img_file).suffix.lower() in valid_extensions
            and os.path.exists(os.path.join(labels_dir, Path(img_file).stem + '.txt'))
        ]
                
        self.logger.info(f"📊 Ditemukan {len(aug_files)} file augmented dengan label")
        return aug_files
    
    def _update_progress(self, step: str, percentage: int, message: str) -> None:
        """One-liner progress update"""
        self.comm and hasattr(self.comm, 'progress') and self.comm.progress(step, percentage, 100, message)
    
    def _create_success_result(self, results: List[Dict], processing_time: float, target_dir: str) -> Dict[str, Any]:
        """Create success result dengan summary statistics"""
        successful = [r for r in results if r.get('status') == 'success']
        size_changes = sum(1 for r in successful if r.get('size_changed', False))
        
        return {
            'status': 'success',
            'total_files_processed': len(results),
            'total_normalized': len(successful),
            'files_resized': size_changes,
            'processing_time': processing_time,
            'target_dir': target_dir,
            'normalization_speed': len(results) / processing_time if processing_time > 0 else 0
        }
    
    # One-liner error result creator
    _create_error_result = lambda self, msg: {'status': 'error', 'message': msg, 'total_files_processed': 0, 'total_normalized': 0}