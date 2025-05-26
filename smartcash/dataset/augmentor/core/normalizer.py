"""
File: smartcash/dataset/augmentor/core/normalizer.py
Deskripsi: Normalization engine untuk post-augmentasi processing dengan optimized batch operations
"""

import os
import shutil
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import time

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count

class NormalizationEngine:
    """Engine untuk normalisasi pasca-augmentasi dengan optimized performance"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        self.logger = self.comm.logger if self.comm else get_logger(__name__)
        self.stats = defaultdict(int)
        
    def normalize_augmented_data(self, augmented_dir: str, preprocessed_dir: str, target_split: str = "train") -> Dict[str, Any]:
        """
        Normalisasi data augmented ke preprocessed dengan split targeting
        Flow: /data/augmented → /data/preprocessed/{split}
        """
        self.logger.info(f"🔄 Memulai normalisasi: {augmented_dir} → {preprocessed_dir}/{target_split}")
        if self.comm: self.comm.progress("step", 5, "Inisialisasi normalization engine")
        
        start_time = time.time()
        
        try:
            # Validate input directory
            if not self._validate_augmented_directory(augmented_dir):
                return self._create_error_result("Directory augmented tidak valid atau kosong")
            
            # Setup target directory
            target_dir = os.path.join(preprocessed_dir, target_split)
            self._setup_target_directory(target_dir)
            
            # Get augmented files
            aug_files = self._get_augmented_files(augmented_dir)
            if not aug_files:
                return self._create_error_result("Tidak ada file augmented ditemukan")
            
            if self.comm: self.comm.progress("step", 20, f"Ditemukan {len(aug_files)} file untuk normalisasi")
            
            # Process normalization
            norm_results = self._process_normalization_batch(aug_files, target_dir)
            
            # Generate summary
            processing_time = time.time() - start_time
            result = self._create_success_result(norm_results, processing_time, target_dir)
            
            self.logger.success(f"✅ Normalisasi selesai: {result['total_normalized']} file dalam {processing_time:.1f}s")
            if self.comm: self.comm.progress("step", 100, "Normalisasi selesai")
            
            return result
            
        except Exception as e:
            error_msg = f"Error pada normalization engine: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            if self.comm: self.comm.log("error", error_msg)
            return self._create_error_result(error_msg)
    
    def _validate_augmented_directory(self, augmented_dir: str) -> bool:
        """Validasi directory augmented"""
        if not os.path.exists(augmented_dir):
            return False
            
        images_dir = os.path.join(augmented_dir, 'images')
        labels_dir = os.path.join(augmented_dir, 'labels')
        
        return os.path.exists(images_dir) and os.path.exists(labels_dir)
    
    def _setup_target_directory(self, target_dir: str) -> None:
        """Setup target directory structure"""
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'labels'), exist_ok=True)
        self.logger.info(f"📁 Target directory siap: {target_dir}")
    
    def _get_augmented_files(self, augmented_dir: str) -> List[Tuple[str, str]]:
        """Get augmented files (image, label) pairs"""
        images_dir = os.path.join(augmented_dir, 'images')
        labels_dir = os.path.join(augmented_dir, 'labels')
        
        # Get all augmented image files (dengan prefix aug_)
        aug_files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for img_file in os.listdir(images_dir):
            if not img_file.startswith('aug_'):
                continue
                
            if Path(img_file).suffix.lower() not in valid_extensions:
                continue
                
            img_path = os.path.join(images_dir, img_file)
            
            # Check corresponding label
            label_name = Path(img_file).stem + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            
            if os.path.exists(label_path):
                aug_files.append((img_path, label_path))
                
        self.logger.info(f"📊 Ditemukan {len(aug_files)} file augmented dengan label")
        return aug_files
    
    def _process_normalization_batch(self, aug_files: List[Tuple[str, str]], target_dir: str) -> List[Dict[str, Any]]:
        """Process normalization dalam batch dengan ThreadPoolExecutor"""
        self.logger.info(f"⚡ Normalisasi {len(aug_files)} file dengan parallelism")
        
        # Get normalization config
        norm_config = self.config.get('preprocessing', {}).get('normalization', {})
        target_size = norm_config.get('image_size', (640, 640))
        quality = norm_config.get('jpeg_quality', 95)
        
        results = []
        total_files = len(aug_files)
        
        # Process dengan ThreadPoolExecutor untuk I/O bound operations
        max_workers = min(get_optimal_thread_count(), 6)  # Conservative untuk Colab
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            future_to_file = {
                executor.submit(self._normalize_single_file, img_path, label_path, target_dir, target_size, quality): (img_path, label_path)
                for img_path, label_path in aug_files
            }
            
            # Collect results dengan progress tracking
            processed = 0
            for future in as_completed(future_to_file):
                img_path, label_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed += 1
                    
                    # Update progress
                    progress = 20 + int((processed / total_files) * 70)
                    if self.comm: self.comm.progress("step", progress, f"Dinormalisasi: {processed}/{total_files}")
                    
                    if processed % max(1, total_files // 10) == 0:
                        self.logger.info(f"📊 Progress normalisasi: {processed}/{total_files} file")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error normalizing {img_path}: {str(e)}")
                    results.append({'status': 'error', 'file': img_path, 'error': str(e)})
        
        return results
    
    def _normalize_single_file(self, img_path: str, label_path: str, target_dir: str, target_size: Tuple[int, int], quality: int) -> Dict[str, Any]:
        """Normalize single file dengan size dan quality optimization"""
        try:
            # Read dan validate image
            image = cv2.imread(img_path)
            if image is None:
                return {'status': 'error', 'file': img_path, 'error': 'Tidak dapat membaca gambar'}
            
            original_size = image.shape[:2]  # (height, width)
            
            # Resize image jika diperlukan
            if original_size != target_size:
                image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # Get filename tanpa aug_ prefix untuk normalisasi
            filename = Path(img_path).stem
            if filename.startswith('aug_'):
                normalized_filename = filename[4:]  # Remove 'aug_' prefix
            else:
                normalized_filename = filename
                
            # Save normalized image
            target_img_path = os.path.join(target_dir, 'images', f"{normalized_filename}.jpg")
            cv2.imwrite(target_img_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Copy dan normalize label
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
        """Copy dan validate label file"""
        try:
            # Read source label
            with open(source_label, 'r') as f:
                lines = f.readlines()
            
            # Validate dan write target label
            valid_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Validate YOLO format: class_id x_center y_center width height
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        
                        # Validate coordinates (0-1 range)
                        if all(0.0 <= coord <= 1.0 for coord in coords):
                            valid_lines.append(line)
                        else:
                            self.logger.warning(f"⚠️ Invalid coordinates in {source_label}: {line.strip()}")
                    except ValueError:
                        self.logger.warning(f"⚠️ Invalid format in {source_label}: {line.strip()}")
            
            # Write validated labels
            with open(target_label, 'w') as f:
                f.writelines(valid_lines)
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error processing label {source_label}: {str(e)}")
            # Fallback: copy as-is
            shutil.copy2(source_label, target_label)
    
    def cleanup_augmented_prefix(self, preprocessed_dir: str, target_split: str = "train") -> Dict[str, Any]:
        """
        Cleanup aug_ prefix dari file di preprocessed directory
        Utility untuk clean naming convention
        """
        self.logger.info(f"🧹 Cleanup aug_ prefix di {preprocessed_dir}/{target_split}")
        
        target_dir = os.path.join(preprocessed_dir, target_split)
        if not os.path.exists(target_dir):
            return {'status': 'error', 'message': f'Target directory tidak ditemukan: {target_dir}'}
        
        cleanup_count = 0
        
        # Cleanup images
        images_dir = os.path.join(target_dir, 'images')
        if os.path.exists(images_dir):
            cleanup_count += self._cleanup_prefix_in_directory(images_dir, 'aug_')
        
        # Cleanup labels  
        labels_dir = os.path.join(target_dir, 'labels')
        if os.path.exists(labels_dir):
            cleanup_count += self._cleanup_prefix_in_directory(labels_dir, 'aug_')
        
        self.logger.success(f"✅ Cleanup selesai: {cleanup_count} file renamed")
        
        return {
            'status': 'success',
            'cleanup_count': cleanup_count,
            'target_dir': target_dir
        }
    
    def _cleanup_prefix_in_directory(self, directory: str, prefix: str) -> int:
        """Cleanup prefix dari nama file dalam directory"""
        cleanup_count = 0
        
        for filename in os.listdir(directory):
            if filename.startswith(prefix):
                old_path = os.path.join(directory, filename)
                new_filename = filename[len(prefix):]  # Remove prefix
                new_path = os.path.join(directory, new_filename)
                
                try:
                    # Rename file
                    os.rename(old_path, new_path)
                    cleanup_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Error renaming {filename}: {str(e)}")
        
        return cleanup_count
    
    def get_augmented_stats(self, augmented_dir: str) -> Dict[str, Any]:
        """Get statistik file augmented untuk monitoring"""
        if not os.path.exists(augmented_dir):
            return {'status': 'error', 'message': 'Directory tidak ditemukan'}
        
        images_dir = os.path.join(augmented_dir, 'images')
        labels_dir = os.path.join(augmented_dir, 'labels')
        
        # Count augmented files
        aug_images = len([f for f in os.listdir(images_dir) if f.startswith('aug_')]) if os.path.exists(images_dir) else 0
        aug_labels = len([f for f in os.listdir(labels_dir) if f.startswith('aug_')]) if os.path.exists(labels_dir) else 0
        
        # Analyze class distribution dari augmented labels
        class_distribution = defaultdict(int)
        if os.path.exists(labels_dir):
            for label_file in os.listdir(labels_dir):
                if label_file.startswith('aug_'):
                    label_path = os.path.join(labels_dir, label_file)
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_distribution[class_id] += 1
                    except Exception:
                        continue
        
        return {
            'status': 'success',
            'aug_images': aug_images,
            'aug_labels': aug_labels,
            'class_distribution': dict(class_distribution),
            'total_instances': sum(class_distribution.values()),
            'unique_classes': len(class_distribution)
        }
    
    def _create_success_result(self, results: List[Dict], processing_time: float, target_dir: str) -> Dict[str, Any]:
        """Create success result summary"""
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
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'status': 'error',
            'message': error_message,
            'total_files_processed': 0,
            'total_normalized': 0
        }

# One-liner utilities
create_normalization_engine = lambda config, comm=None: NormalizationEngine(config, comm)
normalize_augmented_to_split = lambda config, aug_dir, prep_dir, split='train', comm=None: NormalizationEngine(config, comm).normalize_augmented_data(aug_dir, prep_dir, split)
cleanup_aug_prefix = lambda config, prep_dir, split='train': NormalizationEngine(config).cleanup_augmented_prefix(prep_dir, split)
get_augmented_file_stats = lambda config, aug_dir: NormalizationEngine(config).get_augmented_stats(aug_dir)