"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Main augmentation engine untuk proses augmentasi dari raw data dengan one-liner optimizations
"""

import os
import cv2
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import time

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count
from .pipeline import PipelineFactory

class AugmentationEngine:
    """Core engine untuk proses augmentasi dari raw data dengan optimized performance"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        self.logger = self.comm.logger if self.comm else get_logger(__name__)
        self.pipeline_factory = PipelineFactory(config, self.logger)
        self.stats = defaultdict(int)
        
    def process_raw_data(self, raw_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Process raw data untuk augmentasi dengan optimized flow
        Flow: /data → /data/augmented
        """
        if not output_dir:
            output_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
            
        self.logger.info(f"🚀 Memulai augmentasi dari {raw_dir} → {output_dir}")
        if self.comm: self.comm.progress("overall", 5, "Inisialisasi augmentation engine")
        
        start_time = time.time()
        
        try:
            # Get raw files dengan validation
            raw_files = self._get_raw_files(raw_dir)
            if not raw_files:
                return self._create_error_result("Tidak ada file raw yang valid ditemukan")
            
            # Setup output directory
            self._ensure_output_directory(output_dir)
            
            # Filter files untuk augmentasi berdasarkan class balancing
            selected_files = self._select_files_for_augmentation(raw_files)
            if not selected_files:
                return self._create_error_result("Tidak ada file yang memenuhi criteria augmentasi")
            
            if self.comm: self.comm.progress("overall", 20, f"Terpilih {len(selected_files)} file untuk augmentasi")
            
            # Process files dengan batch processing
            aug_results = self._process_files_batch(selected_files, output_dir)
            
            # Generate summary
            processing_time = time.time() - start_time
            result = self._create_success_result(aug_results, processing_time, output_dir)
            
            self.logger.success(f"✅ Augmentasi selesai: {result['total_generated']} file dalam {processing_time:.1f}s")
            if self.comm: self.comm.progress("overall", 100, "Augmentasi selesai")
            
            return result
            
        except Exception as e:
            error_msg = f"Error pada augmentation engine: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            if self.comm: self.comm.log("error", error_msg)
            return self._create_error_result(error_msg)
    
    def _get_raw_files(self, raw_dir: str) -> List[str]:
        """Get raw image files dengan validation"""
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Raw directory tidak ditemukan: {raw_dir}")
            
        images_dir = os.path.join(raw_dir, 'images')
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory tidak ditemukan: {images_dir}")
            
        # Get all image files
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        raw_files = [
            os.path.join(images_dir, f) for f in os.listdir(images_dir)
            if Path(f).suffix.lower() in valid_extensions
        ]
        
        self.logger.info(f"📁 Ditemukan {len(raw_files)} file gambar di {images_dir}")
        return raw_files
    
    def _ensure_output_directory(self, output_dir: str) -> None:
        """Setup output directory structure"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
        self.logger.info(f"📁 Output directory siap: {output_dir}")
    
    def _select_files_for_augmentation(self, raw_files: List[str]) -> List[str]:
        """Select files berdasarkan strategy (simplified untuk sekarang)"""
        # Untuk implementasi saat ini, ambil semua file yang memiliki label
        raw_dir = Path(raw_files[0]).parent.parent  # ../images -> ..
        labels_dir = raw_dir / 'labels'
        
        if not labels_dir.exists():
            self.logger.warning(f"⚠️ Labels directory tidak ditemukan: {labels_dir}")
            return raw_files[:10]  # Fallback ambil 10 file pertama
        
        # Filter files yang memiliki label
        selected = []
        for img_file in raw_files:
            img_name = Path(img_file).stem
            label_file = labels_dir / f"{img_name}.txt"
            if label_file.exists():
                selected.append(img_file)
                
        self.logger.info(f"🎯 Terpilih {len(selected)} file dengan label untuk augmentasi")
        return selected
    
    def _process_files_batch(self, files: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process files dalam batch dengan ThreadPoolExecutor"""
        self.logger.info(f"⚡ Memproses {len(files)} file dengan parallelism")
        
        # Setup augmentation parameters
        aug_config = self.config.get('augmentation', {})
        aug_type = aug_config.get('type', 'combined')
        intensity = aug_config.get('intensity', 0.5)
        num_variants = aug_config.get('num_variants', 2)
        
        # Create pipeline
        pipeline = self.pipeline_factory.create_pipeline(aug_type, intensity)
        
        results = []
        total_files = len(files)
        
        # Process dengan ThreadPoolExecutor untuk I/O bound operations
        max_workers = min(get_optimal_thread_count(), 8)  # Limit untuk Colab
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            future_to_file = {
                executor.submit(self._process_single_file, file_path, pipeline, output_dir, num_variants): file_path
                for file_path in files
            }
            
            # Collect results dengan progress tracking
            processed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed += 1
                    
                    # Update progress
                    progress = 20 + int((processed / total_files) * 70)
                    if self.comm: self.comm.progress("overall", progress, f"Diproses: {processed}/{total_files}")
                    
                    if processed % max(1, total_files // 10) == 0:
                        self.logger.info(f"📊 Progress: {processed}/{total_files} file diproses")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing {file_path}: {str(e)}")
                    results.append({'status': 'error', 'file': file_path, 'error': str(e)})
        
        return results
    
    def _process_single_file(self, file_path: str, pipeline, output_dir: str, num_variants: int) -> Dict[str, Any]:
        """Process single file dengan multiple variants"""
        try:
            # Read image
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar'}
            
            # Get corresponding label
            raw_dir = Path(file_path).parent.parent
            labels_dir = raw_dir / 'labels'
            img_name = Path(file_path).stem
            label_path = labels_dir / f"{img_name}.txt"
            
            # Read bboxes
            bboxes, class_labels = self._read_yolo_labels(str(label_path)) if label_path.exists() else ([], [])
            
            generated_count = 0
            
            # Generate variants
            for variant_idx in range(num_variants):
                try:
                    # Apply augmentation
                    augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    # Save augmented files
                    aug_filename = f"aug_{img_name}_v{variant_idx}"
                    self._save_augmented_files(aug_image, aug_bboxes, aug_labels, aug_filename, output_dir)
                    generated_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error variant {variant_idx} for {img_name}: {str(e)}")
                    continue
            
            return {
                'status': 'success',
                'file': file_path,
                'generated': generated_count,
                'variants': num_variants
            }
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _read_yolo_labels(self, label_path: str) -> Tuple[List, List]:
        """Read YOLO format labels"""
        bboxes, class_labels = [], []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]  # x_center, y_center, width, height
                        bboxes.append(bbox)
                        class_labels.append(class_id)
        except Exception as e:
            self.logger.warning(f"⚠️ Error reading label {label_path}: {str(e)}")
            
        return bboxes, class_labels
    
    def _save_augmented_files(self, image, bboxes: List, class_labels: List, filename: str, output_dir: str) -> None:
        """Save augmented image dan label files"""
        # Save image
        img_path = os.path.join(output_dir, 'images', f"{filename}.jpg")
        cv2.imwrite(img_path, image)
        
        # Save labels
        if bboxes and class_labels:
            label_path = os.path.join(output_dir, 'labels', f"{filename}.txt")
            with open(label_path, 'w') as f:
                for bbox, class_id in zip(bboxes, class_labels):
                    # Convert back to YOLO format
                    line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
                    f.write(line)
    
    def _create_success_result(self, results: List[Dict], processing_time: float, output_dir: str) -> Dict[str, Any]:
        """Create success result summary"""
        successful = [r for r in results if r.get('status') == 'success']
        total_generated = sum(r.get('generated', 0) for r in successful)
        
        return {
            'status': 'success',
            'total_files_processed': len(results),
            'successful_files': len(successful),
            'total_generated': total_generated,
            'processing_time': processing_time,
            'output_dir': output_dir,
            'avg_variants_per_file': total_generated / len(successful) if successful else 0,
            'processing_speed': len(results) / processing_time if processing_time > 0 else 0
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'status': 'error',
            'message': error_message,
            'total_files_processed': 0,
            'successful_files': 0,
            'total_generated': 0
        }

# One-liner utilities
create_augmentation_engine = lambda config, comm=None: AugmentationEngine(config, comm)
process_raw_to_augmented = lambda config, raw_dir, comm=None: AugmentationEngine(config, comm).process_raw_data(raw_dir)
validate_raw_directory = lambda raw_dir: os.path.exists(raw_dir) and os.path.exists(os.path.join(raw_dir, 'images'))