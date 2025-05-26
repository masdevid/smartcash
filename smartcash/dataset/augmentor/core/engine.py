"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Fixed augmentation engine dengan overall + steps progress tracking dan reduced logging
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
from ..utils.dataset_detector import detect_dataset_structure

class AugmentationEngine:
    """Fixed augmentation engine dengan proper progress tracking (overall + steps)"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
        self.pipeline_factory = PipelineFactory(config, self.logger)
        self.stats = defaultdict(int)
        
    def process_raw_data(self, raw_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """Fixed process raw data dengan overall + steps progress"""
        if not output_dir:
            output_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
            
        self.logger.info(f"🚀 Memulai augmentasi dari {raw_dir} → {output_dir}")
        
        # Overall progress: Initialization
        self._update_progress("overall", 5, "Inisialisasi augmentation engine")
        
        start_time = time.time()
        
        try:
            # Smart dataset detection
            detection_result = detect_dataset_structure(raw_dir)
            
            if detection_result['status'] == 'error':
                return self._create_error_result(f"Raw directory tidak valid: {detection_result['message']}")
            
            if detection_result['total_images'] == 0:
                return self._create_error_result(f"Tidak ada gambar ditemukan di {detection_result['data_dir']}")
            
            resolved_raw_dir = detection_result['data_dir']
            self.logger.info(f"📁 Dataset terdeteksi: {detection_result['structure_type']}, {detection_result['total_images']} gambar")
            
            # Overall progress: Dataset detected
            self._update_progress("overall", 15, f"Dataset terdeteksi: {detection_result['total_images']} gambar")
            
            # Get and validate files
            raw_files = self._get_raw_files_from_detection(detection_result)
            if not raw_files:
                return self._create_error_result("Tidak ada file raw yang valid ditemukan")
            
            self._ensure_output_directory(output_dir)
            selected_files = self._select_files_for_augmentation(raw_files, resolved_raw_dir)
            
            if not selected_files:
                return self._create_error_result("Tidak ada file yang memenuhi criteria augmentasi")
            
            # Overall progress: Files selected
            self._update_progress("overall", 25, f"Terpilih {len(selected_files)} file untuk augmentasi")
            
            # Process files dengan dual progress tracking
            aug_results = self._process_files_batch(selected_files, output_dir)
            
            # Generate result
            processing_time = time.time() - start_time
            result = self._create_success_result(aug_results, processing_time, output_dir)
            
            # Overall progress: Complete
            self._update_progress("overall", 100, "Augmentasi selesai")
            self.logger.info(f"✅ Augmentasi selesai: {result['total_generated']} file dalam {processing_time:.1f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Error pada augmentation engine: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self.comm and hasattr(self.comm, 'log') and self.comm.log("error", error_msg)
            return self._create_error_result(error_msg)
    
    def _process_files_batch(self, files: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Fixed batch processing dengan proper progress tracking"""
        self.logger.info(f"⚡ Memproses {len(files)} file dengan parallelism")
        
        # Extract config parameters
        aug_config = self.config.get('augmentation', {})
        aug_type = aug_config.get('types', ['combined'])[0] if aug_config.get('types') else 'combined'
        intensity = aug_config.get('intensity', 0.7)
        num_variants = aug_config.get('num_variations', 2)
        
        # Create pipeline
        pipeline = self.pipeline_factory.create_pipeline(aug_type, intensity)
        
        results = []
        total_files = len(files)
        max_workers = min(get_optimal_thread_count(), 8)
        
        # Progress tracking - reduced intervals to prevent flooding
        progress_interval = max(1, total_files // 10)  # Report every 10%
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._process_single_file, file_path, pipeline, output_dir, num_variants): file_path for file_path in files}
            
            processed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed += 1
                    
                    # Update progress at intervals only
                    if processed % progress_interval == 0 or processed == total_files:
                        # Dual progress tracking
                        overall_progress = 25 + int((processed / total_files) * 65)  # 25-90% of overall
                        current_progress = int((processed / total_files) * 100)
                        
                        self._update_progress("overall", overall_progress, f"Memproses file: {processed}/{total_files}")
                        self._update_progress("current", current_progress, f"File {processed}/{total_files} selesai")
                        
                        # Reduced logging frequency
                        if processed % max(1, total_files // 5) == 0:
                            self.logger.info(f"📊 Progress: {processed}/{total_files} file diproses")
                        
                except Exception as e:
                    self.logger.debug(f"❌ Error processing {Path(file_path).name}: {str(e)}")
                    results.append({'status': 'error', 'file': file_path, 'error': str(e)})
                    processed += 1
        
        return results
    
    def _process_single_file(self, file_path: str, pipeline, output_dir: str, num_variants: int) -> Dict[str, Any]:
        """Process single file dengan reduced logging"""
        try:
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar'}
            
            # Smart label detection
            img_name = Path(file_path).stem
            img_dir = Path(file_path).parent
            
            potential_label_dirs = [
                img_dir.parent / 'labels',
                img_dir.parent / 'label', 
                img_dir / 'labels',
                img_dir.parent,
                img_dir
            ]
            
            label_path = None
            for label_dir in potential_label_dirs:
                potential_label = label_dir / f"{img_name}.txt"
                if potential_label.exists():
                    label_path = potential_label
                    break
            
            # Read bboxes
            bboxes, class_labels = self._read_yolo_labels(str(label_path)) if label_path else ([], [])
            
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
                    self.logger.debug(f"⚠️ Error variant {variant_idx} for {img_name}: {str(e)}")
                    continue
            
            return {'status': 'success', 'file': file_path, 'generated': generated_count, 'variants': num_variants}
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _read_yolo_labels(self, label_path: str) -> Tuple[List, List]:
        """Fixed YOLO label reader dengan proper class_id conversion"""
        bboxes, class_labels = [], []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Fixed: proper class_id conversion
                            class_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            bboxes.append(bbox)
                            class_labels.append(class_id)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.debug(f"⚠️ Error reading label {Path(label_path).name}: {str(e)}")
            
        return bboxes, class_labels
    
    def _save_augmented_files(self, image, bboxes: List, class_labels: List, filename: str, output_dir: str) -> None:
        """Save augmented files dengan fixed class_id format"""
        # Save image
        img_path = os.path.join(output_dir, 'images', f"{filename}.jpg")
        cv2.imwrite(img_path, image)
        
        # Save labels dengan proper class_id format
        if bboxes and class_labels:
            label_path = os.path.join(output_dir, 'labels', f"{filename}.txt")
            with open(label_path, 'w') as f:
                for bbox, class_id in zip(bboxes, class_labels):
                    # Fixed: ensure integer class_id
                    f.write(f"{int(class_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
    
    def _get_raw_files_from_detection(self, detection_result: Dict[str, Any]) -> List[str]:
        """Get raw files dari detection result"""
        image_files = []
        
        for img_location in detection_result['image_locations']:
            img_dir = img_location['path']
            
            try:
                if os.path.exists(img_dir):
                    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                            if os.path.isfile(os.path.join(img_dir, f)) and Path(f).suffix.lower() in valid_extensions]
                    image_files.extend(files)
            except (PermissionError, OSError):
                continue
        
        self.logger.info(f"📁 Ditemukan {len(image_files)} file gambar dari smart detection")
        return image_files
    
    def _ensure_output_directory(self, output_dir: str) -> None:
        """One-liner output directory setup"""
        [os.makedirs(path, exist_ok=True) for path in [output_dir, os.path.join(output_dir, 'images'), os.path.join(output_dir, 'labels')]]
        self.logger.info(f"📁 Output directory siap: {output_dir}")
    
    def _select_files_for_augmentation(self, raw_files: List[str], raw_dir: str) -> List[str]:
        """Select files dengan label validation"""
        if not raw_files:
            return []
        
        detection_result = detect_dataset_structure(raw_dir)
        label_locations = detection_result.get('label_locations', [])
        
        if not label_locations:
            self.logger.warning(f"⚠️ Tidak ada labels directory ditemukan")
            return raw_files[:10]  # Fallback sample
        
        primary_label_dir = label_locations[0]['path']
        
        selected = []
        for img_file in raw_files:
            img_stem = Path(img_file).stem
            
            # Search label file
            label_found = False
            for label_loc in label_locations:
                label_path = os.path.join(label_loc['path'], f"{img_stem}.txt")
                if os.path.exists(label_path):
                    label_found = True
                    break
            
            if label_found:
                selected.append(img_file)
        
        self.logger.info(f"🎯 Terpilih {len(selected)} file dengan label untuk augmentasi")
        return selected
    
    def _update_progress(self, step: str, percentage: int, message: str) -> None:
        """One-liner progress update"""
        self.comm and hasattr(self.comm, 'progress') and self.comm.progress(step, percentage, 100, message)
    
    def _create_success_result(self, results: List[Dict], processing_time: float, output_dir: str) -> Dict[str, Any]:
        """Create success result dengan statistics"""
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
    
    # One-liner error result creator
    _create_error_result = lambda self, msg: {'status': 'error', 'message': msg, 'total_files_processed': 0, 'successful_files': 0, 'total_generated': 0}