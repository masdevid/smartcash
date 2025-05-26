"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Fixed augmentation engine dengan proper Google Drive path resolution dan smart directory detection
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
    """Fixed core engine dengan proper Google Drive path resolution dan smart detection"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        # One-liner logger setup dengan fallback
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
        self.pipeline_factory = PipelineFactory(config, self.logger)
        self.stats = defaultdict(int)
        
    def process_raw_data(self, raw_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """Process raw data dengan fixed Google Drive path resolution."""
        if not output_dir:
            output_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
            
        self.logger.info(f"🚀 Memulai augmentasi dari {raw_dir} → {output_dir}")
        # One-liner progress update dengan safe communicator access
        self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 5, 100, "Inisialisasi augmentation engine")
        
        start_time = time.time()
        
        try:
            # Use dataset detector untuk smart path resolution
            detection_result = detect_dataset_structure(raw_dir)
            
            if detection_result['status'] == 'error':
                return self._create_error_result(f"Raw directory tidak valid: {detection_result['message']}")
            
            if detection_result['total_images'] == 0:
                return self._create_error_result(f"Tidak ada gambar ditemukan di {detection_result['data_dir']}")
            
            # Use resolved path dari detector
            resolved_raw_dir = detection_result['data_dir']
            self.logger.info(f"📁 Dataset terdeteksi: {detection_result['structure_type']}, {detection_result['total_images']} gambar")
            
            # Get raw files dari detection result
            raw_files = self._get_raw_files_from_detection(detection_result)
            if not raw_files:
                return self._create_error_result("Tidak ada file raw yang valid ditemukan")
            
            # Setup dan process files
            self._ensure_output_directory(output_dir)
            selected_files = self._select_files_for_augmentation(raw_files, resolved_raw_dir)
            
            if not selected_files:
                return self._create_error_result("Tidak ada file yang memenuhi criteria augmentasi")
            
            # One-liner progress update
            self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 20, 100, f"Terpilih {len(selected_files)} file untuk augmentasi")
            
            # Process files dengan batch processing
            aug_results = self._process_files_batch(selected_files, output_dir)
            
            # Generate result
            processing_time = time.time() - start_time
            result = self._create_success_result(aug_results, processing_time, output_dir)
            
            self.logger.success if hasattr(self.logger, 'success') else self.logger.info(f"✅ Augmentasi selesai: {result['total_generated']} file dalam {processing_time:.1f}s")
            # One-liner final progress
            self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", 100, 100, "Augmentasi selesai")
            
            return result
            
        except Exception as e:
            error_msg = f"Error pada augmentation engine: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            # One-liner error logging dengan safe access
            self.comm and hasattr(self.comm, 'log') and self.comm.log("error", error_msg)
            return self._create_error_result(error_msg)
    
    def _get_raw_files_from_detection(self, detection_result: Dict[str, Any]) -> List[str]:
        """Get raw files dari detection result dengan Google Drive support."""
        image_files = []
        
        # Extract files dari image locations
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
    
    def _get_raw_files(self, raw_dir: str) -> List[str]:
        """Get raw image files dengan validation - kept for compatibility."""
        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Raw directory tidak ditemukan: {raw_dir}")
            
        images_dir = os.path.join(raw_dir, 'images')
        if not os.path.exists(images_dir):
            # Try alternative structure - langsung di root
            if os.path.exists(raw_dir):
                valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
                raw_files = [os.path.join(raw_dir, f) for f in os.listdir(raw_dir) 
                            if os.path.isfile(os.path.join(raw_dir, f)) and Path(f).suffix.lower() in valid_extensions]
                self.logger.info(f"📁 Menggunakan struktur root: {len(raw_files)} file gambar")
                return raw_files
            else:
                raise FileNotFoundError(f"Images directory tidak ditemukan: {images_dir}")
            
        # One-liner file collection
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        raw_files = [os.path.join(images_dir, f) for f in os.listdir(images_dir) if Path(f).suffix.lower() in valid_extensions]
        
        self.logger.info(f"📁 Ditemukan {len(raw_files)} file gambar di {images_dir}")
        return raw_files
    
    def _ensure_output_directory(self, output_dir: str) -> None:
        """One-liner output directory setup dengan Google Drive compatibility."""
        # Resolve output path untuk Google Drive
        if output_dir.startswith('data/') and not os.path.exists(output_dir):
            # Try resolve to Drive path
            drive_base = '/content/drive/MyDrive/SmartCash'
            if os.path.exists(drive_base):
                output_dir = os.path.join(drive_base, output_dir)
        
        [os.makedirs(path, exist_ok=True) for path in [output_dir, os.path.join(output_dir, 'images'), os.path.join(output_dir, 'labels')]]
        self.logger.info(f"📁 Output directory siap: {output_dir}")
    
    def _select_files_for_augmentation(self, raw_files: List[str], raw_dir: str) -> List[str]:
        """Select files dengan label validation menggunakan smart detection."""
        if not raw_files:
            return []
        
        # Detect label directory dari raw_dir structure
        detection_result = detect_dataset_structure(raw_dir)
        label_locations = detection_result.get('label_locations', [])
        
        if not label_locations:
            self.logger.warning(f"⚠️ Tidak ada labels directory ditemukan")
            return raw_files[:10]  # Fallback dengan sample files
        
        # Use first label location sebagai primary
        primary_label_dir = label_locations[0]['path']
        
        # One-liner file selection dengan label check
        selected = []
        for img_file in raw_files:
            img_stem = Path(img_file).stem
            
            # Search label file di berbagai lokasi
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
    
    def _process_files_batch(self, files: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process files dengan fixed progress tracking."""
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
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit jobs
            future_to_file = {executor.submit(self._process_single_file, file_path, pipeline, output_dir, num_variants): file_path for file_path in files}
            
            # Collect results dengan progress tracking
            processed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                    processed += 1
                    
                    # One-liner progress update
                    progress = 20 + int((processed / total_files) * 70)
                    self.comm and hasattr(self.comm, 'progress') and self.comm.progress("overall", progress, 100, f"Diproses: {processed}/{total_files}")
                    
                    # Periodic logging
                    processed % max(1, total_files // 10) == 0 and self.logger.info(f"📊 Progress: {processed}/{total_files} file diproses")
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing {file_path}: {str(e)}")
                    results.append({'status': 'error', 'file': file_path, 'error': str(e)})
        
        return results
    
    def _process_single_file(self, file_path: str, pipeline, output_dir: str, num_variants: int) -> Dict[str, Any]:
        """Process single file dengan variants generation dan smart label detection."""
        try:
            # Read image dan get label path
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar'}
            
            # Smart label detection
            img_name = Path(file_path).stem
            img_dir = Path(file_path).parent
            
            # Try multiple label directory structures
            potential_label_dirs = [
                img_dir.parent / 'labels',  # Standard YOLO: data/labels
                img_dir.parent / 'label',   # Alternative: data/label
                img_dir / 'labels',         # Same level: images/labels
                img_dir.parent,             # Root level mixed
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
            
            # Generate variants dengan error handling
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
            
            return {'status': 'success', 'file': file_path, 'generated': generated_count, 'variants': num_variants}
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _read_yolo_labels(self, label_path: str) -> Tuple[List, List]:
        """One-liner YOLO label reader."""
        bboxes, class_labels = [], []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        bboxes.append(bbox)
                        class_labels.append(class_id)
        except Exception as e:
            self.logger.warning(f"⚠️ Error reading label {label_path}: {str(e)}")
            
        return bboxes, class_labels
    
    def _save_augmented_files(self, image, bboxes: List, class_labels: List, filename: str, output_dir: str) -> None:
        """Save augmented files dengan one-liner."""
        # Save image
        img_path = os.path.join(output_dir, 'images', f"{filename}.jpg")
        cv2.imwrite(img_path, image)
        
        # Save labels
        if bboxes and class_labels:
            label_path = os.path.join(output_dir, 'labels', f"{filename}.txt")
            with open(label_path, 'w') as f:
                [f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n") 
                 for bbox, class_id in zip(bboxes, class_labels)]
    
    def _create_success_result(self, results: List[Dict], processing_time: float, output_dir: str) -> Dict[str, Any]:
        """Create success result dengan one-liner calculations."""
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

# One-liner factory functions
create_augmentation_engine = lambda config, comm=None: AugmentationEngine(config, comm)
process_raw_to_augmented = lambda config, raw_dir, comm=None: AugmentationEngine(config, comm).process_raw_data(raw_dir)
validate_raw_directory = lambda raw_dir: detect_dataset_structure(raw_dir)['status'] == 'success' and detect_dataset_structure(raw_dir)['total_images'] > 0