"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Fixed augmentation engine dengan smart image detection untuk resolve "Tidak ada file gambar ditemukan"
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
from ..utils.core import detect_structure, smart_find_images, resolve_drive_path

class AugmentationEngine:
    """Fixed augmentation engine dengan smart image detection dan robust file handling"""
    
    def __init__(self, config: Dict[str, Any], communicator=None):
        self.config = config
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
        self.pipeline_factory = PipelineFactory(config, self.logger)
        self.stats = defaultdict(int)
        
    def process_raw_data(self, raw_dir: str, output_dir: str = None) -> Dict[str, Any]:
        """Fixed process raw data dengan enhanced image detection"""
        if not output_dir:
            output_dir = self.config.get('augmentation', {}).get('output_dir', 'data/augmented')
            
        self.logger.info(f"🚀 Memulai augmentasi dari {raw_dir} → {output_dir}")
        
        # Overall progress: Initialization
        self._update_progress("overall", 5, "Inisialisasi augmentation engine")
        
        start_time = time.time()
        
        try:
            # Enhanced dataset detection dengan smart finder
            detection_result = detect_structure(raw_dir)
            
            if detection_result['status'] == 'error':
                return self._create_error_result(f"Raw directory error: {detection_result['message']}")
            
            # Enhanced validation dengan detailed logging
            total_images = detection_result['total_images']
            image_locations = detection_result['image_locations']
            
            if total_images == 0:
                # Debug info untuk troubleshooting
                resolved_dir = resolve_drive_path(raw_dir)
                self.logger.error(f"❌ Debug Info:")
                self.logger.error(f"   • Raw dir input: {raw_dir}")
                self.logger.error(f"   • Resolved path: {resolved_dir}")
                self.logger.error(f"   • Path exists: {os.path.exists(resolved_dir)}")
                self.logger.error(f"   • Image locations checked: {len(image_locations)}")
                
                # Try manual search as last resort
                manual_images = smart_find_images(resolved_dir)
                if manual_images:
                    self.logger.info(f"🔍 Manual search found {len(manual_images)} images")
                    # Update detection result
                    detection_result['total_images'] = len(manual_images)
                    detection_result['image_locations'] = [{'path': resolved_dir, 'count': len(manual_images)}]
                else:
                    return self._create_error_result(f"Tidak ada gambar ditemukan di {resolved_dir}. Periksa path dan format file (.jpg, .png, .jpeg)")
            
            resolved_raw_dir = detection_result['data_dir']
            self.logger.info(f"📁 Dataset terdeteksi: {detection_result['structure_type']}, {total_images} gambar di {len(image_locations)} lokasi")
            
            # Log image locations untuk debugging
            for location in image_locations[:3]:  # Show first 3 locations
                self.logger.info(f"   📂 {location['path']}: {location['count']} files")
            
            # Overall progress: Dataset detected
            self._update_progress("overall", 15, f"Dataset terdeteksi: {total_images} gambar")
            
            # Get and validate files dengan enhanced approach
            raw_files = self._get_raw_files_from_detection_enhanced(detection_result)
            if not raw_files:
                return self._create_error_result("Tidak ada file raw yang valid ditemukan setelah validasi")
            
            self._ensure_output_directory(output_dir)
            selected_files = self._select_files_for_augmentation_enhanced(raw_files, resolved_raw_dir)
            
            if not selected_files:
                # Show some files that were found but not selected
                self.logger.warning(f"⚠️ Dari {len(raw_files)} file ditemukan, tidak ada yang memenuhi kriteria")
                self.logger.info(f"📋 Sample files found: {[Path(f).name for f in raw_files[:5]]}")
                return self._create_error_result("Tidak ada file yang memenuhi criteria augmentasi - periksa label files")
            
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
    
    def _get_raw_files_from_detection_enhanced(self, detection_result: Dict[str, Any]) -> List[str]:
        """Enhanced raw files extraction dengan multiple strategies"""
        image_files = []
        
        # Strategy 1: Use detection result locations
        for img_location in detection_result['image_locations']:
            img_dir = img_location['path']
            
            try:
                if os.path.exists(img_dir):
                    # Scan direct files
                    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
                    
                    # Check main directory
                    for file in os.listdir(img_dir):
                        file_path = os.path.join(img_dir, file)
                        if os.path.isfile(file_path) and Path(file).suffix.lower() in valid_extensions:
                            image_files.append(file_path)
                    
                    # Check images subdirectory
                    images_subdir = os.path.join(img_dir, 'images')
                    if os.path.exists(images_subdir):
                        for file in os.listdir(images_subdir):
                            file_path = os.path.join(images_subdir, file)
                            if os.path.isfile(file_path) and Path(file).suffix.lower() in valid_extensions:
                                image_files.append(file_path)
                        
            except (PermissionError, OSError) as e:
                self.logger.debug(f"⚠️ Cannot access {img_dir}: {str(e)}")
                continue
        
        # Strategy 2: Smart finder fallback
        if not image_files:
            self.logger.info("🔍 Menggunakan smart finder sebagai fallback")
            image_files = smart_find_images(detection_result['data_dir'])
        
        # Remove duplicates dan validate
        unique_files = list(set(image_files))
        valid_files = [f for f in unique_files if os.path.exists(f) and os.path.getsize(f) > 0]
        
        self.logger.info(f"📁 Ditemukan {len(valid_files)} file gambar valid dari smart detection")
        
        # Debug info jika sedikit file
        if len(valid_files) < 10:
            self.logger.info(f"📋 Sample files: {[Path(f).name for f in valid_files[:5]]}")
        
        return valid_files
    
    def _select_files_for_augmentation_enhanced(self, raw_files: List[str], raw_dir: str) -> List[str]:
        """Enhanced file selection dengan flexible label matching"""
        if not raw_files:
            return []
        
        self.logger.info(f"🎯 Memvalidasi {len(raw_files)} file untuk augmentasi")
        
        # Get label locations dari dataset detection
        detection_result = detect_structure(raw_dir)
        
        selected = []
        label_dirs_to_check = []
        
        # Build list of potential label directories
        for img_file in raw_files[:5]:  # Sample first 5 to find label pattern
            img_dir = Path(img_file).parent
            potential_label_dirs = [
                img_dir.parent / 'labels',      # ../labels
                img_dir / 'labels',             # ./labels  
                img_dir.parent,                 # ../
                img_dir,                        # ./
                img_dir.parent / 'label',       # ../label
                img_dir / 'label'               # ./label
            ]
            label_dirs_to_check.extend([str(d) for d in potential_label_dirs if d.exists()])
        
        # Remove duplicates
        label_dirs_to_check = list(set(label_dirs_to_check))
        self.logger.info(f"🏷️ Memeriksa label di {len(label_dirs_to_check)} direktori")
        
        # Check each image file for corresponding label
        files_with_labels = 0
        for img_file in raw_files:
            img_stem = Path(img_file).stem
            label_found = False
            
            # Search for label file
            for label_dir in label_dirs_to_check:
                label_path = os.path.join(label_dir, f"{img_stem}.txt")
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    selected.append(img_file)
                    label_found = True
                    files_with_labels += 1
                    break
            
            # Log first few files without labels for debugging
            if not label_found and len(selected) < 5:
                self.logger.debug(f"⚠️ Label tidak ditemukan untuk: {Path(img_file).name}")
        
        # Fallback: jika sangat sedikit yang punya label, ambil sample tanpa label requirement
        if len(selected) < 5 and len(raw_files) > 10:
            self.logger.warning(f"⚠️ Hanya {len(selected)} file dengan label, menggunakan sample tanpa requirement")
            selected = raw_files[:20]  # Take first 20 as sample
            files_with_labels = 0
        
        self.logger.info(f"🎯 Terpilih {len(selected)} file ({files_with_labels} dengan label) untuk augmentasi")
        
        return selected
    
    def _process_files_batch(self, files: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process files batch dengan enhanced error handling"""
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
        
        # Progress tracking
        progress_interval = max(1, total_files // 10)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(self._process_single_file_enhanced, file_path, pipeline, output_dir, num_variants): file_path for file_path in files}
            
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
                        
                        # Log progress dengan success rate
                        if processed % max(1, total_files // 5) == 0:
                            successful = sum(1 for r in results if r.get('status') == 'success')
                            success_rate = (successful / processed) * 100 if processed > 0 else 0
                            self.logger.info(f"📊 Progress: {processed}/{total_files} file ({success_rate:.1f}% berhasil)")
                        
                except Exception as e:
                    self.logger.debug(f"❌ Error processing {Path(file_path).name}: {str(e)}")
                    results.append({'status': 'error', 'file': file_path, 'error': str(e)})
                    processed += 1
        
        return results
    
    def _process_single_file_enhanced(self, file_path: str, pipeline, output_dir: str, num_variants: int) -> Dict[str, Any]:
        """Enhanced single file processing dengan robust label detection"""
        try:
            # Read and validate image
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar'}
            
            # Enhanced label detection dengan multiple strategies
            img_name = Path(file_path).stem
            img_dir = Path(file_path).parent
            
            # Strategy 1: Standard YOLO locations
            potential_label_dirs = [
                img_dir.parent / 'labels',      # Standard YOLO: ../labels/
                img_dir / 'labels',             # Alternative: ./labels/
                img_dir.parent / 'label',       # Alternative: ../label/
                img_dir / 'label',              # Alternative: ./label/
                img_dir.parent,                 # Same level as images dir
                img_dir                         # Same as image directory
            ]
            
            # Strategy 2: Sibling directory search
            if img_dir.name == 'images':
                parent_dir = img_dir.parent
                for item in parent_dir.iterdir():
                    if item.is_dir() and 'label' in item.name.lower():
                        potential_label_dirs.append(item)
            
            # Find label file
            label_path = None
            for label_dir in potential_label_dirs:
                if label_dir.exists():
                    potential_label = label_dir / f"{img_name}.txt"
                    if potential_label.exists() and potential_label.stat().st_size > 0:
                        label_path = potential_label
                        break
            
            # Read bboxes if label exists
            bboxes, class_labels = self._read_yolo_labels(str(label_path)) if label_path else ([], [])
            
            generated_count = 0
            
            # Generate variants
            for variant_idx in range(num_variants):
                try:
                    # Apply augmentation
                    if bboxes and class_labels:
                        # With bboxes
                        augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                        aug_image = augmented['image']
                        aug_bboxes = augmented['bboxes']
                        aug_labels = augmented['class_labels']
                    else:
                        # Image only (no bboxes)
                        augmented = pipeline(image=image)
                        aug_image = augmented['image']
                        aug_bboxes = []
                        aug_labels = []
                    
                    # Save augmented files
                    aug_filename = f"aug_{img_name}_v{variant_idx}"
                    self._save_augmented_files(aug_image, aug_bboxes, aug_labels, aug_filename, output_dir)
                    generated_count += 1
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Error variant {variant_idx} for {img_name}: {str(e)}")
                    continue
            
            return {
                'status': 'success', 
                'file': file_path, 
                'generated': generated_count, 
                'variants': num_variants,
                'has_labels': len(bboxes) > 0,
                'label_path': str(label_path) if label_path else None
            }
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e)}
    
    def _read_yolo_labels(self, label_path: str) -> Tuple[List, List]:
        """Enhanced YOLO label reader dengan validation"""
        bboxes, class_labels = [], []
        
        try:
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            # Validate and convert
                            class_id = int(float(parts[0]))
                            coords = [float(x) for x in parts[1:5]]
                            
                            # Validate coordinates (should be 0-1)
                            if all(0.0 <= coord <= 1.0 for coord in coords):
                                bboxes.append(coords)
                                class_labels.append(class_id)
                            else:
                                self.logger.debug(f"⚠️ Invalid coords in {Path(label_path).name}:{line_num}")
                                
                        except (ValueError, IndexError) as e:
                            self.logger.debug(f"⚠️ Parse error in {Path(label_path).name}:{line_num}: {str(e)}")
                            continue
                            
        except Exception as e:
            self.logger.debug(f"⚠️ Error reading label {Path(label_path).name}: {str(e)}")
            
        return bboxes, class_labels
    
    def _save_augmented_files(self, image, bboxes: List, class_labels: List, filename: str, output_dir: str) -> None:
        """Enhanced file saving dengan validation"""
        try:
            # Ensure output directories exist
            img_dir = os.path.join(output_dir, 'images')
            label_dir = os.path.join(output_dir, 'labels')
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            
            # Save image
            img_path = os.path.join(img_dir, f"{filename}.jpg")
            success = cv2.imwrite(img_path, image)
            
            if not success:
                raise Exception(f"Failed to save image: {img_path}")
            
            # Save labels jika ada
            if bboxes and class_labels and len(bboxes) == len(class_labels):
                label_path = os.path.join(label_dir, f"{filename}.txt")
                with open(label_path, 'w') as f:
                    for bbox, class_id in zip(bboxes, class_labels):
                        # Ensure valid format
                        if len(bbox) >= 4:
                            f.write(f"{int(class_id)} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error saving augmented files {filename}: {str(e)}")
            raise
    
    def _ensure_output_directory(self, output_dir: str) -> None:
        """Enhanced output directory setup"""
        dirs_to_create = [
            output_dir,
            os.path.join(output_dir, 'images'),
            os.path.join(output_dir, 'labels')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
        
        self.logger.info(f"📁 Output directory siap: {output_dir}")
    
    def _update_progress(self, step: str, percentage: int, message: str) -> None:
        """Progress update helper"""
        if self.comm and hasattr(self.comm, 'progress'):
            self.comm.progress(step, percentage, 100, message)
    
    def _create_success_result(self, results: List[Dict], processing_time: float, output_dir: str) -> Dict[str, Any]:
        """Enhanced success result dengan detailed statistics"""
        successful = [r for r in results if r.get('status') == 'success']
        with_labels = [r for r in successful if r.get('has_labels', False)]
        total_generated = sum(r.get('generated', 0) for r in successful)
        
        return {
            'status': 'success',
            'total_files_processed': len(results),
            'successful_files': len(successful),
            'files_with_labels': len(with_labels),
            'total_generated': total_generated,
            'processing_time': processing_time,
            'output_dir': output_dir,
            'avg_variants_per_file': total_generated / len(successful) if successful else 0,
            'processing_speed': len(results) / processing_time if processing_time > 0 else 0,
            'success_rate': (len(successful) / len(results)) * 100 if results else 0
        }
    
    def _create_error_result(self, msg: str) -> Dict[str, Any]:
        """Enhanced error result dengan debug info"""
        return {
            'status': 'error', 
            'message': msg, 
            'total_files_processed': 0, 
            'successful_files': 0, 
            'total_generated': 0,
            'timestamp': time.time()
        }