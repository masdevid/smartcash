"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Core engine untuk augmentasi dengan support berbagai transformasi
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.dataset.augmentor.core.pipeline_factory import PipelineFactory
from smartcash.dataset.augmentor.utils.file_processor import FileProcessor
from smartcash.dataset.augmentor.utils.balance_calculator import BalanceCalculator
from smartcash.dataset.augmentor.utils.filename_manager import FilenameManager
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class AugmentationEngine:
    """🎨 Core engine untuk augmentasi dengan threading dan progress tracking"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_bridge=None):
        self.logger = get_logger(__name__)
        # Config validation dengan defaults
        if config is None:
            self.logger.warning("⚠️ No config provided, menggunakan defaults")
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        # Extract augmentation config dengan safe defaults
        self.aug_config = self.config.get('augmentation', {})
        self.num_variations = self.aug_config.get('num_variations', 2)
        self.target_count = self.aug_config.get('target_count', 500)
        self.types = self.aug_config.get('types', ['combined'])
        self.intensity = self.aug_config.get('intensity', 0.7)
        self.progress = progress_bridge
        
        # Initialize components
        self.pipeline_factory = PipelineFactory(config)
        self.file_processor = FileProcessor(config)
        self.balance_calculator = BalanceCalculator(config)
        self.filename_manager = FilenameManager()
        
        # Configuration
        self.aug_config = self.config.get('augmentation', {})
        self.num_variations = self.aug_config.get('num_variations', 2)
        self.target_count = self.aug_config.get('target_count', 500)
        self.types = self.aug_config.get('types', ['combined'])
        
    def augment_split(self, target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Augment specific split dengan progress tracking"""
        try:
            self._report_progress("overall", 5, 100, f"📊 Analyzing {target_split} data", progress_callback)
            
            # Get source files
            source_files = self.file_processor.get_split_files(target_split)
            if not source_files:
                return {'status': 'error', 'message': f'No source files found for split {target_split}'}
            
            self._report_progress("overall", 15, 100, f"⚖️ Calculating class balance", progress_callback)
            
            # Calculate class balance needs
            class_balance = self.balance_calculator.calculate_needs(source_files, self.target_count)
            needed_files = self.balance_calculator.select_files_for_augmentation(source_files, class_balance)
            
            if not needed_files:
                return {'status': 'success', 'message': 'Dataset already balanced', 'total_generated': 0}
            
            self._report_progress("overall", 25, 100, f"🎨 Starting augmentation for {len(needed_files)} files", progress_callback)
            
            # Execute augmentation
            aug_result = self._execute_augmentation(needed_files, target_split, progress_callback)
            
            return aug_result
            
        except Exception as e:
            error_msg = f"🚨 Augmentation error: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_generated': 0}
    
    def _execute_augmentation(self, files: List[str], target_split: str, 
                            progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute augmentation dengan threading"""
        total_files = len(files)
        total_generated = 0
        processed_files = 0
        
        # Setup output directory
        output_dir = self._setup_output_directory(target_split)
        
        # Create augmentation pipeline
        pipeline = self._create_pipeline()
        
        def process_file(file_path: str) -> Dict[str, Any]:
            """Process single file dengan multiple variations"""
            try:
                return self._augment_single_file(file_path, pipeline, output_dir)
            except Exception as e:
                return {'status': 'error', 'file': file_path, 'error': str(e), 'generated': 0}
        
        # Execute dengan ThreadPoolExecutor
        max_workers = min(4, len(files))  # Optimal untuk I/O bound operations
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files}
            
            # Process results sebagai completion
            for future in as_completed(future_to_file):
                result = future.result()
                processed_files += 1
                total_generated += result.get('generated', 0)
                
                # Update progress (25-95% range untuk augmentation)
                progress_pct = 25 + int((processed_files / total_files) * 70)
                self._report_progress("overall", progress_pct, 100, 
                                    f"🎨 Processed {processed_files}/{total_files} files", progress_callback)
                
                if result['status'] == 'error':
                    self.logger.warning(f"⚠️ Error processing {result['file']}: {result['error']}")
        
        self.logger.success(f"✅ Augmentation complete: {total_generated} files generated from {processed_files} source files")
        
        return {
            'status': 'success',
            'total_generated': total_generated,
            'processed_files': processed_files,
            'output_dir': str(output_dir)
        }
    
    def _augment_single_file(self, file_path: str, pipeline, output_dir: Path) -> Dict[str, Any]:
        """Augment single file dengan multiple variations"""
        try:
            # Load image dan labels
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Cannot load image', 'generated': 0}
            
            # Parse filename untuk extract metadata
            parsed = self.filename_manager.parse_filename(Path(file_path).name)
            if not parsed:
                return {'status': 'error', 'file': file_path, 'error': 'Invalid filename format', 'generated': 0}
            
            # Load corresponding labels
            label_path = self.file_processor.get_label_path(file_path)
            bboxes, class_labels = self._load_labels(label_path)
            
            generated_count = 0
            
            # Generate variations
            for var_idx in range(1, self.num_variations + 1):
                try:
                    # Apply augmentation
                    augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                    
                    # Generate output filename
                    aug_filename = self.filename_manager.create_augmented_filename(parsed, var_idx)
                    
                    # Save augmented image dan labels
                    if self._save_augmented_pair(aug_image, aug_bboxes, aug_class_labels, 
                                               aug_filename, output_dir):
                        generated_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error generating variation {var_idx} for {file_path}: {str(e)}")
                    continue
            
            return {'status': 'success', 'file': file_path, 'generated': generated_count}
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e), 'generated': 0}
    
    def _create_pipeline(self):
        """Create augmentation pipeline berdasarkan config"""
        aug_type = self.types[0] if self.types else 'combined'
        intensity = self.aug_config.get('intensity', 0.7)
        
        return self.pipeline_factory.create_pipeline(aug_type, intensity)
    
    def _setup_output_directory(self, target_split: str) -> Path:
        """Setup output directory dengan structure yang benar"""
        base_dir = self.config.get('data', {}).get('dir', 'data')
        output_dir = Path(base_dir) / 'augmented' / target_split
        
        # Create directories
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _load_labels(self, label_path: str) -> tuple:
        """Load YOLO format labels"""
        bboxes = []
        class_labels = []
        
        if not Path(label_path).exists():
            return bboxes, class_labels
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        bbox = [float(x) for x in parts[1:5]]
                        
                        # Validate bbox coordinates
                        if all(0 <= x <= 1 for x in bbox) and bbox[2] > 0.001 and bbox[3] > 0.001:
                            bboxes.append(bbox)
                            class_labels.append(class_id)
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading labels from {label_path}: {str(e)}")
        
        return bboxes, class_labels
    
    def _save_augmented_pair(self, image: np.ndarray, bboxes: List, class_labels: List, 
                           filename: str, output_dir: Path) -> bool:
        """Save augmented image dan label pair"""
        try:
            # Save image
            img_path = output_dir / 'images' / f"{filename}.jpg"
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            
            if not cv2.imwrite(str(img_path), image, save_params):
                return False
            
            # Save labels
            label_path = output_dir / 'labels' / f"{filename}.txt"
            
            with open(label_path, 'w') as f:
                for bbox, class_label in zip(bboxes, class_labels):
                    # Clamp coordinates ke valid range
                    x, y, w, h = [max(0.0, min(1.0, float(coord))) for coord in bbox]
                    
                    # Skip invalid bboxes
                    if w > 0.001 and h > 0.001:
                        f.write(f"{int(class_label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"🚨 Error saving augmented pair {filename}: {str(e)}")
            return False
    
    def _report_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Report progress ke bridge dan callback"""
        if self.progress:
            self.progress.update(level, current, total, message)
        
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass


# Utility functions
def create_augmentation_engine(config: Dict[str, Any], progress_bridge=None) -> AugmentationEngine:
    """🏭 Factory untuk create augmentation engine"""
    return AugmentationEngine(config, progress_bridge)

def augment_dataset_split(config: Dict[str, Any], target_split: str = "train", 
                         progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """🚀 One-liner untuk augment dataset split"""
    engine = create_augmentation_engine(config, progress_tracker)
    return engine.augment_split(target_split, progress_callback)