"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Enhanced engine dengan directory auto-creation dan improved logging
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
    """🎨 Enhanced engine dengan auto directory creation dan improved summary"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_bridge=None):
        self.logger = get_logger(__name__)
        if config is None:
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        # Extract config
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
        
    def augment_split(self, target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Enhanced augmentation dengan auto directory setup dan improved summary"""
        try:
            self._report_progress("overall", 0, 4, "Memulai setup directory dan analisis", progress_callback)
            
            # Phase 0: Auto-create directories (NEW)
            output_dir = self._ensure_all_directories(target_split)
            
            # Phase 1: Analysis
            source_files = self.file_processor.get_split_files(target_split)
            if not source_files:
                return {'status': 'error', 'message': f'Tidak ada file sumber untuk split {target_split}'}
            
            class_balance = self.balance_calculator.calculate_needs(source_files, self.target_count)
            needed_files = self.balance_calculator.select_files_for_augmentation(source_files, class_balance)
            
            if not needed_files:
                return {
                    'status': 'success', 
                    'message': 'Dataset sudah seimbang - tidak perlu augmentasi', 
                    'total_generated': 0,
                    'summary': {
                        'source_files': len(source_files),
                        'balanced': True,
                        'output_dir': str(output_dir)
                    }
                }
            
            self._report_progress("overall", 1, 4, f"Memproses {len(needed_files)} file", progress_callback)
            
            # Phase 2: Augmentation
            aug_result = self._execute_augmentation(needed_files, target_split, progress_callback)
            
            # Phase 3: Enhanced summary (NEW)
            if aug_result.get('status') == 'success':
                enhanced_summary = self._create_summary(aug_result, source_files, target_split)
                aug_result.update(enhanced_summary)
            
            self._report_progress("overall", 4, 4, "Augmentasi selesai", progress_callback)
            
            return aug_result
            
        except Exception as e:
            error_msg = f"Error augmentasi: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'total_generated': 0}
    
    def _ensure_all_directories(self, target_split: str) -> Path:
        """🏗️ NEW: Auto-create semua directory yang diperlukan"""
        base_dir = self.config.get('data', {}).get('dir', 'data')
        
        # Directories yang perlu dibuat
        directories = [
            # Augmented directories
            f"{base_dir}/augmented/{target_split}/images",
            f"{base_dir}/augmented/{target_split}/labels",
            # Preprocessed directories (CRITICAL FIX)
            f"{base_dir}/preprocessed/{target_split}/images", 
            f"{base_dir}/preprocessed/{target_split}/labels"
        ]
        
        created_dirs = []
        for dir_path in directories:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(path))
        
        if created_dirs:
            self.logger.info(f"📁 Created {len(created_dirs)} directories: {', '.join(created_dirs)}")
        
        return Path(base_dir) / 'augmented' / target_split
    
    def _create_summary(self, aug_result: Dict[str, Any], source_files: List[str], target_split: str) -> Dict[str, Any]:
        """📊 NEW: Create detailed summary untuk UI logging"""
        total_generated = aug_result.get('total_generated', 0)
        processed_files = aug_result.get('processed_files', 0)
        
        # Calculate rates
        success_rate = (total_generated / (processed_files * self.num_variations)) * 100 if processed_files > 0 else 0
        processing_rate = (processed_files / len(source_files)) * 100 if source_files else 0
        
        # Enhanced summary
        summary = {
            'summary': {
                'input': {
                    'source_files': len(source_files),
                    'processed_files': processed_files,
                    'processing_rate': f"{processing_rate:.1f}%"
                },
                'output': {
                    'total_generated': total_generated,
                    'success_rate': f"{success_rate:.1f}%",
                    'target_variations': self.num_variations,
                    'intensity_applied': self.intensity
                },
                'configuration': {
                    'types': ', '.join(self.types),
                    'target_split': target_split,
                    'target_count': self.target_count
                },
                'directories': {
                    'augmented': f"data/augmented/{target_split}",
                    'preprocessed': f"data/preprocessed/{target_split}"
                }
            }
        }
        
        # Log enhanced summary
        self._log_detailed_summary(summary['summary'])
        
        return summary
    
    def _log_detailed_summary(self, summary: Dict[str, Any]):
        """📋 NEW: Log detailed summary untuk UI"""
        input_info = summary['input']
        output_info = summary['output']
        config_info = summary['configuration']
        
        self.logger.success(f"✅ Augmentasi berhasil!")
        self.logger.info(f"📊 Input: {input_info['processed_files']}/{input_info['source_files']} files ({input_info['processing_rate']})")
        self.logger.info(f"🎯 Output: {output_info['total_generated']} generated ({output_info['success_rate']} success rate)")
        self.logger.info(f"⚙️ Config: {config_info['types']} @ intensity {output_info['intensity_applied']}")
        self.logger.info(f"📁 Saved to: {summary['directories']['augmented']} & {summary['directories']['preprocessed']}")
    
    def _execute_augmentation(self, files: List[str], target_split: str, 
                            progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute augmentation dengan enhanced progress tracking"""
        total_files = len(files)
        total_generated = 0
        processed_files = 0
        
        # Setup output directory
        output_dir = self._setup_augmented_output_directory(target_split)
        
        # Create augmentation pipeline
        pipeline = self._create_pipeline()
        
        def process_file(file_path: str) -> Dict[str, Any]:
            try:
                return self._augment_single_file(file_path, pipeline, output_dir)
            except Exception as e:
                return {'status': 'error', 'file': file_path, 'error': str(e), 'generated': 0}
        
        # Execute dengan ThreadPoolExecutor
        max_workers = min(4, len(files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files}
            
            for future in as_completed(future_to_file):
                result = future.result()
                processed_files += 1
                total_generated += result.get('generated', 0)
                
                # Enhanced progress reporting
                progress_pct = (processed_files / total_files) * 100
                self._report_progress("current", processed_files, total_files, 
                                    f"Generated {total_generated} files ({progress_pct:.1f}%)", progress_callback)
                
                if result['status'] == 'error':
                    self.logger.warning(f"⚠️ Error {result['file']}: {result['error']}")
        
        return {
            'status': 'success',
            'total_generated': total_generated,
            'processed_files': processed_files,
            'output_dir': str(output_dir)
        }
    
    def _setup_augmented_output_directory(self, target_split: str) -> Path:
        """Setup directory untuk augmented output"""
        base_dir = self.config.get('data', {}).get('dir', 'data')
        output_dir = Path(base_dir) / 'augmented' / target_split
        
        # Create directories with auto-creation
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _augment_single_file(self, file_path: str, pipeline, output_dir: Path) -> Dict[str, Any]:
        """Augment single file dengan detailed tracking"""
        try:
            # Load image dan labels
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar', 'generated': 0}
            
            # Parse filename
            parsed = self.filename_manager.parse_filename(Path(file_path).name)
            if not parsed:
                return {'status': 'error', 'file': file_path, 'error': 'Format filename tidak valid', 'generated': 0}
            
            # Load labels
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
                    
                    # Save augmented pair
                    if self._save_augmented_pair(aug_image, aug_bboxes, aug_class_labels, 
                                               aug_filename, output_dir):
                        generated_count += 1
                        
                except Exception as e:
                    self.logger.debug(f"⚠️ Error variasi {var_idx} untuk {file_path}: {str(e)}")
                    continue
            
            return {'status': 'success', 'file': file_path, 'generated': generated_count}
            
        except Exception as e:
            return {'status': 'error', 'file': file_path, 'error': str(e), 'generated': 0}
    
    def _save_augmented_pair(self, image: np.ndarray, bboxes: List, class_labels: List, 
                           filename: str, output_dir: Path) -> bool:
        """Save augmented pair dengan error handling"""
        try:
            # Save image as regular JPEG
            img_path = output_dir / 'images' / f"{filename}.jpg"
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            
            if not cv2.imwrite(str(img_path), image, save_params):
                return False
            
            # Save labels as .txt
            label_path = output_dir / 'labels' / f"{filename}.txt"
            
            with open(label_path, 'w') as f:
                for bbox, class_label in zip(bboxes, class_labels):
                    x, y, w, h = [max(0.0, min(1.0, float(coord))) for coord in bbox]
                    
                    if w > 0.001 and h > 0.001:
                        f.write(f"{int(class_label)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error menyimpan {filename}: {str(e)}")
            return False
    
    def _create_pipeline(self):
        """Create augmentation pipeline"""
        aug_type = self.types[0] if self.types else 'combined'
        intensity = self.aug_config.get('intensity', 0.7)
        return self.pipeline_factory.create_pipeline(aug_type, intensity)
    
    def _load_labels(self, label_path: str) -> tuple:
        """Load YOLO format labels"""
        bboxes, class_labels = [], []
        
        if not Path(label_path).exists():
            return bboxes, class_labels
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            
                            if all(0 <= x <= 1 for x in bbox) and bbox[2] > 0.001 and bbox[3] > 0.001:
                                bboxes.append(bbox)
                                class_labels.append(class_id)
                        except (ValueError, IndexError):
                            continue
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading labels {label_path}: {str(e)}")
        
        return bboxes, class_labels
    
    def _report_progress(self, level: str, current: int, total: int, message: str, callback: Optional[Callable]):
        """Report progress dengan enhanced messaging"""
        if self.progress:
            self.progress.update(level, current, total, message)
        
        if callback and callable(callback):
            try:
                callback(level, current, total, message)
            except Exception:
                pass


# Utility functions
def create_augmentation_engine(config: Dict[str, Any], progress_bridge=None) -> AugmentationEngine:
    """Factory untuk create augmentation engine"""
    return AugmentationEngine(config, progress_bridge)

def augment_dataset_split(config: Dict[str, Any], target_split: str = "train", 
                         progress_tracker=None, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """One-liner untuk augment dataset split"""
    engine = create_augmentation_engine(config, progress_tracker)
    return engine.augment_split(target_split, progress_callback)