"""
File: smartcash/dataset/augmentor/core/engine.py
Deskripsi: Enhanced engine dengan FileNamingManager integration, variance support dan live preview API
"""

import cv2
import numpy as np
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.dataset.augmentor.core.pipeline_factory import PipelineFactory
from smartcash.dataset.augmentor.utils.file_processor import FileProcessor
from smartcash.dataset.augmentor.utils.balance_calculator import BalanceCalculator
from smartcash.common.utils.file_naming_manager import FileNamingManager, create_file_naming_manager
from smartcash.common.logger import get_logger
from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config

class AugmentationEngine:
    """🎨 Enhanced engine dengan FileNamingManager integration, variance support dan live preview"""
    
    def __init__(self, config: Dict[str, Any] = None, progress_bridge=None):
        self.logger = get_logger(__name__)
        if config is None:
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        # Extract config
        self.aug_config = self.config.get('augmentation', {})
        self.num_variations = self.aug_config.get('num_variations', 2)
        self.target_count = self.aug_config.get('target_count', 500)  # Use proposed config
        self.types = self.aug_config.get('types', ['combined'])
        self.intensity = self.aug_config.get('intensity', 0.7)
        self.progress = progress_bridge
        
        # Initialize components dengan FileNamingManager
        self.pipeline_factory = PipelineFactory(config)
        self.file_processor = FileProcessor(config)
        self.balance_calculator = BalanceCalculator(config)
        self.naming_manager = create_file_naming_manager(config)  # NEW: FileNamingManager
    
    def create_live_preview(self, target_split: str = 'train') -> Dict[str, Any]:
        """🎥 NEW: Create live preview augmentation tanpa normalization"""
        try:
            self.logger.info(f"🎥 Creating augmentation preview untuk {target_split}")
            
            # Get random source file
            source_files = self.file_processor.get_split_files(target_split)
            if not source_files:
                return {
                    'status': 'error', 
                    'message': f'Tidak ada file sumber untuk preview di split {target_split}',
                    'preview_path': None
                }
            
            # Random select 1 file
            selected_file = random.choice(source_files)
            
            # Load image dan labels
            image = cv2.imread(selected_file)
            if image is None:
                return {
                    'status': 'error',
                    'message': f'Tidak dapat membaca gambar: {selected_file}',
                    'preview_path': None
                }
            
            # Load labels
            label_path = self.file_processor.get_label_path(selected_file)
            bboxes, class_labels = self._load_labels(label_path)
            
            # Create augmentation pipeline
            pipeline = self._create_pipeline()
            
            # Apply augmentation
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            
            # Save preview to data/aug_preview.jpg
            data_dir = self.config.get('data', {}).get('dir', 'data')
            preview_path = Path(data_dir) / 'aug_preview.jpg'
            
            # Ensure data directory exists
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save dengan high quality
            save_params = [cv2.IMWRITE_JPEG_QUALITY, 95, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            success = cv2.imwrite(str(preview_path), aug_image, save_params)
            
            if not success:
                return {
                    'status': 'error',
                    'message': 'Gagal menyimpan preview image',
                    'preview_path': None
                }
            
            self.logger.success(f"✅ Preview berhasil dibuat: {preview_path}")
            
            return {
                'status': 'success',
                'message': f'Preview augmentation dari {Path(selected_file).name}',
                'preview_path': str(preview_path),
                'source_file': selected_file,
                'source_filename': Path(selected_file).name,
                'augmentation_applied': self._get_applied_transforms()
            }
            
        except Exception as e:
            error_msg = f"Error creating preview: {str(e)}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'preview_path': None
            }
    
    def _get_applied_transforms(self) -> Dict[str, Any]:
        """Get info about applied transforms dari config structure yang diusulkan"""
        # Extract dari combined section sesuai config structure
        combined_config = self.aug_config.get('combined', {})
        
        transforms = {}
        transforms['horizontal_flip'] = combined_config.get('horizontal_flip', 0.5)
        transforms['rotation_limit'] = combined_config.get('rotation_limit', 12)
        transforms['scale_limit'] = combined_config.get('scale_limit', 0.04)
        transforms['translate_limit'] = combined_config.get('translate_limit', 0.08)
        transforms['brightness_limit'] = combined_config.get('brightness_limit', 0.2)
        transforms['contrast_limit'] = combined_config.get('contrast_limit', 0.15)
        transforms['hsv_hue'] = combined_config.get('hsv_hue', 10)
        transforms['hsv_saturation'] = combined_config.get('hsv_saturation', 15)
        
        return transforms
        
    def augment_split(self, target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """🎯 Enhanced augmentation dengan FileNamingManager dan variance support"""
        try:
            self._report_progress("overall", 0, 4, "Memulai setup directory dan analisis", progress_callback)
            
            # Phase 0: Auto-create directories
            output_dir = self._ensure_all_directories(target_split)
            
            # Phase 1: Analysis dengan FileNamingManager
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
            
            # Phase 2: Augmentation dengan variance tracking
            aug_result = self._execute_augmentation(needed_files, target_split, progress_callback)
            
            # Phase 3: Enhanced summary
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
        """🏗️ Auto-create semua directory yang diperlukan"""
        base_dir = self.config.get('data', {}).get('dir', 'data')
        
        # Directories yang perlu dibuat
        directories = [
            # Augmented directories
            f"{base_dir}/augmented/{target_split}/images",
            f"{base_dir}/augmented/{target_split}/labels",
            # Preprocessed directories
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
            self.logger.info(f"📁 Created {len(created_dirs)} directories")
        
        return Path(base_dir) / 'augmented' / target_split
    
    def _execute_augmentation(self, files: List[str], target_split: str, 
                            progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """Execute augmentation dengan variance support dan FileNamingManager"""
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
    
    def _augment_single_file(self, file_path: str, pipeline, output_dir: Path) -> Dict[str, Any]:
        """Augment single file dengan FileNamingManager dan variance support"""
        try:
            # Load image dan labels
            image = cv2.imread(file_path)
            if image is None:
                return {'status': 'error', 'file': file_path, 'error': 'Tidak dapat membaca gambar', 'generated': 0}
            
            # Extract class ID untuk naming
            label_path = self.file_processor.get_label_path(file_path)
            primary_class_id = self.naming_manager.extract_primary_class_from_label(Path(label_path))
            
            # Generate file naming info
            original_filename = Path(file_path).name
            generated_count = 0
            
            # Load labels
            bboxes, class_labels = self._load_labels(label_path)
            
            # Generate variations dengan variance support
            for var_idx in range(1, self.num_variations + 1):
                try:
                    # Apply augmentation
                    augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_class_labels = augmented['class_labels']
                    
                    # Generate filename dengan FileNamingManager variance support
                    file_info = self.naming_manager.generate_file_info(
                        original_filename, 
                        primary_class_id, 
                        source_type='augmented',
                        variance=var_idx  # NEW: Variance support
                    )
                    aug_filename = file_info.get_filename('.jpg')
                    
                    # Save augmented pair
                    if self._save_augmented_pair(aug_image, aug_bboxes, aug_class_labels, 
                                               Path(aug_filename).stem, output_dir):
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
    
    def _create_summary(self, aug_result: Dict[str, Any], source_files: List[str], target_split: str) -> Dict[str, Any]:
        """📊 Create detailed summary dengan variance info"""
        total_generated = aug_result.get('total_generated', 0)
        processed_files = aug_result.get('processed_files', 0)
        
        # Calculate rates
        success_rate = (total_generated / (processed_files * self.num_variations)) * 100 if processed_files > 0 else 0
        processing_rate = (processed_files / len(source_files)) * 100 if source_files else 0
        
        # Enhanced summary dengan variance info
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
                    'intensity_applied': self.intensity,
                    'variance_pattern': f"aug_*_{{uuid}}_{{variance:03d}}.jpg"  # NEW: Variance pattern info
                },
                'configuration': {
                    'types': ', '.join(self.types),
                    'target_split': target_split,
                    'target_count': self.target_count,
                    'naming_manager': 'FileNamingManager integrated'  # NEW: Naming info
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
        """📋 Log detailed summary dengan variance info"""
        input_info = summary['input']
        output_info = summary['output']
        config_info = summary['configuration']
        
        self.logger.success(f"✅ Augmentasi berhasil!")
        self.logger.info(f"📊 Input: {input_info['processed_files']}/{input_info['source_files']} files ({input_info['processing_rate']})")
        self.logger.info(f"🎯 Output: {output_info['total_generated']} generated ({output_info['success_rate']} success rate)")
        self.logger.info(f"⚙️ Config: {config_info['types']} @ intensity {output_info['intensity_applied']}")
        self.logger.info(f"🔢 Variance: {output_info['target_variations']} variations per file")
        self.logger.info(f"📁 Pattern: {output_info['variance_pattern']}")
    
    def _setup_augmented_output_directory(self, target_split: str) -> Path:
        """Setup directory untuk augmented output"""
        base_dir = self.config.get('data', {}).get('dir', 'data')
        output_dir = Path(base_dir) / 'augmented' / target_split
        
        # Create directories with auto-creation
        (output_dir / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        return output_dir
    
    def _create_pipeline(self):
        """Create augmentation pipeline dengan recommended parameters"""
        aug_type = self.types[0] if self.types else 'combined'
        
        # Use direct config parameters instead of intensity-based
        if aug_type == 'combined':
            return self.pipeline_factory.create_custom_pipeline(self.aug_config)
        else:
            return self.pipeline_factory.create_pipeline(aug_type, self.intensity)
    
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