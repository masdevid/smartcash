"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Fixed service dengan aggressive log suppression dan parameter alignment
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

from .utils.core import create_context, process_batch, ensure_dirs, read_image, save_image, read_yolo_labels, save_yolo_labels, get_stem, safe_execute, safe_copy_file, find_aug_files, count_dataset_files
from .core.pipeline import PipelineFactory
from .communicator import create_communicator

class AugmentationService:
    """Fixed service dengan aggressive log suppression dan parameter alignment"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        # Setup aggressive log suppression FIRST
        self._setup_aggressive_log_suppression()
        
        # Create communicator dengan UI components
        self.comm = create_communicator(ui_components) if ui_components else None
        
        # Create context dengan aligned parameters
        self.context = create_context(self._align_config_parameters(config), self.comm)
        self.config, self.progress, self.paths = self.context['config'], self.context['progress'], self.context['paths']
        self.stats = defaultdict(int)
        self.pipeline_factory = PipelineFactory(config, self.progress.logger if self.progress else None)
    
    def _setup_aggressive_log_suppression(self):
        """Aggressive log suppression untuk prevent console flooding"""
        suppression_targets = [
            'smartcash.dataset.augmentor', 'smartcash.dataset.augmentor.core',
            'smartcash.dataset.augmentor.utils', 'smartcash.dataset.augmentor.strategies',
            'smartcash.common.threadpools', 'concurrent.futures', 'threading',
            'albumentations', 'cv2', 'numpy', 'PIL', 'matplotlib', 'tqdm',
            'requests', 'urllib3', 'http.client', 'ipywidgets'
        ]
        
        for target in suppression_targets:
            try:
                logger = logging.getLogger(target)
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
                # Clear existing handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
            except Exception:
                pass
        
        # Suppress root logger aggressively
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.CRITICAL)
    
    def _align_config_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Align parameter names between UI dan service"""
        augmentation_config = config.get('augmentation', {})
        
        # Parameter alignment mapping
        aligned_config = {
            'data': {'dir': config.get('data', {}).get('dir', 'data')},
            'augmentation': {
                'num_variations': augmentation_config.get('num_variations', 2),
                'target_count': augmentation_config.get('target_count', 500),
                'types': augmentation_config.get('types', ['combined']),
                'intensity': augmentation_config.get('intensity', 0.7),
                'output_dir': augmentation_config.get('output_dir', 'data/augmented'),
                # UI parameter mapping
                'fliplr': augmentation_config.get('fliplr', 0.5),
                'degrees': augmentation_config.get('degrees', 10),
                'translate': augmentation_config.get('translate', 0.1),
                'scale': augmentation_config.get('scale', 0.1),
                'hsv_h': augmentation_config.get('hsv_h', 0.015),
                'hsv_s': augmentation_config.get('hsv_s', 0.7),
                'brightness': augmentation_config.get('brightness', 0.2),
                'contrast': augmentation_config.get('contrast', 0.2),
                'target_split': augmentation_config.get('target_split', 'train')
            },
            'preprocessing': {
                'output_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            }
        }
        
        return aligned_config
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Fixed pipeline dengan aligned parameters"""
        start_time = time.time()
        
        # Start operation dengan suppressed logging
        self.comm and self.comm.start_operation("Augmentation Pipeline", 100)
        
        try:
            # Use aligned target_split from config
            actual_target_split = self.config.get('target_split', target_split)
            
            # Dataset validation
            self._update_progress_safe("overall", 5, "Validasi dataset")
            dataset_info = self.context['detector']()
            if dataset_info['status'] == 'error' or dataset_info['total_images'] == 0:
                return self._error_result(f"Dataset tidak valid: {dataset_info.get('message', 'Tidak ada gambar')}")
            
            self._log_safe("info", f"🚀 Pipeline dimulai: {dataset_info['total_images']} gambar")
            
            # Augmentation step
            self._update_progress_safe("overall", 10, "Memulai augmentasi dataset")
            aug_result = self._process_augmentation_with_progress(dataset_info, progress_callback)
            if aug_result['status'] != 'success':
                return self._error_result(f"Augmentasi gagal: {aug_result.get('message', 'Unknown error')}")
            
            # Normalization step
            self._update_progress_safe("overall", 60, "Memulai normalisasi ke preprocessed")
            norm_result = self._process_normalization_with_progress(actual_target_split, progress_callback)
            if norm_result['status'] != 'success':
                return self._error_result(f"Normalisasi gagal: {norm_result.get('message', 'Unknown error')}")
            
            # Success result
            total_time = time.time() - start_time
            result = {
                'status': 'success', 
                'total_files': aug_result.get('total_generated', 0), 
                'final_output': actual_target_split,
                'processing_time': total_time, 
                'steps': {'augmentation': aug_result, 'normalization': norm_result}
            }
            
            # Complete operation
            self.comm and self.comm.complete_operation("Augmentation Pipeline", 
                f"Pipeline selesai: {result['total_files']} file dalam {total_time:.1f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.comm and self.comm.error_operation("Augmentation Pipeline", error_msg)
            return self._error_result(error_msg)
    
    def _process_augmentation_with_progress(self, dataset_info: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Fixed augmentation dengan suppressed progress"""
        try:
            ensure_dirs(self.paths['aug_images'], self.paths['aug_labels'])
            
            # Get image files
            image_files = [str(f) for location in dataset_info['image_locations'] 
                          for images_dir in [location['path'] if '/images' in location['path'] 
                                            else f"{location['path']}/images" if Path(f"{location['path']}/images").exists() 
                                            else location['path']]
                          for f in Path(images_dir).glob('*.*') 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()]
            
            if not image_files:
                return {'status': 'error', 'message': 'Tidak ada file gambar ditemukan'}
            
            self._update_progress_safe("step", 15, f"Ditemukan {len(image_files)} file gambar")
            
            # Create pipeline dengan aligned parameters
            aug_type = self.config.get('types', ['combined'])[0] if self.config.get('types') else 'combined'
            pipeline = self.pipeline_factory.create_pipeline(aug_type, self.config.get('intensity', 0.7))
            
            # Process dengan suppressed logging
            def process_func(img_path):
                return self._process_single_image(img_path, pipeline, self.config.get('num_variations', 2))
            
            # Batch processing dengan minimal progress updates
            results = process_batch(
                image_files, 
                process_func, 
                progress_tracker=self.progress,
                operation_name="augmentasi gambar"
            )
            
            # Calculate statistics
            successful = [r for r in results if r.get('status') == 'success']
            total_generated = sum(r.get('generated', 0) for r in successful)
            
            self._update_progress_safe("step", 100, f"Augmentasi selesai: {total_generated} file")
            
            return {
                'status': 'success', 
                'total_generated': total_generated,
                'processed_files': len(results), 
                'success_rate': len(successful) / len(results) * 100 if results else 0
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Error pada augmentasi: {str(e)}"}
    
    def _process_normalization_with_progress(self, target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Fixed normalization dengan minimal logging"""
        try:
            target_dir = f"{self.paths['prep_dir']}/{target_split}"
            ensure_dirs(f"{target_dir}/images", f"{target_dir}/labels")
            
            aug_files = find_aug_files(self.paths['aug_dir'])
            if not aug_files: 
                return {'status': 'error', 'message': 'Tidak ada file augmented ditemukan'}
            
            # Create pairs
            image_files = [f for f in aug_files if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
            pairs = [(img_file, f"{self.paths['aug_labels']}/{get_stem(img_file)}.txt") 
                    for img_file in image_files if Path(f"{self.paths['aug_labels']}/{get_stem(img_file)}.txt").exists()]
            
            if not pairs:
                return {'status': 'error', 'message': 'Tidak ada pasangan image-label yang valid'}
            
            # Process normalization dengan minimal progress
            normalized = sum(1 for img_file, label_file in pairs if self._normalize_pair(img_file, label_file, target_dir))
            
            self._update_progress_safe("step", 100, f"Normalisasi selesai: {normalized} file")
            
            return {'status': 'success', 'total_normalized': normalized, 'target_dir': target_dir}
            
        except Exception as e:
            return {'status': 'error', 'message': f"Error pada normalisasi: {str(e)}"}
    
    def _process_single_image(self, image_path: str, pipeline, num_variations: int) -> Dict[str, Any]:
        """Process single image dengan suppressed logging"""
        try:
            image = read_image(image_path)
            if image is None: 
                return {'status': 'error', 'error': 'Cannot read image'}
            
            # Find label
            img_stem, img_dir = get_stem(image_path), Path(image_path).parent
            label_paths = [img_dir.parent / 'labels' / f'{img_stem}.txt', img_dir.parent / f'{img_stem}.txt', img_dir / f'{img_stem}.txt']
            
            # Extract bboxes
            bboxes, class_labels = next(((lambda data: ([bbox[1:5] for bbox in data], [bbox[0] for bbox in data]))(read_yolo_labels(str(lp))) 
                                        for lp in label_paths if lp.exists() and (data := read_yolo_labels(str(lp)))), ([], []))
            
            # Generate variants dengan suppressed logging
            generated = sum(1 for var_idx in range(num_variations) 
                           if self._generate_variant(image, bboxes, class_labels, pipeline, img_stem, var_idx))
            
            return {'status': 'success', 'generated': generated}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_variant(self, image, bboxes: List, class_labels: List, pipeline, img_stem: str, var_idx: int) -> bool:
        """Generate variant dengan no logging"""
        try:
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image, aug_bboxes, aug_labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
            
            aug_filename = f"aug_{img_stem}_v{var_idx}"
            img_saved = save_image(aug_image, f"{self.paths['aug_images']}/{aug_filename}.jpg")
            
            if img_saved and aug_bboxes and aug_labels:
                full_bboxes = [[label] + bbox for bbox, label in zip(aug_bboxes, aug_labels)]
                save_yolo_labels(full_bboxes, f"{self.paths['aug_labels']}/{aug_filename}.txt")
            
            return img_saved
        except Exception:
            return False
    
    def _normalize_pair(self, img_file: str, label_file: str, target_dir: str) -> bool:
        """Normalize pair dengan no logging"""
        try:
            stem = get_stem(img_file)
            norm_stem = stem[4:] if stem.startswith('aug_') else stem
            
            return (safe_copy_file(img_file, f"{target_dir}/images/{norm_stem}.jpg") and 
                   safe_copy_file(label_file, f"{target_dir}/labels/{norm_stem}.txt"))
        except Exception:
            return False
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Cleanup dengan suppressed logging"""
        self.comm and self.comm.start_operation("Cleanup Dataset", 100)
        
        try:
            result = self.context['cleaner']()
            
            if result['status'] == 'success':
                self.comm and self.comm.complete_operation("Cleanup Dataset", result['message'])
            else:
                self.comm and self.comm.error_operation("Cleanup Dataset", result['message'])
            
            return result
        except Exception as e:
            error_msg = f"Cleanup error: {str(e)}"
            self.comm and self.comm.error_operation("Cleanup Dataset", error_msg)
            return {'status': 'error', 'message': error_msg}
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Get status dengan suppressed logging"""
        try:
            raw_images, raw_labels = count_dataset_files(self.paths['raw_dir'])
            aug_images, aug_labels = count_dataset_files(self.paths['aug_dir'])
            
            # Check preprocessed files
            prep_images = prep_labels = 0
            if Path(self.paths['prep_dir']).exists():
                for split in ['train', 'valid', 'test']:
                    split_path = f"{self.paths['prep_dir']}/{split}"
                    if Path(split_path).exists():
                        split_images, split_labels = count_dataset_files(split_path)
                        prep_images += split_images
                        prep_labels += split_labels
            
            return {
                'raw_dataset': {'exists': raw_images > 0, 'total_images': raw_images, 'total_labels': raw_labels},
                'augmented_dataset': {'exists': aug_images > 0, 'total_images': aug_images, 'total_labels': aug_labels},
                'preprocessed_dataset': {'exists': prep_images > 0, 'total_files': prep_images + prep_labels},
                'ready_for_augmentation': raw_images > 0
            }
            
        except Exception as e:
            return {
                'raw_dataset': {'exists': False, 'total_images': 0, 'total_labels': 0},
                'augmented_dataset': {'exists': False, 'total_images': 0, 'total_labels': 0},
                'preprocessed_dataset': {'exists': False, 'total_files': 0},
                'ready_for_augmentation': False,
                'error': str(e)
            }
    
    # Safe utility methods
    def _update_progress_safe(self, step: str, percentage: int, message: str) -> None:
        """Safe progress update"""
        try:
            self.comm and hasattr(self.comm, 'progress') and self.comm.progress(step, percentage, 100, message)
        except Exception:
            pass
    
    def _log_safe(self, level: str, message: str) -> None:
        """Safe logging hanya ke UI"""
        try:
            self.comm and hasattr(self.comm, f'log_{level}') and getattr(self.comm, f'log_{level}')(message)
        except Exception:
            pass
    
    # Error result creator
    _error_result = lambda self, msg: {'status': 'error', 'message': msg, 'timestamp': time.time()}

# Factory functions dengan parameter alignment
def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    """Create service dari UI components dengan aligned parameters"""
    from .config import extract_ui_config
    config = extract_ui_config(ui_components)
    return AugmentationService(config, ui_components)

# One-liner factories
create_augmentation_service = lambda config, ui_components=None: AugmentationService(config, ui_components)