"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Fixed service dengan real-time progress integration dan proper communicator flow
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

from .utils.core import create_context, process_batch, ensure_dirs, read_image, save_image, read_yolo_labels, save_yolo_labels, get_stem, safe_execute, safe_copy_file, find_aug_files, count_dataset_files
from .core.pipeline import PipelineFactory
from .communicator import create_communicator

class AugmentationService:
    """Fixed orchestrator service dengan real-time progress integration dan proper communicator flow"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        # Create communicator dengan UI components
        self.comm = create_communicator(ui_components) if ui_components else None
        
        # Create context dengan communicator integration
        self.context = create_context(config, self.comm)
        self.config, self.progress, self.paths = self.context['config'], self.context['progress'], self.context['paths']
        self.stats = defaultdict(int)
        self.pipeline_factory = PipelineFactory(config, self.progress.logger)
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Fixed full pipeline dengan real-time progress updates dan proper callback integration"""
        start_time = time.time()
        
        # Start operation tracking
        self.comm and self.comm.start_operation("Augmentation Pipeline", 100)
        
        # Dataset validation dengan progress
        self.progress.progress("overall", 5, 100, "Validasi dataset")
        dataset_info = self.context['detector']()
        if dataset_info['status'] == 'error' or dataset_info['total_images'] == 0:
            return self._error_result(f"Dataset tidak valid: {dataset_info.get('message', 'Tidak ada gambar')}")
        
        self.progress.log_info(f"🚀 Pipeline dimulai: {dataset_info['total_images']} gambar")
        
        try:
            # Augmentation step dengan real-time progress
            self.progress.progress("overall", 10, 100, "Memulai augmentasi dataset")
            aug_result = self._process_augmentation_with_progress(dataset_info, progress_callback)
            if aug_result['status'] != 'success':
                return self._error_result(f"Augmentasi gagal: {aug_result.get('message', 'Unknown error')}")
            
            # Normalization step dengan real-time progress
            self.progress.progress("overall", 60, 100, "Memulai normalisasi ke preprocessed")
            norm_result = self._process_normalization_with_progress(target_split, progress_callback)
            if norm_result['status'] != 'success':
                return self._error_result(f"Normalisasi gagal: {norm_result.get('message', 'Unknown error')}")
            
            # Success result dengan proper timing
            total_time = time.time() - start_time
            result = {
                'status': 'success', 
                'total_files': aug_result.get('total_generated', 0), 
                'final_output': target_split,
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
        """Fixed augmentation processing dengan detailed progress updates"""
        try:
            ensure_dirs(self.paths['aug_images'], self.paths['aug_labels'])
            
            # Enhanced progress tracking
            self.progress.progress("step", 5, 100, "Mencari file gambar")
            
            # Image files collection dengan better error handling
            image_files = [str(f) for location in dataset_info['image_locations'] 
                          for images_dir in [location['path'] if '/images' in location['path'] 
                                            else f"{location['path']}/images" if Path(f"{location['path']}/images").exists() 
                                            else location['path']]
                          for f in Path(images_dir).glob('*.*') 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()]
            
            if not image_files:
                return {'status': 'error', 'message': 'Tidak ada file gambar ditemukan'}
            
            self.progress.progress("step", 15, 100, f"Ditemukan {len(image_files)} file gambar")
            
            # Pipeline creation dengan progress
            aug_type = self.config.get('types', ['combined'])[0] if self.config.get('types') else 'combined'
            pipeline = self.pipeline_factory.create_pipeline(aug_type, self.config.get('intensity', 0.7))
            
            self.progress.progress("step", 20, 100, "Pipeline augmentasi siap")
            
            # Enhanced batch processing dengan real-time callback
            def enhanced_process_func(img_path):
                return self._process_single_image(img_path, pipeline, self.config.get('num_variations', 2))
            
            # Custom progress callback untuk batch processing
            def batch_progress_callback(step: str, current: int, total: int, message: str):
                # Map to step progress (20-90%)
                step_progress = 20 + int((current / max(1, total)) * 70)
                self.progress.progress("step", step_progress, 100, message)
                
                # Also call external callback
                if progress_callback:
                    overall_progress = 10 + int((current / max(1, total)) * 50)  # 10-60% of overall
                    progress_callback("overall", overall_progress, 100, f"Augmentasi: {message}")
            
            # Process dengan enhanced progress tracking
            results = process_batch(
                image_files, 
                enhanced_process_func, 
                progress_tracker=self.progress,
                operation_name="augmentasi gambar"
            )
            
            # Statistics calculation dengan safety checks
            successful = [r for r in results if r.get('status') == 'success']
            total_generated = sum(r.get('generated', 0) for r in successful)
            
            self.progress.progress("step", 100, 100, f"Augmentasi selesai: {total_generated} file")
            
            if total_generated == 0:
                return {'status': 'error', 'message': 'Tidak ada file yang berhasil diaugmentasi'}
            
            return {
                'status': 'success', 
                'total_generated': total_generated,
                'processed_files': len(results), 
                'success_rate': len(successful) / len(results) * 100 if results else 0
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Error pada augmentasi: {str(e)}"}
    
    def _process_normalization_with_progress(self, target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Fixed normalization processing dengan detailed progress updates"""
        try:
            # Target setup dengan progress
            target_dir = f"{self.paths['prep_dir']}/{target_split}"
            ensure_dirs(f"{target_dir}/images", f"{target_dir}/labels")
            
            self.progress.progress("step", 10, 100, "Setup direktori target")
            
            # Find augmented files dengan progress
            aug_files = find_aug_files(self.paths['aug_dir'])
            if not aug_files: 
                return {'status': 'error', 'message': 'Tidak ada file augmented ditemukan'}
            
            self.progress.progress("step", 20, 100, f"Ditemukan {len(aug_files)} file augmented")
            
            # Create pairs dengan progress tracking
            image_files = [f for f in aug_files if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
            pairs = [(img_file, f"{self.paths['aug_labels']}/{get_stem(img_file)}.txt") 
                    for img_file in image_files if Path(f"{self.paths['aug_labels']}/{get_stem(img_file)}.txt").exists()]
            
            if not pairs:
                return {'status': 'error', 'message': 'Tidak ada pasangan image-label yang valid'}
            
            self.progress.progress("step", 30, 100, f"Validasi {len(pairs)} pasangan file")
            
            # Process normalization dengan real-time updates
            normalized = 0
            total_pairs = len(pairs)
            
            for i, (img_file, label_file) in enumerate(pairs):
                if self._normalize_pair(img_file, label_file, target_dir):
                    normalized += 1
                
                # Real-time progress update
                current_progress = 30 + int((i / total_pairs) * 60)  # 30-90%
                self.progress.progress("step", current_progress, 100, 
                                     f"Normalisasi: {i+1}/{total_pairs}")
                
                # External callback
                if progress_callback:
                    overall_progress = 60 + int((i / total_pairs) * 30)  # 60-90% of overall
                    progress_callback("overall", overall_progress, 100, 
                                    f"Normalisasi: {i+1}/{total_pairs}")
            
            self.progress.progress("step", 100, 100, f"Normalisasi selesai: {normalized} file")
            
            if normalized == 0:
                return {'status': 'error', 'message': 'Tidak ada file yang berhasil dinormalisasi'}
            
            return {'status': 'success', 'total_normalized': normalized, 'target_dir': target_dir}
            
        except Exception as e:
            return {'status': 'error', 'message': f"Error pada normalisasi: {str(e)}"}
    
    def _process_single_image(self, image_path: str, pipeline, num_variations: int) -> Dict[str, Any]:
        """Single image processing - unchanged logic"""
        try:
            # Read image
            image = read_image(image_path)
            if image is None: 
                return {'status': 'error', 'error': 'Cannot read image'}
            
            # Find label
            img_stem, img_dir = get_stem(image_path), Path(image_path).parent
            label_paths = [img_dir.parent / 'labels' / f'{img_stem}.txt', img_dir.parent / f'{img_stem}.txt', img_dir / f'{img_stem}.txt']
            
            # Extract bboxes
            bboxes, class_labels = next(((lambda data: ([bbox[1:5] for bbox in data], [bbox[0] for bbox in data]))(read_yolo_labels(str(lp))) 
                                        for lp in label_paths if lp.exists() and (data := read_yolo_labels(str(lp)))), ([], []))
            
            # Generate variants
            generated = sum(1 for var_idx in range(num_variations) 
                           if self._generate_variant(image, bboxes, class_labels, pipeline, img_stem, var_idx))
            
            return {'status': 'success', 'generated': generated}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_variant(self, image, bboxes: List, class_labels: List, pipeline, img_stem: str, var_idx: int) -> bool:
        """Generate single variant - unchanged logic"""
        try:
            # Apply augmentation
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image, aug_bboxes, aug_labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
            
            # Save files
            aug_filename = f"aug_{img_stem}_v{var_idx}"
            img_saved = save_image(aug_image, f"{self.paths['aug_images']}/{aug_filename}.jpg")
            
            # Save labels
            if img_saved and aug_bboxes and aug_labels:
                full_bboxes = [[label] + bbox for bbox, label in zip(aug_bboxes, aug_labels)]
                save_yolo_labels(full_bboxes, f"{self.paths['aug_labels']}/{aug_filename}.txt")
            
            return img_saved
        except Exception:
            return False
    
    def _normalize_pair(self, img_file: str, label_file: str, target_dir: str) -> bool:
        """Normalize single pair - unchanged logic"""
        try:
            # Generate normalized filename (remove aug_ prefix)
            stem = get_stem(img_file)
            norm_stem = stem[4:] if stem.startswith('aug_') else stem
            
            # Copy files
            return (safe_copy_file(img_file, f"{target_dir}/images/{norm_stem}.jpg") and 
                   safe_copy_file(label_file, f"{target_dir}/labels/{norm_stem}.txt"))
        except Exception:
            return False
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Enhanced cleanup dengan real-time progress"""
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
        """Get status dengan enhanced progress tracking"""
        try:
            self.comm and self.comm.start_operation("Check Dataset Status", 100)
            
            raw_images, raw_labels = count_dataset_files(self.paths['raw_dir'])
            aug_images, aug_labels = count_dataset_files(self.paths['aug_dir'])
            
            self.comm and self.comm.progress("overall", 50, 100, "Menghitung file preprocessed")
            
            # Check preprocessed files
            prep_images = prep_labels = 0
            if Path(self.paths['prep_dir']).exists():
                for split in ['train', 'valid', 'test']:
                    split_path = f"{self.paths['prep_dir']}/{split}"
                    if Path(split_path).exists():
                        split_images, split_labels = count_dataset_files(split_path)
                        prep_images += split_images
                        prep_labels += split_labels
            
            result = {
                'raw_dataset': {'exists': raw_images > 0, 'total_images': raw_images, 'total_labels': raw_labels},
                'augmented_dataset': {'exists': aug_images > 0, 'total_images': aug_images, 'total_labels': aug_labels},
                'preprocessed_dataset': {'exists': prep_images > 0, 'total_files': prep_images + prep_labels},
                'ready_for_augmentation': raw_images > 0
            }
            
            self.comm and self.comm.complete_operation("Check Dataset Status", "Status check selesai")
            return result
            
        except Exception as e:
            error_msg = f"Status check error: {str(e)}"
            self.comm and self.comm.error_operation("Check Dataset Status", error_msg)
            return {
                'raw_dataset': {'exists': False, 'total_images': 0, 'total_labels': 0},
                'augmented_dataset': {'exists': False, 'total_images': 0, 'total_labels': 0},
                'preprocessed_dataset': {'exists': False, 'total_files': 0},
                'ready_for_augmentation': False,
                'error': error_msg
            }
    
    # Error result creator
    _error_result = lambda self, msg: {'status': 'error', 'message': msg, 'timestamp': time.time()}

# Factory functions
create_augmentation_service = lambda config, ui_components=None: AugmentationService(config, ui_components)

def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    """Create service dari UI components dengan proper communicator integration"""
    from .config import extract_ui_config
    config = extract_ui_config(ui_components)
    return AugmentationService(config, ui_components)