"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Fixed service dengan proper error handling dan one-liner style yang benar
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

from .utils.core import (
    create_context, process_batch, ensure_dirs, read_image, save_image, 
    read_yolo_labels, save_yolo_labels, get_stem, safe_execute, safe_copy_file,
    find_aug_files, count_dataset_files
)
from .core.pipeline import PipelineFactory

class AugmentationService:
    """One-liner style orchestrator service dengan consolidated utilities dan fixed error handling"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        self.context = create_context(config, ui_components.get('comm') if ui_components else None)
        self.config, self.progress, self.paths = self.context['config'], self.context['progress'], self.context['paths']
        self.stats = defaultdict(int)
        self.pipeline_factory = PipelineFactory(config, self.progress.logger)
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Fixed full pipeline dengan proper error handling dan return statements"""
        start_time = time.time()
        
        # One-liner dataset validation dengan proper error return
        dataset_info = self.context['detector']()
        if dataset_info['status'] == 'error' or dataset_info['total_images'] == 0:
            return self._error_result(f"Dataset tidak valid: {dataset_info.get('message', 'Tidak ada gambar')}")
        
        self.progress.log_info(f"🚀 Pipeline dimulai: {dataset_info['total_images']} gambar")
        
        try:
            # Fixed augmentation step dengan proper error checking
            self.progress.progress("overall", 10, "Memulai augmentasi dataset")
            aug_result = self._process_augmentation(dataset_info)
            if aug_result['status'] != 'success':
                return self._error_result(f"Augmentasi gagal: {aug_result.get('message', 'Unknown error')}")
            
            # Fixed normalization step dengan proper error checking
            self.progress.progress("overall", 60, "Memulai normalisasi ke preprocessed")
            norm_result = self._process_normalization(target_split)
            if norm_result['status'] != 'success':
                return self._error_result(f"Normalisasi gagal: {norm_result.get('message', 'Unknown error')}")
            
            # One-liner success result dengan proper data access
            total_time = time.time() - start_time
            result = {
                'status': 'success', 
                'total_files': aug_result.get('total_generated', 0), 
                'final_output': target_split,
                'processing_time': total_time, 
                'steps': {'augmentation': aug_result, 'normalization': norm_result}
            }
            
            self.progress.log_success(f"✅ Pipeline selesai: {result['total_files']} file dalam {total_time:.1f}s")
            return result
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.progress.log_error(error_msg)
            return self._error_result(error_msg)
    
    def _process_augmentation(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fixed augmentation processing dengan proper error handling"""
        try:
            ensure_dirs(self.paths['aug_images'], self.paths['aug_labels'])
            
            # One-liner image files collection dengan better error handling
            image_files = [str(f) for location in dataset_info['image_locations'] 
                          for images_dir in [location['path'] if '/images' in location['path'] 
                                            else f"{location['path']}/images" if Path(f"{location['path']}/images").exists() 
                                            else location['path']]
                          for f in Path(images_dir).glob('*.*') 
                          if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()]
            
            if not image_files:
                return {'status': 'error', 'message': 'Tidak ada file gambar ditemukan'}
            
            # One-liner pipeline creation dengan error handling
            aug_type = self.config.get('types', ['combined'])[0] if self.config.get('types') else 'combined'
            pipeline = self.pipeline_factory.create_pipeline(aug_type, self.config.get('intensity', 0.7))
            
            # One-liner batch processing
            results = process_batch(image_files, 
                                   lambda img_path: self._process_single_image(img_path, pipeline, self.config.get('num_variations', 2)), 
                                   progress_tracker=self.progress)
            
            # One-liner stats calculation dengan safety checks
            successful = [r for r in results if r.get('status') == 'success']
            total_generated = sum(r.get('generated', 0) for r in successful)
            
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
    
    def _process_single_image(self, image_path: str, pipeline, num_variations: int) -> Dict[str, Any]:
        """One-liner single image processing dengan consolidated pattern"""
        try:
            # One-liner image reading
            image = read_image(image_path)
            if image is None: 
                return {'status': 'error', 'error': 'Cannot read image'}
            
            # One-liner label finding
            img_stem, img_dir = get_stem(image_path), Path(image_path).parent
            label_paths = [img_dir.parent / 'labels' / f'{img_stem}.txt', img_dir.parent / f'{img_stem}.txt', img_dir / f'{img_stem}.txt']
            
            # One-liner bbox extraction
            bboxes, class_labels = next(((lambda data: ([bbox[1:5] for bbox in data], [bbox[0] for bbox in data]))(read_yolo_labels(str(lp))) 
                                        for lp in label_paths if lp.exists() and (data := read_yolo_labels(str(lp)))), ([], []))
            
            # One-liner variants generation
            generated = sum(1 for var_idx in range(num_variations) 
                           if self._generate_variant(image, bboxes, class_labels, pipeline, img_stem, var_idx))
            
            return {'status': 'success', 'generated': generated}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_variant(self, image, bboxes: List, class_labels: List, pipeline, img_stem: str, var_idx: int) -> bool:
        """One-liner variant generation"""
        try:
            # One-liner augmentation
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image, aug_bboxes, aug_labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
            
            # One-liner file saving
            aug_filename = f"aug_{img_stem}_v{var_idx}"
            img_saved = save_image(aug_image, f"{self.paths['aug_images']}/{aug_filename}.jpg")
            
            # One-liner label saving
            if img_saved and aug_bboxes and aug_labels:
                full_bboxes = [[label] + bbox for bbox, label in zip(aug_bboxes, aug_labels)]
                save_yolo_labels(full_bboxes, f"{self.paths['aug_labels']}/{aug_filename}.txt")
            
            return img_saved
        except Exception:
            return False
    
    def _process_normalization(self, target_split: str) -> Dict[str, Any]:
        """Fixed normalization processing dengan proper error handling"""
        try:
            # One-liner target setup
            target_dir = f"{self.paths['prep_dir']}/{target_split}"
            ensure_dirs(f"{target_dir}/images", f"{target_dir}/labels")
            
            # One-liner augmented files finding
            aug_files = find_aug_files(self.paths['aug_dir'])
            if not aug_files: 
                return {'status': 'error', 'message': 'Tidak ada file augmented ditemukan'}
            
            # One-liner pairs creation
            image_files = [f for f in aug_files if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
            pairs = [(img_file, f"{self.paths['aug_labels']}/{get_stem(img_file)}.txt") 
                    for img_file in image_files if Path(f"{self.paths['aug_labels']}/{get_stem(img_file)}.txt").exists()]
            
            if not pairs:
                return {'status': 'error', 'message': 'Tidak ada pasangan image-label yang valid'}
            
            # One-liner normalization processing
            normalized = sum(1 for img_file, label_file in pairs 
                            if self._normalize_pair(img_file, label_file, target_dir))
            
            if normalized == 0:
                return {'status': 'error', 'message': 'Tidak ada file yang berhasil dinormalisasi'}
            
            return {'status': 'success', 'total_normalized': normalized, 'target_dir': target_dir}
            
        except Exception as e:
            return {'status': 'error', 'message': f"Error pada normalisasi: {str(e)}"}
    
    def _normalize_pair(self, img_file: str, label_file: str, target_dir: str) -> bool:
        """One-liner pair normalization"""
        try:
            # One-liner filename normalization (remove aug_ prefix)
            stem = get_stem(img_file)
            norm_stem = stem[4:] if stem.startswith('aug_') else stem
            
            # One-liner file copying
            return (safe_copy_file(img_file, f"{target_dir}/images/{norm_stem}.jpg") and 
                   safe_copy_file(label_file, f"{target_dir}/labels/{norm_stem}.txt"))
        except Exception:
            return False
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """One-liner cleanup menggunakan consolidated cleaner"""
        return self.context['cleaner']()
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """One-liner status menggunakan consolidated detector"""
        try:
            raw_images, raw_labels = count_dataset_files(self.paths['raw_dir'])
            aug_images, aug_labels = count_dataset_files(self.paths['aug_dir'])
            
            return {
                'raw_dataset': {'exists': raw_images > 0, 'total_images': raw_images, 'total_labels': raw_labels},
                'augmented_dataset': {'exists': aug_images > 0, 'total_images': aug_images, 'total_labels': aug_labels},
                'ready_for_augmentation': raw_images > 0
            }
        except Exception as e:
            return {
                'raw_dataset': {'exists': False, 'total_images': 0, 'total_labels': 0},
                'augmented_dataset': {'exists': False, 'total_images': 0, 'total_labels': 0},
                'ready_for_augmentation': False,
                'error': str(e)
            }
    
    # One-liner error result creator
    _error_result = lambda self, msg: {'status': 'error', 'message': msg, 'timestamp': time.time()}

# One-liner factory functions
create_augmentation_service = lambda config, ui_components=None: AugmentationService(config, ui_components)

def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    """One-liner service creation dari UI components"""
    from .config import extract_ui_config
    config = extract_ui_config(ui_components)
    return AugmentationService(config, ui_components)