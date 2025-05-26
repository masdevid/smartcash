"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Refactored service menggunakan core utilities untuk mengeliminasi duplikasi
"""

import time
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

from .utils.core import (
    create_context, process_batch, ensure_dirs, read_image, save_image, 
    read_yolo_labels, save_yolo_labels, get_stem, safe_execute
)
from .core.pipeline import PipelineFactory

class AugmentationService:
    """Main orchestrator service dengan consolidated utilities"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        self.context = create_context(config, ui_components.get('comm') if ui_components else None)
        self.config = self.context['config']
        self.progress = self.context['progress']
        self.paths = self.context['paths']
        self.stats = defaultdict(int)
        self.pipeline_factory = PipelineFactory(config, self.progress.logger)
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Full pipeline dengan consolidated flow"""
        start_time = time.time()
        
        # Validate dataset
        dataset_info = self.context['detector']()
        if dataset_info['status'] == 'error' or dataset_info['total_images'] == 0:
            return self._error_result(f"Dataset tidak valid: {dataset_info.get('message', 'Tidak ada gambar')}")
        
        self.progress.log_info(f"🚀 Pipeline dimulai: {dataset_info['total_images']} gambar")
        
        try:
            # Step 1: Augmentasi
            self.progress.progress("overall", 10, "Memulai augmentasi dataset")
            aug_result = self._process_augmentation(dataset_info)
            
            if aug_result['status'] != 'success':
                return self._error_result(f"Augmentasi gagal: {aug_result['message']}")
            
            # Step 2: Normalisasi
            self.progress.progress("overall", 60, "Memulai normalisasi ke preprocessed")
            norm_result = self._process_normalization(target_split)
            
            if norm_result['status'] != 'success':
                return self._error_result(f"Normalisasi gagal: {norm_result['message']}")
            
            # Success summary
            total_time = time.time() - start_time
            result = {
                'status': 'success',
                'total_files': aug_result['total_generated'],
                'final_output': f"{target_split}",
                'processing_time': total_time,
                'steps': {'augmentation': aug_result, 'normalization': norm_result}
            }
            
            self.progress.log_success(f"Pipeline selesai: {result['total_files']} file dalam {total_time:.1f}s")
            return result
            
        except Exception as e:
            return self._error_result(f"Pipeline error: {str(e)}")
    
    def _process_augmentation(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process augmentation dengan consolidated utilities"""
        ensure_dirs(self.paths['aug_images'], self.paths['aug_labels'])
        
        # Get image files
        image_files = []
        for location in dataset_info['image_locations']:
            images_dir = location['path']
            if '/images' not in images_dir:
                images_dir = f"{images_dir}/images" if Path(f"{images_dir}/images").exists() else images_dir
            
            image_files.extend([str(f) for f in Path(images_dir).glob('*.*') 
                              if f.suffix.lower() in ['.jpg', '.jpeg', '.png'] and f.is_file()])
        
        if not image_files:
            return {'status': 'error', 'message': 'Tidak ada file gambar ditemukan'}
        
        # Create pipeline
        aug_config = self.config
        aug_type = aug_config['types'][0] if aug_config['types'] else 'combined'
        pipeline = self.pipeline_factory.create_pipeline(aug_type, aug_config['intensity'])
        
        # Process files
        process_func = lambda img_path: self._process_single_image(img_path, pipeline, aug_config['num_variations'])
        results = process_batch(image_files, process_func, progress_tracker=self.progress)
        
        # Calculate stats
        successful = [r for r in results if r.get('status') == 'success']
        total_generated = sum(r.get('generated', 0) for r in successful)
        
        return {
            'status': 'success',
            'total_generated': total_generated,
            'processed_files': len(results),
            'success_rate': len(successful) / len(results) * 100 if results else 0
        }
    
    def _process_single_image(self, image_path: str, pipeline, num_variations: int) -> Dict[str, Any]:
        """Process single image dengan consolidated pattern"""
        try:
            image = read_image(image_path)
            if image is None:
                return {'status': 'error', 'error': 'Cannot read image'}
            
            # Find label file
            img_stem = get_stem(image_path)
            img_dir = Path(image_path).parent
            label_paths = [
                img_dir.parent / 'labels' / f'{img_stem}.txt',
                img_dir.parent / f'{img_stem}.txt',
                img_dir / f'{img_stem}.txt'
            ]
            
            bboxes, class_labels = [], []
            for label_path in label_paths:
                if label_path.exists():
                    label_data = read_yolo_labels(str(label_path))
                    if label_data:
                        bboxes = [bbox[1:5] for bbox in label_data]
                        class_labels = [bbox[0] for bbox in label_data]
                    break
            
            generated = 0
            for var_idx in range(num_variations):
                try:
                    # Apply augmentation
                    augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_image = augmented['image']
                    aug_bboxes = augmented['bboxes']
                    aug_labels = augmented['class_labels']
                    
                    # Save files
                    aug_filename = f"aug_{img_stem}_v{var_idx}"
                    img_path = f"{self.paths['aug_images']}/{aug_filename}.jpg"
                    label_path = f"{self.paths['aug_labels']}/{aug_filename}.txt"
                    
                    if save_image(aug_image, img_path):
                        if aug_bboxes and aug_labels:
                            # Reconstruct full bbox format
                            full_bboxes = [[label] + bbox for bbox, label in zip(aug_bboxes, aug_labels)]
                            save_yolo_labels(full_bboxes, label_path)
                        generated += 1
                        
                except Exception:
                    continue
            
            return {'status': 'success', 'generated': generated}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _process_normalization(self, target_split: str) -> Dict[str, Any]:
        """Process normalization dengan consolidated utilities"""
        from .utils.core import find_aug_files
        
        # Setup target directories
        target_dir = f"{self.paths['prep_dir']}/{target_split}"
        ensure_dirs(f"{target_dir}/images", f"{target_dir}/labels")
        
        # Find augmented files
        aug_files = find_aug_files(self.paths['aug_dir'])
        if not aug_files:
            return {'status': 'error', 'message': 'Tidak ada file augmented ditemukan'}
        
        # Group by image/label pairs
        pairs = []
        image_files = [f for f in aug_files if any(f.endswith(ext) for ext in ['.jpg', '.jpeg', '.png'])]
        
        for img_file in image_files:
            img_stem = get_stem(img_file)
            label_file = f"{self.paths['aug_labels']}/{img_stem}.txt"
            if Path(label_file).exists():
                pairs.append((img_file, label_file))
        
        # Process normalization
        normalized = 0
        for img_file, label_file in pairs:
            try:
                # Generate normalized filename (remove aug_ prefix)
                stem = get_stem(img_file)
                norm_stem = stem[4:] if stem.startswith('aug_') else stem
                
                # Copy files
                target_img = f"{target_dir}/images/{norm_stem}.jpg"
                target_label = f"{target_dir}/labels/{norm_stem}.txt"
                
                if safe_copy_file(img_file, target_img) and safe_copy_file(label_file, target_label):
                    normalized += 1
                    
            except Exception:
                continue
        
        return {
            'status': 'success',
            'total_normalized': normalized,
            'target_dir': target_dir
        }
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Cleanup menggunakan consolidated cleaner"""
        prep_dir = self.paths['prep_dir'] if include_preprocessed else None
        return self.context['cleaner']()
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Get status menggunakan consolidated detector"""
        from .utils.core import count_dataset_files
        
        raw_images, raw_labels = count_dataset_files(self.paths['raw_dir'])
        aug_images, aug_labels = count_dataset_files(self.paths['aug_dir'])
        
        return {
            'raw_dataset': {
                'exists': raw_images > 0,
                'total_images': raw_images,
                'total_labels': raw_labels
            },
            'augmented_dataset': {
                'exists': aug_images > 0,
                'total_images': aug_images,
                'total_labels': aug_labels
            },
            'ready_for_augmentation': raw_images > 0
        }
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """One-liner error result"""
        return {'status': 'error', 'message': message, 'timestamp': time.time()}

# Factory functions
def create_augmentation_service(config: Dict[str, Any], ui_components: Dict[str, Any] = None) -> AugmentationService:
    return AugmentationService(config, ui_components)

def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    from .utils.core import extract_config
    config = extract_config(ui_components.get('config', {}))
    return AugmentationService({'augmentation': config, 'data': {'dir': config['raw_dir']}}, ui_components)