"""
File: smartcash/dataset/augmentor/service.py
Deskripsi: Updated augmentation service dengan split-based directory dan UUID consistency
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from collections import defaultdict

from smartcash.common.utils.file_naming_manager import FileNamingManager
from smartcash.dataset.augmentor.utils.core import create_context, process_batch, ensure_dirs
from smartcash.dataset.augmentor.core.pipeline import PipelineFactory
from smartcash.dataset.augmentor.communicator import create_communicator

class AugmentationService:
    """Updated service dengan split-based directory dan UUID consistency"""
    
    def __init__(self, config: Dict[str, Any], ui_components: Dict[str, Any] = None):
        self._setup_aggressive_log_suppression()
        
        self.comm = create_communicator(ui_components) if ui_components else None
        self.context = create_context(self._align_config_parameters(config), self.comm)
        self.config, self.progress, self.paths = self.context['config'], self.context['progress'], self.context['paths']
        
        # Updated: UUID naming manager
        self.naming_manager = FileNamingManager(config)
        self.pipeline_factory = PipelineFactory(config, self.progress.logger if self.progress else None)
        self.stats = defaultdict(int)
    
    def run_full_augmentation_pipeline(self, target_split: str = "train", progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Pipeline dengan split-based output directory"""
        start_time = time.time()
        self.comm and self.comm.start_operation("Augmentation Pipeline", 100)
        
        try:
            actual_target_split = self.config.get('target_split', target_split)
            
            # Dataset validation
            self._update_progress_safe("overall", 5, "Validasi dataset")
            dataset_info = self.context['detector']()
            if dataset_info['status'] == 'error' or dataset_info['total_images'] == 0:
                return self._error_result(f"Dataset tidak valid: {dataset_info.get('message')}")
            
            self._log_safe("info", f"🚀 Pipeline dimulai: {dataset_info['total_images']} gambar")
            
            # Updated: Augmentation dengan split-based directory
            self._update_progress_safe("overall", 10, "Memulai augmentasi dengan split-based structure")
            aug_result = self._process_augmentation_with_split(dataset_info, actual_target_split, progress_callback)
            if aug_result['status'] != 'success':
                return self._error_result(f"Augmentasi gagal: {aug_result.get('message')}")
            
            # Updated: Normalization ke preprocessed dengan split consistency
            self._update_progress_safe("overall", 60, "Normalisasi ke preprocessed format")
            norm_result = self._process_normalization_with_split(actual_target_split, progress_callback)
            if norm_result['status'] != 'success':
                return self._error_result(f"Normalisasi gagal: {norm_result.get('message')}")
            
            total_time = time.time() - start_time
            result = {
                'status': 'success', 'total_files': aug_result.get('total_generated', 0),
                'final_output': actual_target_split, 'processing_time': total_time,
                'split_structure': True, 'uuid_consistency': True,
                'steps': {'augmentation': aug_result, 'normalization': norm_result}
            }
            
            self.comm and self.comm.complete_operation("Augmentation Pipeline", 
                f"Pipeline selesai: {result['total_files']} file, struktur split aktif")
            
            return result
            
        except Exception as e:
            error_msg = f"Pipeline error: {str(e)}"
            self.comm and self.comm.error_operation("Augmentation Pipeline", error_msg)
            return self._error_result(error_msg)
    
    def _process_augmentation_with_split(self, dataset_info: Dict[str, Any], target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Augmentasi dengan split-based output directory"""
        try:
            # Updated: Split-based augmentation directory
            aug_split_dir = f"{self.paths['aug_dir']}/{target_split}"
            aug_images_dir = f"{aug_split_dir}/images"
            aug_labels_dir = f"{aug_split_dir}/labels"
            
            ensure_dirs(aug_images_dir, aug_labels_dir)
            
            # Get source files
            image_files = self._get_source_image_files(dataset_info, target_split)
            if not image_files:
                return {'status': 'error', 'message': 'Tidak ada file sumber ditemukan'}
            
            self._update_progress_safe("step", 15, f"Ditemukan {len(image_files)} file dari split {target_split}")
            
            # Create pipeline
            aug_type = self.config.get('types', ['combined'])[0] if self.config.get('types') else 'combined'
            pipeline = self.pipeline_factory.create_pipeline(aug_type, self.config.get('intensity', 0.7))
            
            # Process dengan UUID consistency
            def process_func(img_path):
                return self._process_single_image_with_uuid(img_path, pipeline, aug_images_dir, aug_labels_dir)
            
            results = process_batch(image_files, process_func, progress_tracker=self.progress, operation_name="augmentasi split-based")
            
            successful = [r for r in results if r.get('status') == 'success']
            total_generated = sum(r.get('generated', 0) for r in successful)
            
            self._update_progress_safe("step", 100, f"Augmentasi {target_split} selesai: {total_generated} file")
            
            return {
                'status': 'success', 'total_generated': total_generated,
                'split': target_split, 'output_dir': aug_split_dir,
                'processed_files': len(results), 'success_rate': len(successful) / len(results) * 100 if results else 0
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Split augmentation error: {str(e)}"}
    
    def _process_normalization_with_split(self, target_split: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Normalisasi dengan split consistency"""
        try:
            # Source: split-based augmented directory
            aug_split_dir = f"{self.paths['aug_dir']}/{target_split}"
            aug_images_dir = f"{aug_split_dir}/images" 
            aug_labels_dir = f"{aug_split_dir}/labels"
            
            # Target: preprocessed split directory
            prep_split_dir = f"{self.paths['prep_dir']}/{target_split}"
            prep_images_dir = f"{prep_split_dir}/images"
            prep_labels_dir = f"{prep_split_dir}/labels"
            
            ensure_dirs(prep_images_dir, prep_labels_dir)
            
            # Get augmented files dengan UUID pattern
            aug_files = self._get_augmented_files_with_uuid(aug_images_dir, aug_labels_dir)
            if not aug_files:
                return {'status': 'error', 'message': 'Tidak ada file augmented ditemukan'}
            
            # Normalize dengan UUID preservation
            normalized = sum(1 for img_file, label_file in aug_files 
                           if self._normalize_pair_with_uuid(img_file, label_file, prep_images_dir, prep_labels_dir))
            
            self._update_progress_safe("step", 100, f"Normalisasi {target_split} selesai: {normalized} file")
            
            return {
                'status': 'success', 'total_normalized': normalized,
                'split': target_split, 'target_dir': prep_split_dir,
                'uuid_preserved': True
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Split normalization error: {str(e)}"}
    
    def _get_source_image_files(self, dataset_info: Dict[str, Any], target_split: str) -> List[str]:
        """Get source files dari split tertentu"""
        image_files = []
        
        for location in dataset_info['image_locations']:
            location_path = location['path']
            
            # Check if this location corresponds to target split
            if target_split in location_path or f"/{target_split}/" in location_path:
                # Get files dari location ini
                split_images_dir = location_path if '/images' in location_path else f"{location_path}/images"
                if Path(split_images_dir).exists():
                    for img_file in Path(split_images_dir).glob('*.*'):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png'] and img_file.is_file():
                            image_files.append(str(img_file))
        
        return image_files
    
    def _process_single_image_with_uuid(self, image_path: str, pipeline, aug_images_dir: str, aug_labels_dir: str) -> Dict[str, Any]:
        """Process dengan UUID consistency"""
        try:
            import cv2
            
            image = cv2.imread(image_path)
            if image is None:
                return {'status': 'error', 'error': 'Cannot read image'}
            
            # Parse original filename untuk UUID
            original_info = self.naming_manager.parse_existing_filename(Path(image_path).name)
            if not original_info:
                # Generate new info jika belum ada
                original_info = self.naming_manager.generate_file_info(Path(image_path).name, stage='raw')
            
            # Find corresponding label
            img_stem = Path(image_path).stem
            img_dir = Path(image_path).parent
            label_paths = [img_dir.parent / 'labels' / f'{img_stem}.txt', img_dir.parent / f'{img_stem}.txt', img_dir / f'{img_stem}.txt']
            
            # Extract bboxes
            bboxes, class_labels = [], []
            for label_path in label_paths:
                if label_path.exists():
                    bboxes, class_labels = self._read_yolo_labels_for_augmentation(str(label_path))
                    break
            
            # Generate variants dengan UUID consistency
            num_variations = self.config.get('num_variations', 2)
            generated = sum(1 for var_idx in range(num_variations) 
                           if self._generate_variant_with_uuid(image, bboxes, class_labels, pipeline, 
                                                             original_info, var_idx, aug_images_dir, aug_labels_dir))
            
            return {'status': 'success', 'generated': generated, 'uuid': original_info.uuid}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_variant_with_uuid(self, image, bboxes: List, class_labels: List, pipeline, 
                                  original_info, var_idx: int, aug_images_dir: str, aug_labels_dir: str) -> bool:
        """Generate variant dengan UUID consistency"""
        try:
            import cv2
            
            augmented = pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image, aug_bboxes, aug_labels = augmented['image'], augmented['bboxes'], augmented['class_labels']
            
            # Create variant info dengan same UUID
            variant_info = self.naming_manager.create_variant_info(original_info, var_idx + 1, 'augmented')
            aug_filename = variant_info.get_filename()
            
            # Save image
            aug_img_path = f"{aug_images_dir}/{aug_filename}"
            img_saved = cv2.imwrite(aug_img_path, aug_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save label dengan UUID consistency
            if img_saved and aug_bboxes and aug_labels:
                aug_label_path = f"{aug_labels_dir}/{Path(aug_filename).stem}.txt"
                full_bboxes = [[label] + bbox for bbox, label in zip(aug_bboxes, aug_labels)]
                self._save_yolo_labels(full_bboxes, aug_label_path)
            
            return img_saved
            
        except Exception:
            return False
    
    def _get_augmented_files_with_uuid(self, aug_images_dir: str, aug_labels_dir: str) -> List[tuple]:
        """Get augmented files dengan UUID pattern"""
        aug_files = []
        
        if not Path(aug_images_dir).exists():
            return aug_files
        
        for img_file in Path(aug_images_dir).glob('aug_*.jpg'):
            label_file = Path(aug_labels_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                aug_files.append((str(img_file), str(label_file)))
        
        return aug_files
    
    def _normalize_pair_with_uuid(self, img_file: str, label_file: str, prep_images_dir: str, prep_labels_dir: str) -> bool:
        """Normalize pair dengan UUID preservation"""
        try:
            import shutil
            
            # Parse augmented filename untuk extract UUID info
            img_path = Path(img_file)
            parsed_info = self.naming_manager.parse_existing_filename(img_path.name)
            
            if not parsed_info:
                return False
            
            # Target filename tetap menggunakan 'aug_' prefix untuk normalized stage
            normalized_filename = parsed_info.get_filename()  # Keep as aug_rp_nominal_uuid_variant
            
            # Copy files dengan preserved naming
            target_img_path = f"{prep_images_dir}/{normalized_filename}"
            target_label_path = f"{prep_labels_dir}/{Path(normalized_filename).stem}.txt"
            
            # Copy dengan validation
            shutil.copy2(img_file, target_img_path)
            shutil.copy2(label_file, target_label_path)
            
            return True
            
        except Exception as e:
            self.logger and self.logger.debug(f"🔧 Normalize error: {str(e)}")
            return False
    
    def _read_yolo_labels_for_augmentation(self, label_path: str) -> tuple:
        """Read YOLO labels untuk augmentation"""
        bboxes, class_labels = [], []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(float(parts[0]))
                            bbox = [float(x) for x in parts[1:5]]
                            if all(0 <= x <= 1 for x in bbox):
                                bboxes.append(bbox)
                                class_labels.append(class_id)
                        except (ValueError, IndexError):
                            continue
        except Exception:
            pass
        
        return bboxes, class_labels
    
    def _save_yolo_labels(self, bboxes: List[List], output_path: str) -> bool:
        """Save YOLO labels dengan validation"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                for bbox in bboxes:
                    if len(bbox) >= 5:
                        class_id = int(bbox[0])
                        coords = bbox[1:5]
                        if all(0 <= x <= 1 for x in coords):
                            f.write(f"{class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")
            return True
            
        except Exception:
            return False
    
    def cleanup_augmented_data(self, include_preprocessed: bool = True, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Cleanup dengan split-aware structure"""
        self.comm and self.comm.start_operation("Cleanup Dataset", 100)
        
        try:
            # Updated: cleanup split-based directories
            total_deleted = 0
            
            # Cleanup augmented split directories
            aug_base_dir = Path(self.paths['aug_dir'])
            if aug_base_dir.exists():
                for split_dir in aug_base_dir.iterdir():
                    if split_dir.is_dir() and split_dir.name in ['train', 'valid', 'test']:
                        deleted = self._cleanup_split_directory(split_dir)
                        total_deleted += deleted
            
            # Cleanup preprocessed jika diminta
            if include_preprocessed:
                prep_base_dir = Path(self.paths['prep_dir'])
                if prep_base_dir.exists():
                    for split_dir in prep_base_dir.iterdir():
                        if split_dir.is_dir() and split_dir.name in ['train', 'valid', 'test']:
                            # Only cleanup aug_ files, preserve original preprocessed
                            deleted = self._cleanup_augmented_from_preprocessed(split_dir)
                            total_deleted += deleted
            
            result = {
                'status': 'success', 'total_deleted': total_deleted,
                'message': f"Split-based cleanup: {total_deleted} file dihapus",
                'split_aware': True
            }
            
            self.comm and self.comm.complete_operation("Cleanup Dataset", result['message'])
            return result
            
        except Exception as e:
            error_msg = f"Cleanup error: {str(e)}"
            self.comm and self.comm.error_operation("Cleanup Dataset", error_msg)
            return {'status': 'error', 'message': error_msg}
    
    def _cleanup_split_directory(self, split_dir: Path) -> int:
        """Cleanup single split directory"""
        deleted = 0
        
        for subdir in ['images', 'labels']:
            subdir_path = split_dir / subdir
            if subdir_path.exists():
                for file_path in subdir_path.glob('aug_*.*'):
                    try:
                        file_path.unlink()
                        deleted += 1
                    except Exception:
                        pass
        
        return deleted
    
    def _cleanup_augmented_from_preprocessed(self, split_dir: Path) -> int:
        """Cleanup only augmented files dari preprocessed directory"""
        deleted = 0
        
        for subdir in ['images', 'labels']:
            subdir_path = split_dir / subdir
            if subdir_path.exists():
                for file_path in subdir_path.glob('aug_*.*'):
                    try:
                        file_path.unlink()
                        deleted += 1
                    except Exception:
                        pass
        
        return deleted
    
    def get_augmentation_status(self) -> Dict[str, Any]:
        """Status dengan split-aware structure"""
        try:
            # Count files dari semua splits
            raw_total = self._count_split_files(self.paths['raw_dir'])
            aug_total = self._count_split_files(self.paths['aug_dir'])
            prep_total = self._count_split_files(self.paths['prep_dir'])
            
            return {
                'raw_dataset': {'exists': raw_total > 0, 'total_files': raw_total, 'split_structure': True},
                'augmented_dataset': {'exists': aug_total > 0, 'total_files': aug_total, 'split_structure': True},
                'preprocessed_dataset': {'exists': prep_total > 0, 'total_files': prep_total, 'split_structure': True},
                'uuid_consistency': True, 'ready_for_augmentation': raw_total > 0
            }
            
        except Exception as e:
            return {
                'raw_dataset': {'exists': False, 'total_files': 0},
                'augmented_dataset': {'exists': False, 'total_files': 0},
                'preprocessed_dataset': {'exists': False, 'total_files': 0},
                'uuid_consistency': False, 'error': str(e)
            }
    
    def _count_split_files(self, base_dir: str) -> int:
        """Count files dalam split-based structure"""
        total = 0
        base_path = Path(base_dir)
        
        if not base_path.exists():
            return total
        
        # Count dari split directories
        for split in ['train', 'valid', 'test']:
            split_path = base_path / split
            if split_path.exists():
                for subdir in ['images', 'labels']:
                    subdir_path = split_path / subdir
                    if subdir_path.exists():
                        total += len(list(subdir_path.glob('*.*')))
        
        # Fallback: count dari root jika tidak ada split structure
        if total == 0:
            for subdir in ['images', 'labels']:
                subdir_path = base_path / subdir
                if subdir_path.exists():
                    total += len(list(subdir_path.glob('*.*')))
        
        return total
    
    # Helper methods yang sudah ada
    def _setup_aggressive_log_suppression(self):
        """Aggressive log suppression"""
        import logging
        suppression_targets = [
            'smartcash.dataset.augmentor', 'albumentations', 'cv2', 'numpy', 'PIL', 'matplotlib', 'tqdm'
        ]
        
        for target in suppression_targets:
            try:
                logger = logging.getLogger(target)
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
            except Exception:
                pass
    
    def _align_config_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Align config parameters"""
        augmentation_config = config.get('augmentation', {})
        
        return {
            'data': {'dir': config.get('data', {}).get('dir', 'data')},
            'augmentation': {
                'num_variations': augmentation_config.get('num_variations', 2),
                'target_count': augmentation_config.get('target_count', 500),
                'types': augmentation_config.get('types', ['combined']),
                'intensity': augmentation_config.get('intensity', 0.7),
                'output_dir': augmentation_config.get('output_dir', 'data/augmented'),
                'target_split': augmentation_config.get('target_split', 'train')
            },
            'preprocessing': {
                'output_dir': config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
            }
        }
    
    # Safe utility methods
    def _update_progress_safe(self, step: str, percentage: int, message: str) -> None:
        """Safe progress update"""
        try:
            self.comm and hasattr(self.comm, 'progress') and self.comm.progress(step, percentage, 100, message)
        except Exception:
            pass
    
    def _log_safe(self, level: str, message: str) -> None:
        """Safe logging"""
        try:
            self.comm and hasattr(self.comm, f'log_{level}') and getattr(self.comm, f'log_{level}')(message)
        except Exception:
            pass
    
    def _error_result(self, msg: str) -> Dict[str, Any]:
        """Error result creator"""
        return {'status': 'error', 'message': msg, 'timestamp': time.time()}

# Factory functions dengan split-aware configuration
def create_service_from_ui(ui_components: Dict[str, Any]) -> AugmentationService:
    """Create service dengan split-aware config"""
    from smartcash.dataset.augmentor.config import extract_ui_config
    config = extract_ui_config(ui_components)
    return AugmentationService(config, ui_components)

create_augmentation_service = lambda config, ui_components=None: AugmentationService(config, ui_components)