"""
File: smartcash/dataset/augmentor/utils/sample_generator.py
Deskripsi: Sample generator dengan pattern sample_aug_* yang copy file dari augmented ke preprocessed
"""

import shutil
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from smartcash.common.utils.file_naming_manager import create_file_naming_manager, generate_augmented_sample_filename
from smartcash.common.logger import get_logger

class AugmentationSampleGenerator:
    """📸 Sample generator untuk augmentasi dengan pattern sample_aug_*"""
    
    def __init__(self, config: Dict[str, Any] = None):
        from smartcash.dataset.augmentor.utils.config_validator import validate_augmentation_config, get_default_augmentation_config
        
        if config is None:
            self.config = get_default_augmentation_config()
        else:
            self.config = validate_augmentation_config(config)
        
        self.logger = get_logger(__name__)
        self.naming_manager = create_file_naming_manager(config)
        self.data_dir = self.config.get('data', {}).get('dir', 'data')
    
    def generate_augmentation_samples(self, target_split: str = 'train', max_samples: int = 5, 
                                    max_per_class: int = 2) -> Dict[str, Any]:
        """🎯 Generate samples dari augmented files dengan copy ke preprocessed sebagai sample_aug_*"""
        try:
            self.logger.info(f"📸 Generating {max_samples} augmentation samples dari {target_split}")
            
            # Scan augmented files
            aug_path = Path(self.data_dir) / 'augmented' / target_split
            prep_path = Path(self.data_dir) / 'preprocessed' / target_split
            
            if not aug_path.exists() or not (aug_path / 'images').exists():
                return {
                    'status': 'error',
                    'message': f'Augmented directory tidak ditemukan: {aug_path}',
                    'samples': []
                }
            
            # Create preprocessed directories if not exist
            (prep_path / 'images').mkdir(parents=True, exist_ok=True)
            (prep_path / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Get augmented files dengan FileNamingManager
            aug_files = self._get_augmented_files_by_class(aug_path)
            
            if not aug_files:
                return {
                    'status': 'error',
                    'message': 'Tidak ada file augmented ditemukan',
                    'samples': []
                }
            
            # Select samples dengan class balancing
            selected_samples = self._select_balanced_samples(aug_files, max_samples, max_per_class)
            
            # Generate samples dengan copy
            generated_samples = []
            for sample_info in selected_samples:
                try:
                    result = self._generate_single_sample(sample_info, aug_path, prep_path)
                    if result['status'] == 'success':
                        generated_samples.append(result['sample'])
                except Exception as e:
                    self.logger.warning(f"⚠️ Error generating sample {sample_info['filename']}: {str(e)}")
                    continue
            
            self.logger.success(f"✅ Generated {len(generated_samples)} augmentation samples")
            
            return {
                'status': 'success',
                'samples': generated_samples,
                'total_samples': len(generated_samples),
                'target_split': target_split,
                'source_path': str(aug_path),
                'output_path': str(prep_path)
            }
            
        except Exception as e:
            error_msg = f"Error generating augmentation samples: {str(e)}"
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg, 'samples': []}
    
    def _get_augmented_files_by_class(self, aug_path: Path) -> Dict[str, List[Dict[str, Any]]]:
        """Get augmented files grouped by class menggunakan FileNamingManager"""
        files_by_class = {}
        images_dir = aug_path / 'images'
        
        if not images_dir.exists():
            return files_by_class
        
        # Scan augmented files dengan pattern aug_*
        for img_file in images_dir.glob('aug_*.jpg'):
            try:
                # Parse filename dengan FileNamingManager
                parsed = self.naming_manager.parse_filename(img_file.name)
                
                if parsed and parsed['type'] == 'augmented':
                    nominal = parsed['nominal']
                    variance = parsed.get('variance', '001')
                    
                    # Group by nominal (class representation)
                    if nominal not in files_by_class:
                        files_by_class[nominal] = []
                    
                    files_by_class[nominal].append({
                        'filename': img_file.stem,
                        'filepath': str(img_file),
                        'nominal': nominal,
                        'variance': variance,
                        'uuid': parsed['uuid'],
                        'extension': parsed['extension']
                    })
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error parsing {img_file.name}: {str(e)}")
                continue
        
        self.logger.info(f"📊 Found augmented files: {', '.join([f'{k}({len(v)})' for k, v in files_by_class.items()])}")
        return files_by_class
    
    def _select_balanced_samples(self, files_by_class: Dict[str, List[Dict[str, Any]]], 
                               max_samples: int, max_per_class: int) -> List[Dict[str, Any]]:
        """Select samples dengan class balancing"""
        selected_samples = []
        
        # Sort classes by priority (main banknotes first)
        priority_order = ['001000', '002000', '005000', '010000', '020000', '050000', '100000']
        sorted_classes = sorted(files_by_class.keys(), 
                              key=lambda x: priority_order.index(x) if x in priority_order else 999)
        
        samples_per_class = {}
        total_selected = 0
        
        # Select samples per class
        for nominal in sorted_classes:
            if total_selected >= max_samples:
                break
                
            class_files = files_by_class[nominal]
            remaining_slots = min(max_per_class, max_samples - total_selected)
            
            # Random select dari class
            selected_from_class = random.sample(class_files, min(remaining_slots, len(class_files)))
            selected_samples.extend(selected_from_class)
            
            samples_per_class[nominal] = len(selected_from_class)
            total_selected += len(selected_from_class)
        
        self.logger.info(f"🎯 Selected samples: {', '.join([f'{k}({v})' for k, v in samples_per_class.items()])}")
        return selected_samples
    
    def _generate_single_sample(self, sample_info: Dict[str, Any], aug_path: Path, prep_path: Path) -> Dict[str, Any]:
        """Generate single sample dengan copy augmented file ke preprocessed sebagai sample_aug_*"""
        try:
            # Source paths
            source_img = Path(sample_info['filepath'])
            source_label = aug_path / 'labels' / f"{sample_info['filename']}.txt"
            
            # Generate sample filename dengan FileNamingManager pattern
            original_name = f"rp_{sample_info['nominal']}_{sample_info['uuid']}.jpg"
            variance = int(sample_info['variance'])
            
            sample_filename = generate_augmented_sample_filename(original_name, variance)
            sample_stem = Path(sample_filename).stem
            
            # Target paths
            target_img = prep_path / 'images' / f"{sample_stem}.jpg"
            target_label = prep_path / 'labels' / f"{sample_stem}.txt"
            
            # Copy image
            if source_img.exists():
                shutil.copy2(source_img, target_img)
            else:
                return {'status': 'error', 'error': f'Source image tidak ditemukan: {source_img}'}
            
            # Copy label with deduplication
            if source_label.exists():
                self._copy_and_deduplicate_label(source_label, target_label)
            else:
                self.logger.warning(f"⚠️ Label tidak ditemukan untuk {sample_info['filename']}")
            
            # Create sample info
            sample_data = {
                'filename': sample_stem,
                'nominal': sample_info['nominal'],
                'variance': variance,
                'uuid': sample_info['uuid'],
                'sample_path': str(target_img),
                'label_path': str(target_label) if target_label.exists() else None,
                'source_aug_path': str(source_img),
                'class_display': self.naming_manager.NOMINAL_TO_DESCRIPTION.get(sample_info['nominal'], 'Unknown'),
                'pattern': 'sample_aug_{nominal}_{uuid}_{variance:03d}.jpg'
            }
            
            return {'status': 'success', 'sample': sample_data}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _copy_and_deduplicate_label(self, source_label: Path, target_label: Path):
        """Copy and deduplicate label file for layer_1 classes during augmentation."""
        try:
            target_label.parent.mkdir(parents=True, exist_ok=True)
            
            # Read original label
            with open(source_label, 'r', encoding='utf-8') as f:
                original_lines = f.readlines()
            
            # Apply label deduplication for layer_1 classes only
            from smartcash.dataset.preprocessor.core.label_deduplicator import LabelDeduplicator
            deduplicator = LabelDeduplicator(backup_enabled=False)
            
            deduplicated_lines = deduplicator.deduplicate_labels(
                [line.strip() for line in original_lines if line.strip()], 
                layer1_classes_only=True
            )
            
            # Write deduplicated labels
            with open(target_label, 'w', encoding='utf-8') as f:
                for line in deduplicated_lines:
                    f.write(f"{line}\n")
                    
        except Exception as e:
            self.logger.error(f"❌ Error copying and deduplicating label {source_label}: {str(e)}")
            # Fallback to regular copy
            shutil.copy2(source_label, target_label)
    
    def cleanup_augmentation_samples(self, target_split: str = None) -> Dict[str, Any]:
        """🧹 Cleanup sample_aug_* files dari preprocessed directory"""
        try:
            total_removed = 0
            
            splits = [target_split] if target_split else ['train', 'valid', 'test']
            
            for split in splits:
                prep_path = Path(self.data_dir) / 'preprocessed' / split
                
                if not prep_path.exists():
                    continue
                
                # Remove sample_aug_* files
                for subdir in ['images', 'labels']:
                    dir_path = prep_path / subdir
                    if dir_path.exists():
                        for file_path in dir_path.glob('sample_aug_*'):
                            try:
                                file_path.unlink()
                                total_removed += 1
                            except Exception as e:
                                self.logger.warning(f"⚠️ Error removing {file_path}: {str(e)}")
            
            self.logger.success(f"🧹 Removed {total_removed} augmentation sample files")
            
            return {
                'status': 'success',
                'total_removed': total_removed,
                'message': f'Removed {total_removed} sample_aug_* files'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'total_removed': 0}
    
    def get_sample_statistics(self, target_split: str = 'train') -> Dict[str, Any]:
        """📊 Get statistics dari augmentation samples"""
        try:
            prep_path = Path(self.data_dir) / 'preprocessed' / target_split / 'images'
            
            if not prep_path.exists():
                return {'status': 'error', 'message': 'Preprocessed directory tidak ditemukan'}
            
            # Count sample_aug_* files
            sample_files = list(prep_path.glob('sample_aug_*'))
            
            # Group by nominal
            samples_by_class = {}
            for sample_file in sample_files:
                parsed = self.naming_manager.parse_filename(sample_file.name)
                if parsed and parsed.get('type') == 'augmented_sample':
                    nominal = parsed['nominal']
                    samples_by_class[nominal] = samples_by_class.get(nominal, 0) + 1
            
            return {
                'status': 'success',
                'total_samples': len(sample_files),
                'samples_by_class': samples_by_class,
                'target_split': target_split,
                'path': str(prep_path)
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}


def create_augmentation_sample_generator(config: Dict[str, Any] = None) -> AugmentationSampleGenerator:
    """Factory untuk create augmentation sample generator"""
    return AugmentationSampleGenerator(config)

def generate_augmentation_samples(config: Dict[str, Any], target_split: str = 'train', 
                                max_samples: int = 5) -> Dict[str, Any]:
    """One-liner untuk generate augmentation samples"""
    generator = create_augmentation_sample_generator(config)
    return generator.generate_augmentation_samples(target_split, max_samples)

def cleanup_augmentation_samples(config: Dict[str, Any], target_split: str = None) -> Dict[str, Any]:
    """One-liner untuk cleanup augmentation samples"""
    generator = create_augmentation_sample_generator(config)
    return generator.cleanup_augmentation_samples(target_split)