"""
File: smartcash/dataset/samples/manager.py
Deskripsi: Manager untuk menyediakan sample paths dan metadata untuk perbandingan visualisasi
"""

import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.layer_config_interface import ILayerConfigManager
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.dataset.utils.path_validator import get_path_validator

# One-liner utilities untuk path operations
resolve_path = lambda path: Path(path).resolve() if Path(path).exists() else None
get_image_files = lambda dir_path: [f for f in Path(dir_path).glob('*.*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png']] if Path(dir_path).exists() else []
get_label_path = lambda img_path, labels_dir: Path(labels_dir) / f"{img_path.stem}.txt" if Path(labels_dir).exists() else None
extract_class_from_label = lambda label_path: [int(float(line.split()[0])) for line in Path(label_path).read_text().splitlines() if line.strip()] if Path(label_path).exists() else []

class DatasetSampleManager:
    """Manager untuk menyediakan sample comparison data tanpa visualisasi"""
    
    def __init__(self, config: Dict[str, Any], logger=None, layer_config: Optional[ILayerConfigManager] = None):
        self.config = config
        self.logger = logger or get_logger()
        self.layer_config = layer_config or get_layer_config()
        self.path_validator = get_path_validator(logger)
        self.dataset_utils = DatasetUtils(config, logger=logger, layer_config=layer_config)
        
        # One-liner path setup
        self.raw_dir = resolve_path(config.get('data', {}).get('dir', 'data'))
        self.preprocessed_dir = resolve_path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        self.augmented_dir = resolve_path(config.get('augmentation', {}).get('output_dir', 'data/augmented'))
    
    def get_comparison_samples(self, num_samples: int = 5, target_split: str = 'train') -> Dict[str, Any]:
        """Dapatkan sample paths untuk perbandingan Raw + Processed + Augmented + Normalized"""
        self.logger.info(f"🎯 Mengumpulkan {num_samples} sample untuk perbandingan")
        
        try:
            # Get raw samples dengan class diversity
            raw_samples = self._get_diverse_raw_samples(num_samples, target_split)
            if not raw_samples:
                return {'status': 'error', 'message': 'Tidak ada raw samples ditemukan'}
            
            # Build comprehensive comparison data
            comparison_data = []
            for raw_sample in raw_samples:
                sample_data = self._build_sample_comparison_data(raw_sample, target_split)
                if sample_data['status'] == 'success':
                    comparison_data.append(sample_data['data'])
            
            self.logger.success(f"✅ {len(comparison_data)} sample siap untuk perbandingan")
            
            return {
                'status': 'success',
                'total_samples': len(comparison_data),
                'target_split': target_split,
                'samples': comparison_data,
                'metadata': self._generate_samples_metadata(comparison_data)
            }
            
        except Exception as e:
            error_msg = f"Error getting comparison samples: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            return {'status': 'error', 'message': error_msg}
    
    def _get_diverse_raw_samples(self, num_samples: int, target_split: str) -> List[Dict[str, Any]]:
        """Get raw samples dengan class diversity untuk representasi yang baik"""
        raw_split_dir = self.raw_dir / target_split if self.raw_dir else None
        if not raw_split_dir or not raw_split_dir.exists():
            return []
        
        # Get images dan analyze classes
        images_dir = raw_split_dir / 'images' if (raw_split_dir / 'images').exists() else raw_split_dir
        labels_dir = raw_split_dir / 'labels' if (raw_split_dir / 'labels').exists() else raw_split_dir
        
        image_files = get_image_files(images_dir)[:50]  # Limit untuk performance
        
        # Categorize by class untuk diversity
        class_samples = defaultdict(list)
        for img_path in image_files:
            label_path = get_label_path(img_path, labels_dir)
            classes = extract_class_from_label(label_path) if label_path else []
            primary_class = classes[0] if classes else 'unknown'
            class_name = self.dataset_utils.get_class_name(primary_class) if isinstance(primary_class, int) else 'unknown'
            layer = self.dataset_utils.get_layer_from_class(primary_class) if isinstance(primary_class, int) else 'unknown'
            
            class_samples[primary_class].append({
                'image_path': str(img_path),
                'label_path': str(label_path) if label_path else None,
                'primary_class': primary_class,
                'class_name': class_name,
                'layer': layer,
                'all_classes': classes
            })
        
        # Select diverse samples
        selected_samples = []
        classes_list = list(class_samples.keys())
        random.shuffle(classes_list)
        
        samples_per_class = max(1, num_samples // len(classes_list)) if classes_list else 1
        
        for class_id in classes_list:
            if len(selected_samples) >= num_samples:
                break
            samples_to_take = min(samples_per_class, len(class_samples[class_id]), num_samples - len(selected_samples))
            selected_samples.extend(random.sample(class_samples[class_id], samples_to_take))
        
        # Fill remaining slots jika kurang
        while len(selected_samples) < num_samples and len(selected_samples) < sum(len(samples) for samples in class_samples.values()):
            for class_samples_list in class_samples.values():
                if len(selected_samples) >= num_samples:
                    break
                remaining = [s for s in class_samples_list if s not in selected_samples]
                if remaining:
                    selected_samples.append(random.choice(remaining))
        
        return selected_samples[:num_samples]
    
    def _build_sample_comparison_data(self, raw_sample: Dict[str, Any], target_split: str) -> Dict[str, Any]:
        """Build comprehensive comparison data untuk single sample"""
        try:
            raw_path = Path(raw_sample['image_path'])
            base_filename = raw_path.stem
            
            # Find corresponding files di berbagai stages
            comparison_paths = {
                'raw': {
                    'image': str(raw_path),
                    'label': raw_sample.get('label_path'),
                    'exists': raw_path.exists()
                },
                'preprocessed': self._find_preprocessed_file(base_filename, target_split),
                'augmented': self._find_augmented_files(base_filename, target_split),
                'normalized': self._find_normalized_files(base_filename, target_split)
            }
            
            return {
                'status': 'success',
                'data': {
                    'base_filename': base_filename,
                    'class_info': {
                        'primary_class': raw_sample['primary_class'],
                        'class_name': raw_sample['class_name'],
                        'layer': raw_sample['layer'],
                        'all_classes': raw_sample['all_classes']
                    },
                    'paths': comparison_paths,
                    'availability': self._check_paths_availability(comparison_paths)
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Error building sample data: {str(e)}"}
    
    def _find_preprocessed_file(self, base_filename: str, target_split: str) -> Dict[str, Any]:
        """Find preprocessed file dengan berbagai naming patterns"""
        if not self.preprocessed_dir:
            return {'image': None, 'label': None, 'exists': False}
        
        prep_split_dir = self.preprocessed_dir / target_split
        if not prep_split_dir.exists():
            return {'image': None, 'label': None, 'exists': False}
        
        # Search patterns untuk preprocessed files
        search_patterns = [base_filename, f"pre_{base_filename}", f"prep_{base_filename}"]
        images_dir = prep_split_dir / 'images' if (prep_split_dir / 'images').exists() else prep_split_dir
        labels_dir = prep_split_dir / 'labels' if (prep_split_dir / 'labels').exists() else prep_split_dir
        
        for pattern in search_patterns:
            img_candidates = list(images_dir.glob(f"{pattern}.*"))
            if img_candidates:
                img_path = img_candidates[0]
                label_path = get_label_path(img_path, labels_dir)
                return {
                    'image': str(img_path),
                    'label': str(label_path) if label_path else None,
                    'exists': True
                }
        
        return {'image': None, 'label': None, 'exists': False}
    
    def _find_augmented_files(self, base_filename: str, target_split: str) -> Dict[str, Any]:
        """Find augmented files (multiple variants)"""
        if not self.augmented_dir:
            return {'images': [], 'labels': [], 'exists': False}
        
        aug_split_dir = self.augmented_dir / target_split
        if not aug_split_dir.exists():
            return {'images': [], 'labels': [], 'exists': False}
        
        images_dir = aug_split_dir / 'images' if (aug_split_dir / 'images').exists() else aug_split_dir
        labels_dir = aug_split_dir / 'labels' if (aug_split_dir / 'labels').exists() else aug_split_dir
        
        # Find augmented variants
        aug_patterns = [f"aug_{base_filename}_*", f"aug_*{base_filename}*", f"*{base_filename}*aug*"]
        found_images, found_labels = [], []
        
        for pattern in aug_patterns:
            img_candidates = list(images_dir.glob(f"{pattern}.*"))[:3]  # Limit untuk performance
            for img_path in img_candidates:
                label_path = get_label_path(img_path, labels_dir)
                found_images.append(str(img_path))
                if label_path:
                    found_labels.append(str(label_path))
        
        return {
            'images': found_images,
            'labels': found_labels,
            'exists': len(found_images) > 0
        }
    
    def _find_normalized_files(self, base_filename: str, target_split: str) -> Dict[str, Any]:
        """Find normalized files di preprocessed directory dengan aug prefix"""
        if not self.preprocessed_dir:
            return {'images': [], 'labels': [], 'exists': False}
        
        prep_split_dir = self.preprocessed_dir / target_split
        if not prep_split_dir.exists():
            return {'images': [], 'labels': [], 'exists': False}
        
        images_dir = prep_split_dir / 'images' if (prep_split_dir / 'images').exists() else prep_split_dir
        labels_dir = prep_split_dir / 'labels' if (prep_split_dir / 'labels').exists() else prep_split_dir
        
        # Find normalized (aug_ files di preprocessed)
        norm_patterns = [f"aug_{base_filename}_*", f"aug_*{base_filename}*"]
        found_images, found_labels = [], []
        
        for pattern in norm_patterns:
            img_candidates = list(images_dir.glob(f"{pattern}.*"))[:3]  # Limit untuk performance
            for img_path in img_candidates:
                label_path = get_label_path(img_path, labels_dir)
                found_images.append(str(img_path))
                if label_path:
                    found_labels.append(str(label_path))
        
        return {
            'images': found_images,
            'labels': found_labels,
            'exists': len(found_images) > 0
        }
    
    def _check_paths_availability(self, comparison_paths: Dict[str, Any]) -> Dict[str, bool]:
        """Check availability untuk setiap stage"""
        return {
            'raw': comparison_paths['raw']['exists'],
            'preprocessed': comparison_paths['preprocessed']['exists'],
            'augmented': comparison_paths['augmented']['exists'],
            'normalized': comparison_paths['normalized']['exists']
        }
    
    def _generate_samples_metadata(self, comparison_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metadata untuk samples"""
        class_distribution = defaultdict(int)
        layer_distribution = defaultdict(int)
        availability_stats = defaultdict(int)
        
        for sample in comparison_data:
            class_info = sample['class_info']
            class_distribution[class_info['class_name']] += 1
            layer_distribution[class_info['layer']] += 1
            
            availability = sample['availability']
            for stage, available in availability.items():
                if available:
                    availability_stats[stage] += 1
        
        return {
            'class_distribution': dict(class_distribution),
            'layer_distribution': dict(layer_distribution),
            'availability_stats': dict(availability_stats),
            'total_classes': len(class_distribution),
            'total_layers': len(layer_distribution)
        }
    
    def get_sample_by_class(self, class_id: int, target_split: str = 'train') -> Optional[Dict[str, Any]]:
        """Get specific sample berdasarkan class ID"""
        samples = self.get_comparison_samples(10, target_split)
        if samples['status'] != 'success':
            return None
        
        for sample in samples['samples']:
            if sample['class_info']['primary_class'] == class_id:
                return sample
        
        return None
    
    def get_layer_samples(self, layer_name: str, target_split: str = 'train', num_samples: int = 3) -> List[Dict[str, Any]]:
        """Get samples untuk specific layer"""
        all_samples = self.get_comparison_samples(15, target_split)
        if all_samples['status'] != 'success':
            return []
        
        layer_samples = [sample for sample in all_samples['samples'] 
                        if sample['class_info']['layer'] == layer_name]
        
        return layer_samples[:num_samples]

# One-liner factory functions
create_sample_manager = lambda config, logger=None: DatasetSampleManager(config, logger)
get_quick_samples = lambda config, num=5, split='train': DatasetSampleManager(config).get_comparison_samples(num, split)
check_sample_availability = lambda config, split='train': DatasetSampleManager(config).get_comparison_samples(1, split)['status'] == 'success'

# Utility functions untuk external usage
def get_sample_paths_only(config: Dict[str, Any], num_samples: int = 5, target_split: str = 'train') -> List[Dict[str, str]]:
    """Get hanya paths tanpa metadata lengkap untuk performance"""
    manager = DatasetSampleManager(config)
    result = manager.get_comparison_samples(num_samples, target_split)
    
    if result['status'] != 'success':
        return []
    
    return [{'raw': sample['paths']['raw']['image'], 
             'preprocessed': sample['paths']['preprocessed']['image'],
             'class_name': sample['class_info']['class_name']} 
            for sample in result['samples'] 
            if sample['paths']['raw']['image']]

def validate_sample_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sample structure untuk troubleshooting"""
    manager = DatasetSampleManager(config)
    
    validation = {
        'raw_dir': manager.raw_dir.exists() if manager.raw_dir else False,
        'preprocessed_dir': manager.preprocessed_dir.exists() if manager.preprocessed_dir else False,
        'augmented_dir': manager.augmented_dir.exists() if manager.augmented_dir else False,
        'available_splits': []
    }
    
    if manager.raw_dir and manager.raw_dir.exists():
        validation['available_splits'] = [d.name for d in manager.raw_dir.iterdir() 
                                        if d.is_dir() and d.name in ['train', 'valid', 'test']]
    
    return validation