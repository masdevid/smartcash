"""
File: smartcash/model/evaluation/scenario_augmentation.py
Deskripsi: Research scenario augmentation menggunakan existing AUGMENTATION_API
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import tempfile
import shutil

from smartcash.common.logger import get_logger

class ScenarioAugmentation:
    """Research scenario augmentation dengan integration ke existing API"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger('test_augmentation')
        self.evaluation_dir = Path(self.config.get('evaluation', {}).get('data', {}).get('evaluation_dir', 'data/evaluation'))
        
    def generate_position_variations(self, test_dir: str, output_dir: str, num_variations: int = 5) -> Dict[str, Any]:
        """🔄 Generate position variation data untuk research scenario"""
        position_config = self._get_position_config(num_variations)
        
        return self._generate_scenario_data(
            scenario_name='position_variation',
            test_dir=test_dir,
            output_dir=output_dir,
            augmentation_config=position_config
        )
    
    def generate_lighting_variations(self, test_dir: str, output_dir: str, num_variations: int = 5) -> Dict[str, Any]:
        """💡 Generate lighting variation data untuk research scenario"""
        lighting_config = self._get_lighting_config(num_variations)
        
        return self._generate_scenario_data(
            scenario_name='lighting_variation',
            test_dir=test_dir,
            output_dir=output_dir,
            augmentation_config=lighting_config
        )
    
    def apply_scenario_transforms(self, scenario_type: str, test_dir: str, output_dir: str) -> Dict[str, Any]:
        """🎯 Apply specific scenario transformations"""
        scenario_configs = self.config.get('evaluation', {}).get('scenarios', {})
        
        if scenario_type not in scenario_configs:
            raise ValueError(f"❌ Scenario tidak ditemukan: {scenario_type}")
        
        scenario_config = scenario_configs[scenario_type]
        if not scenario_config.get('enabled', False):
            raise ValueError(f"❌ Scenario tidak aktif: {scenario_type}")
        
        augmentation_config = scenario_config.get('augmentation_config', {})
        
        self.logger.info(f"🎯 Applying {scenario_type} transforms")
        
        return self._generate_scenario_data(
            scenario_name=scenario_type,
            test_dir=test_dir,
            output_dir=output_dir,
            augmentation_config=augmentation_config
        )
    
    def save_scenario_data(self, scenario_results: Dict[str, Any], scenario_name: str) -> str:
        """💾 Save scenario data ke evaluation directory"""
        scenario_dir = self.evaluation_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy results ke scenario directory
        if 'augmented_dir' in scenario_results:
            source_dir = Path(scenario_results['augmented_dir'])
            if source_dir.exists():
                # Copy structure: images/ dan labels/
                for subdir in ['images', 'labels']:
                    src_path = source_dir / subdir
                    dst_path = scenario_dir / subdir
                    
                    if src_path.exists():
                        dst_path.mkdir(parents=True, exist_ok=True)
                        for file in src_path.glob('*'):
                            if file.is_file():
                                shutil.copy2(file, dst_path / file.name)
        
        self.logger.info(f"💾 Scenario data saved to: {scenario_dir}")
        return str(scenario_dir)
    
    def validate_scenario_data(self, scenario_dir: str) -> Dict[str, Any]:
        """✅ Validate generated scenario data"""
        scenario_path = Path(scenario_dir)
        
        validation_result = {
            'valid': False,
            'images_count': 0,
            'labels_count': 0,
            'missing_labels': [],
            'issues': []
        }
        
        # Check directory structure
        images_dir = scenario_path / 'images'
        labels_dir = scenario_path / 'labels'
        
        if not images_dir.exists():
            validation_result['issues'].append("❌ Images directory tidak ditemukan")
            return validation_result
        
        if not labels_dir.exists():
            validation_result['issues'].append("❌ Labels directory tidak ditemukan")
            return validation_result
        
        # Count files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        label_files = list(labels_dir.glob('*.txt'))
        
        validation_result['images_count'] = len(image_files)
        validation_result['labels_count'] = len(label_files)
        
        # Check matching labels
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                validation_result['missing_labels'].append(img_file.name)
        
        # Validation success criteria
        has_data = validation_result['images_count'] > 0 and validation_result['labels_count'] > 0
        has_matching_labels = len(validation_result['missing_labels']) == 0
        
        validation_result['valid'] = has_data and has_matching_labels
        
        if validation_result['valid']:
            self.logger.info(f"✅ Scenario data valid: {validation_result['images_count']} images")
        else:
            self.logger.warning(f"⚠️ Validation issues: {validation_result['issues']}")
        
        return validation_result
    
    def _generate_scenario_data(self, scenario_name: str, test_dir: str, output_dir: str, 
                              augmentation_config: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 Internal method untuk generate scenario data menggunakan existing API"""
        # Import existing augmentation API
        try:
            from smartcash.dataset.augmentor import augment_and_normalize
        except ImportError:
            self.logger.error("❌ Augmentation API tidak tersedia")
            raise ImportError("Augmentation API required for scenario generation")
        
        # Prepare augmentation config
        config = {
            'data': {
                'dir': test_dir
            },
            'augmentation': {
                'target_split': 'test',
                'num_variations': augmentation_config.get('num_variations', 5),
                'types': self._map_scenario_to_augmentation_type(scenario_name),
                **self._prepare_augmentation_params(scenario_name, augmentation_config)
            },
            'preprocessing': {
                'normalization': {
                    'method': 'minmax',
                    'denormalize': True  # Keep original format untuk evaluation
                }
            }
        }
        
        # Create temporary output directory
        temp_output = tempfile.mkdtemp(prefix=f'scenario_{scenario_name}_')
        
        try:
            self.logger.info(f"🔄 Generating {scenario_name} data...")
            
            # Call existing augmentation API
            result = augment_and_normalize(
                config=config,
                target_split='test',
                output_dir=temp_output
            )
            
            # Move results ke final output directory
            final_output_dir = Path(output_dir)
            final_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy augmented results
            temp_path = Path(temp_output)
            augmented_path = temp_path / 'augmented' / 'test'
            
            if augmented_path.exists():
                for subdir in ['images', 'labels']:
                    src = augmented_path / subdir
                    dst = final_output_dir / subdir
                    
                    if src.exists():
                        dst.mkdir(parents=True, exist_ok=True)
                        for file in src.glob('*'):
                            if file.is_file():
                                shutil.copy2(file, dst / file.name)
            
            # Update result dengan final paths
            result.update({
                'scenario_name': scenario_name,
                'output_dir': str(final_output_dir),
                'augmentation_config': augmentation_config
            })
            
            self.logger.info(f"✅ {scenario_name} data generated: {result.get('total_generated', 0)} files")
            return result
            
        finally:
            # Cleanup temporary directory
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
    
    def _get_position_config(self, num_variations: int) -> Dict[str, Any]:
        """📐 Get position variation configuration"""
        scenario_config = self.config.get('evaluation', {}).get('scenarios', {}).get('position_variation', {})
        
        return {
            'num_variations': num_variations,
            'rotation_range': scenario_config.get('augmentation_config', {}).get('rotation_range', [-30, 30]),
            'translation_range': scenario_config.get('augmentation_config', {}).get('translation_range', [-0.2, 0.2]),
            'scale_range': scenario_config.get('augmentation_config', {}).get('scale_range', [0.8, 1.2]),
            'perspective_range': scenario_config.get('augmentation_config', {}).get('perspective_range', 0.1),
            'horizontal_flip': scenario_config.get('augmentation_config', {}).get('horizontal_flip', 0.5)
        }
    
    def _get_lighting_config(self, num_variations: int) -> Dict[str, Any]:
        """💡 Get lighting variation configuration"""
        scenario_config = self.config.get('evaluation', {}).get('scenarios', {}).get('lighting_variation', {})
        
        return {
            'num_variations': num_variations,
            'brightness_range': scenario_config.get('augmentation_config', {}).get('brightness_range', [-0.3, 0.3]),
            'contrast_range': scenario_config.get('augmentation_config', {}).get('contrast_range', [0.7, 1.3]),
            'gamma_range': scenario_config.get('augmentation_config', {}).get('gamma_range', [0.7, 1.3]),
            'hsv_hue': scenario_config.get('augmentation_config', {}).get('hsv_hue', 15),
            'hsv_saturation': scenario_config.get('augmentation_config', {}).get('hsv_saturation', 20)
        }
    
    def _map_scenario_to_augmentation_type(self, scenario_name: str) -> List[str]:
        """🗺️ Map scenario ke augmentation types"""
        mapping = {
            'position_variation': ['position'],
            'lighting_variation': ['lighting'],
            'combined_variation': ['combined']
        }
        return mapping.get(scenario_name, ['combined'])
    
    def _prepare_augmentation_params(self, scenario_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """⚙️ Prepare parameters untuk augmentation API"""
        if scenario_name == 'position_variation':
            return {
                'position': {
                    'horizontal_flip': config.get('horizontal_flip', 0.5),
                    'rotation_limit': max(abs(r) for r in config.get('rotation_range', [-30, 30])),
                    'translate_limit': max(abs(t) for t in config.get('translation_range', [-0.2, 0.2])),
                    'scale_limit': max(abs(s - 1.0) for s in config.get('scale_range', [0.8, 1.2]))
                }
            }
        elif scenario_name == 'lighting_variation':
            return {
                'lighting': {
                    'brightness_limit': max(abs(b) for b in config.get('brightness_range', [-0.3, 0.3])),
                    'contrast_limit': max(abs(c - 1.0) for c in config.get('contrast_range', [0.7, 1.3])),
                    'hsv_hue': config.get('hsv_hue', 15),
                    'hsv_saturation': config.get('hsv_saturation', 20)
                }
            }
        else:
            # Combined parameters
            position_params = self._prepare_augmentation_params('position_variation', config)
            lighting_params = self._prepare_augmentation_params('lighting_variation', config)
            
            return {
                'combined': {
                    **position_params.get('position', {}),
                    **lighting_params.get('lighting', {})
                }
            }


# Factory functions
def create_scenario_augmentation(config: Dict[str, Any] = None) -> ScenarioAugmentation:
    """🏭 Factory untuk ScenarioAugmentation"""
    return ScenarioAugmentation(config)

def generate_research_scenario(scenario_type: str, test_dir: str, output_dir: str, 
                             config: Dict[str, Any] = None) -> Dict[str, Any]:
    """🎯 One-liner untuk generate research scenario"""
    augmenter = create_scenario_augmentation(config)
    return augmenter.apply_scenario_transforms(scenario_type, test_dir, output_dir)