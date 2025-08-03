"""
File: smartcash/model/evaluation/scenario_augmentation.py
Deskripsi: Research scenario augmentation menggunakan existing AUGMENTATION_API
"""

import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import numpy as np
    import random
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # Create dummy classes for type hints
    class Image:
        class Image:
            pass

from smartcash.common.logger import get_logger

class ScenarioAugmentation:
    """Research scenario augmentation dengan integration ke existing API"""
    
    def __init__(self, config: Dict[str, Any] = None):
        # Ensure config is a dictionary
        if not isinstance(config, dict):
            self.logger = get_logger('test_augmentation')
            self.logger.warning(f"Config is not a dictionary (got {type(config).__name__}), using default")
            config = {}
        
        self.config = config or {}
        if not hasattr(self, 'logger'):
            self.logger = get_logger('test_augmentation')
        
        # Safe config access with proper error handling
        try:
            eval_config = self.config.get('evaluation', {})
            if isinstance(eval_config, dict):
                data_config = eval_config.get('data', {})
                if isinstance(data_config, dict):
                    eval_dir = data_config.get('evaluation_dir', 'data/evaluation')
                else:
                    eval_dir = 'data/evaluation'
            else:
                eval_dir = 'data/evaluation'
            
            self.evaluation_dir = Path(eval_dir)
        except Exception as e:
            self.logger.warning(f"Error accessing evaluation directory config: {e}")
            self.evaluation_dir = Path('data/evaluation')
        
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
        # Check if output already exists and is valid
        output_path = Path(output_dir)
        if output_path.exists():
            validation = self.validate_scenario_data(str(output_path))
            if validation['valid']:
                self.logger.info(f"📁 Using existing {scenario_name} data: {validation['images_count']} images")
                return {
                    'status': 'existing',
                    'scenario_name': scenario_name,
                    'output_dir': str(output_path),
                    'total_generated': validation['images_count'],
                    'augmentation_config': augmentation_config
                }
        
        # Try to use augmentation API first
        try:
            from smartcash.dataset.augmentor import augment_and_normalize
            
            # Prepare augmentation config for evaluation (ONE variant per image)
            config = {
                'data': {
                    'dir': test_dir,
                    'preprocessed_test_dir': test_dir  # Use the test directory directly
                },
                'augmentation': {
                    'target_split': 'test',
                    'num_variations': 1,  # Only ONE augmented version per image for evaluation
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
                self.logger.info(f"🔄 Generating {scenario_name} data using augmentation API...")
                
                # Call existing augmentation API
                result = augment_and_normalize(
                    config=config,
                    target_split='test'
                )
                
                # Move results ke final output directory
                final_output_dir = Path(output_dir)
                final_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy augmented results
                temp_path = Path(temp_output)
                augmented_path = temp_path / 'augmented' / 'test'
                
                files_copied = 0
                if augmented_path.exists():
                    for subdir in ['images', 'labels']:
                        src = augmented_path / subdir
                        dst = final_output_dir / subdir
                        
                        if src.exists():
                            dst.mkdir(parents=True, exist_ok=True)
                            for file in src.glob('*'):
                                if file.is_file():
                                    shutil.copy2(file, dst / file.name)
                                    files_copied += 1
                
                # If no files were generated, use fallback
                if files_copied == 0:
                    self.logger.warning(f"⚠️ Augmentation API generated 0 files, using fallback method")
                    result = self._generate_fallback_scenario_data(scenario_name, test_dir, output_dir, augmentation_config)
                else:
                    # Update result dengan final paths
                    result.update({
                        'scenario_name': scenario_name,
                        'output_dir': str(final_output_dir),
                        'total_generated': files_copied,
                        'augmentation_config': augmentation_config
                    })
                    
                    self.logger.info(f"✅ {scenario_name} data generated: {files_copied} files")
                
                return result
                
            finally:
                # Cleanup temporary directory
                if os.path.exists(temp_output):
                    shutil.rmtree(temp_output)
        
        except (ImportError, Exception) as e:
            self.logger.warning(f"⚠️ Augmentation API not available or failed: {e}")
            # Use fallback method
            return self._generate_fallback_scenario_data(scenario_name, test_dir, output_dir, augmentation_config)
    
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
    
    def _locate_and_move_augmented_files(self, result: Dict[str, Any], target_dir: Path) -> None:
        """🔍 Locate and move augmented files to target directory"""
        # Check various possible output locations
        possible_locations = [
            result.get('output_dir'),
            result.get('augmented_dir'),
            result.get('data_dir')
        ]
        
        for location in possible_locations:
            if location and Path(location).exists():
                source_path = Path(location)
                
                # Look for augmented subdirectories
                for subdir_name in ['augmented', 'test', 'evaluation']:
                    augmented_subdir = source_path / subdir_name
                    if augmented_subdir.exists():
                        self._copy_directory_structure(augmented_subdir, target_dir)
                        return
                
                # Direct copy if images/labels found
                if (source_path / 'images').exists() or (source_path / 'labels').exists():
                    self._copy_directory_structure(source_path, target_dir)
                    return
    
    def _copy_directory_structure(self, source_dir: Path, target_dir: Path) -> None:
        """📂 Copy directory structure from source to target"""
        for subdir in ['images', 'labels']:
            src_subdir = source_dir / subdir
            target_subdir = target_dir / subdir
            
            if src_subdir.exists():
                target_subdir.mkdir(parents=True, exist_ok=True)
                for file in src_subdir.glob('*'):
                    if file.is_file():
                        shutil.copy2(file, target_subdir / file.name)
    
    def _generate_fallback_scenario_data(self, scenario_name: str, test_dir: str, 
                                       output_dir: str, augmentation_config: Dict[str, Any]) -> Dict[str, Any]:
        """🔄 Generate scenario data with direct augmentation implementation"""
        self.logger.info(f"🔄 Generating {scenario_name} with built-in augmentation")
        
        test_path = Path(test_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output directories
        images_dir = output_path / 'images'
        labels_dir = output_path / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        files_processed = 0
        
        # Check if PIL is available
        if not PIL_AVAILABLE:
            self.logger.error(f"❌ PIL/Pillow not available for augmentation")
            # Fall back to copying if no augmentation possible
            return self._copy_original_data(test_dir, output_dir, scenario_name)
        
        # Process each image with scenario-specific augmentation
        test_images_dir = test_path / 'images'
        if test_images_dir.exists():
            for img_file in test_images_dir.glob('*.jpg'):
                if img_file.name.startswith('.'):
                    continue
                    
                try:
                    # Load original image
                    with Image.open(img_file) as img:
                        img = img.convert('RGB')
                        
                        # Apply scenario-specific transformations
                        if scenario_name == 'position_variation':
                            augmented_img = self._apply_position_augmentation(img, augmentation_config)
                        elif scenario_name == 'lighting_variation':
                            augmented_img = self._apply_lighting_augmentation(img, augmentation_config)
                        else:
                            augmented_img = img  # No augmentation for unknown scenarios
                        
                        # Save augmented image
                        target_file = images_dir / img_file.name
                        augmented_img.save(target_file, 'JPEG', quality=95)
                        files_processed += 1
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Error processing {img_file.name}: {e}")
                    # Copy original on error
                    shutil.copy2(img_file, images_dir / img_file.name)
                    files_processed += 1
        
        # Copy labels (unchanged for evaluation)
        test_labels_dir = test_path / 'labels'
        if test_labels_dir.exists():
            for label_file in test_labels_dir.glob('*.txt'):
                if not label_file.name.startswith('.'):
                    target_file = labels_dir / label_file.name
                    shutil.copy2(label_file, target_file)
        
        self.logger.info(f"✅ {scenario_name} augmentation complete: {files_processed} images processed")
        
        return {
            'status': 'success',
            'scenario_name': scenario_name,
            'output_dir': str(output_path),
            'total_generated': files_processed,
            'augmentation_config': augmentation_config,
            'method': 'built_in_augmentation'
        }
    
    def _apply_position_augmentation(self, img: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """📐 Apply position-based augmentations (rotation, translation, scaling)"""
        
        # Get parameters with randomization
        rotation_range = config.get('rotation_range', [-15, 15])
        scale_range = config.get('scale_range', [0.9, 1.1])
        
        # Apply random rotation
        angle = random.uniform(rotation_range[0], rotation_range[1])
        img = img.rotate(angle, expand=False, fillcolor=(128, 128, 128))
        
        # Apply random scaling
        scale = random.uniform(scale_range[0], scale_range[1])
        if scale != 1.0:
            w, h = img.size
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Crop or pad to original size
            if scale > 1.0:  # Crop
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = img.crop((left, top, left + w, top + h))
            else:  # Pad
                padding = ((w - new_w) // 2, (h - new_h) // 2)
                img = ImageOps.expand(img, padding, fill=(128, 128, 128))
        
        # Apply horizontal flip sometimes
        if config.get('horizontal_flip', 0.3) > random.random():
            img = ImageOps.mirror(img)
        
        return img
    
    def _apply_lighting_augmentation(self, img: Image.Image, config: Dict[str, Any]) -> Image.Image:
        """💡 Apply lighting-based augmentations (brightness, contrast, gamma)"""
        
        # Get parameters with randomization
        brightness_range = config.get('brightness_range', [0.7, 1.3])
        contrast_range = config.get('contrast_range', [0.8, 1.2])
        gamma_range = config.get('gamma_range', [0.8, 1.2])
        
        # Apply random brightness
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)
        
        # Apply random contrast
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)
        
        # Apply gamma correction
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        if gamma != 1.0:
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.power(img_array, 1.0 / gamma)
            img_array = (img_array * 255).astype(np.uint8)
            img = Image.fromarray(img_array)
        
        return img
    
    def _copy_original_data(self, test_dir: str, output_dir: str, scenario_name: str) -> Dict[str, Any]:
        """📁 Emergency fallback - copy original data when augmentation fails"""
        self.logger.warning(f"⚠️ Using emergency fallback for {scenario_name} - copying original data")
        
        test_path = Path(test_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create output directories
        images_dir = output_path / 'images'
        labels_dir = output_path / 'labels'
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        files_copied = 0
        
        # Copy images
        test_images_dir = test_path / 'images'
        if test_images_dir.exists():
            for img_file in test_images_dir.glob('*.jpg'):
                if not img_file.name.startswith('.'):
                    shutil.copy2(img_file, images_dir / img_file.name)
                    files_copied += 1
        
        # Copy labels
        test_labels_dir = test_path / 'labels'
        if test_labels_dir.exists():
            for label_file in test_labels_dir.glob('*.txt'):
                if not label_file.name.startswith('.'):
                    shutil.copy2(label_file, labels_dir / label_file.name)
        
        self.logger.warning(f"⚠️ Emergency fallback complete: {files_copied} files copied")
        
        return {
            'status': 'fallback',
            'scenario_name': scenario_name,
            'output_dir': str(output_path),
            'total_generated': files_copied,
            'method': 'emergency_copy'
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