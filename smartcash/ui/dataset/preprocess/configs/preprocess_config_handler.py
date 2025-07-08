"""
File: smartcash/ui/dataset/preprocess/configs/preprocess_config_handler.py
Description: Configuration handler for preprocessing module
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.dataset.preprocess.configs.preprocess_defaults import (
    get_default_preprocessing_config, FORM_VALIDATION_RULES, UI_DEFAULTS
)
from smartcash.ui.dataset.preprocess.constants import (
    YOLO_PRESETS, DEFAULT_SPLITS, SUPPORTED_SPLITS, CleanupTarget
)


class PreprocessConfigHandler(ConfigHandler):
    """
    Configuration handler for preprocessing module.
    
    Features:
    - 📊 YOLO preset management
    - 🎯 Form validation with preprocessing rules
    - 🔄 UI-Config synchronization
    - 💾 Configuration persistence
    """
    
    def __init__(self, module_name: str = 'preprocess', parent_module: str = 'dataset'):
        """
        Initialize preprocessing config handler.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
        """
        super().__init__(
            module_name=module_name,
            parent_module=parent_module,
            default_config=get_default_preprocessing_config()
        )
        
        # Configuration is already set in super()
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for preprocessing.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_preprocessing_config()
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate preprocessing configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Validate preprocessing section
            if 'preprocessing' not in config:
                errors.append("Missing 'preprocessing' section")
                return False, errors
            
            preprocessing = config['preprocessing']
            
            # Validate target splits
            if 'target_splits' in preprocessing:
                target_splits = preprocessing['target_splits']
                if not isinstance(target_splits, list) or len(target_splits) == 0:
                    errors.append("target_splits must be a non-empty list")
                else:
                    for split in target_splits:
                        if split not in SUPPORTED_SPLITS:
                            errors.append(f"Invalid split '{split}'. Supported: {SUPPORTED_SPLITS}")
            
            # Validate normalization
            if 'normalization' in preprocessing:
                norm_config = preprocessing['normalization']
                
                # Check preset
                if 'preset' in norm_config:
                    preset = norm_config['preset']
                    if preset not in YOLO_PRESETS:
                        errors.append(f"Invalid preset '{preset}'. Available: {list(YOLO_PRESETS.keys())}")
                
                # Check target size
                if 'target_size' in norm_config:
                    target_size = norm_config['target_size']
                    if not isinstance(target_size, list) or len(target_size) != 2:
                        errors.append("target_size must be a list of two integers")
                    elif not all(isinstance(x, int) and 32 <= x <= 2048 for x in target_size):
                        errors.append("target_size values must be integers between 32 and 2048")
            
            # Validate batch size
            if 'batch_size' in preprocessing:
                batch_size = preprocessing['batch_size']
                if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 256:
                    errors.append("batch_size must be an integer between 1 and 256")
            
            # Validate data section
            if 'data' not in config:
                errors.append("Missing 'data' section")
            else:
                data_config = config['data']
                if 'dir' not in data_config or not data_config['dir']:
                    errors.append("data.dir is required")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Args:
            ui_components: UI components dictionary
            
        Returns:
            Configuration dictionary
        """
        config = self.get_default_config()
        
        try:
            # Extract from form inputs
            preprocessing = config['preprocessing']
            
            # Resolution/Preset
            if 'resolution_dropdown' in ui_components:
                preset = ui_components['resolution_dropdown'].value
                if preset in YOLO_PRESETS:
                    preprocessing['normalization']['preset'] = preset
                    # Update target_size based on preset
                    preprocessing['normalization']['target_size'] = YOLO_PRESETS[preset]['target_size']
            
            # Normalization method
            if 'normalization_dropdown' in ui_components:
                preprocessing['normalization']['method'] = ui_components['normalization_dropdown'].value
            
            # Preserve aspect ratio
            if 'preserve_aspect_checkbox' in ui_components:
                preprocessing['normalization']['preserve_aspect_ratio'] = ui_components['preserve_aspect_checkbox'].value
            
            # Target splits
            if 'target_splits_select' in ui_components:
                selected_splits = ui_components['target_splits_select'].value
                if isinstance(selected_splits, (list, tuple)):
                    preprocessing['target_splits'] = list(selected_splits)
            
            # Batch size
            if 'batch_size_input' in ui_components:
                try:
                    batch_size = int(ui_components['batch_size_input'].value)
                    if 1 <= batch_size <= 256:
                        preprocessing['batch_size'] = batch_size
                except (ValueError, TypeError):
                    pass
            
            # Validation settings
            if 'validation_checkbox' in ui_components:
                preprocessing['validation']['enabled'] = ui_components['validation_checkbox'].value
            
            # Move invalid files
            if 'move_invalid_checkbox' in ui_components:
                preprocessing['move_invalid'] = ui_components['move_invalid_checkbox'].value
            
            # Invalid directory
            if 'invalid_dir_input' in ui_components:
                invalid_dir = ui_components['invalid_dir_input'].value
                if invalid_dir and invalid_dir.strip():
                    preprocessing['invalid_dir'] = invalid_dir.strip()
            
            # Cleanup target
            if 'cleanup_target_dropdown' in ui_components:
                cleanup_target = ui_components['cleanup_target_dropdown'].value
                if cleanup_target in [e.value for e in CleanupTarget]:
                    preprocessing['cleanup_target'] = cleanup_target
            
            # Backup setting
            if 'backup_checkbox' in ui_components:
                preprocessing['backup_enabled'] = ui_components['backup_checkbox'].value
            
        except Exception as e:
            self.logger.error(f"Error extracting config from UI: {e}")
        
        return config
    
    def update_ui_from_config(self, ui_components: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> None:
        """
        Update UI components from configuration.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary (uses current if None)
        """
        if config is None:
            config = self.get_config()
        
        try:
            preprocessing = config.get('preprocessing', {})
            
            # Update resolution/preset dropdown
            if 'resolution_dropdown' in ui_components:
                preset = preprocessing.get('normalization', {}).get('preset', 'yolov5s')
                ui_components['resolution_dropdown'].value = preset
            
            # Update normalization method
            if 'normalization_dropdown' in ui_components:
                method = preprocessing.get('normalization', {}).get('method', 'minmax')
                ui_components['normalization_dropdown'].value = method
            
            # Update preserve aspect ratio
            if 'preserve_aspect_checkbox' in ui_components:
                preserve_aspect = preprocessing.get('normalization', {}).get('preserve_aspect_ratio', True)
                ui_components['preserve_aspect_checkbox'].value = preserve_aspect
            
            # Update target splits
            if 'target_splits_select' in ui_components:
                target_splits = preprocessing.get('target_splits', DEFAULT_SPLITS)
                ui_components['target_splits_select'].value = target_splits
            
            # Update batch size
            if 'batch_size_input' in ui_components:
                batch_size = preprocessing.get('batch_size', 32)
                ui_components['batch_size_input'].value = str(batch_size)
            
            # Update validation checkbox
            if 'validation_checkbox' in ui_components:
                validation_enabled = preprocessing.get('validation', {}).get('enabled', False)
                ui_components['validation_checkbox'].value = validation_enabled
            
            # Update move invalid checkbox
            if 'move_invalid_checkbox' in ui_components:
                move_invalid = preprocessing.get('move_invalid', False)
                ui_components['move_invalid_checkbox'].value = move_invalid
            
            # Update invalid directory
            if 'invalid_dir_input' in ui_components:
                invalid_dir = preprocessing.get('invalid_dir', 'data/invalid')
                ui_components['invalid_dir_input'].value = invalid_dir
            
            # Update cleanup target
            if 'cleanup_target_dropdown' in ui_components:
                cleanup_target = preprocessing.get('cleanup_target', 'preprocessed')
                ui_components['cleanup_target_dropdown'].value = cleanup_target
            
            # Update backup checkbox
            if 'backup_checkbox' in ui_components:
                backup_enabled = preprocessing.get('backup_enabled', True)
                ui_components['backup_checkbox'].value = backup_enabled
            
        except Exception as e:
            self.logger.error(f"Error updating UI from config: {e}")
    
    def get_yolo_preset_config(self, preset: str) -> Dict[str, Any]:
        """
        Get YOLO preset configuration.
        
        Args:
            preset: Preset name
            
        Returns:
            Preset configuration dictionary
        """
        return YOLO_PRESETS.get(preset, YOLO_PRESETS['default']).copy()
    
    def get_effective_normalization_config(self) -> Dict[str, Any]:
        """
        Get effective normalization configuration with preset applied.
        
        Returns:
            Effective normalization configuration
        """
        config = self.get_config()
        norm_config = config.get('preprocessing', {}).get('normalization', {})
        
        # Get preset configuration
        preset = norm_config.get('preset', 'yolov5s')
        preset_config = self.get_yolo_preset_config(preset)
        
        # Merge preset with custom settings
        effective_config = preset_config.copy()
        effective_config.update(norm_config)
        
        return effective_config
    
    def get_processing_splits(self) -> List[str]:
        """
        Get list of splits to process.
        
        Returns:
            List of split names
        """
        config = self.get_config()
        return config.get('preprocessing', {}).get('target_splits', DEFAULT_SPLITS)
    
    def get_data_directories(self) -> Dict[str, str]:
        """
        Get data directory configuration.
        
        Returns:
            Dictionary with data directory paths
        """
        config = self.get_config()
        data_config = config.get('data', {})
        
        return {
            'source_dir': data_config.get('dir', 'data'),
            'preprocessed_dir': data_config.get('preprocessed_dir', 'data/preprocessed')
        }
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = self.get_default_config()
        self.save_config()
        self.logger.info("Configuration reset to defaults")