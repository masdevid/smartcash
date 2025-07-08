"""
File: smartcash/ui/dataset/augment/configs/augment_config_handler.py
Description: Configuration handler for augment module following core patterns

This handler inherits from the core ConfigHandler and implements
augment-specific configuration management while preserving business logic.
"""

from typing import Dict, Any, Optional, Tuple, List
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.core.errors.handlers import handle_ui_errors
from .augment_defaults import get_default_augment_config, get_augment_config_schema
from ..constants import (
    AugmentationTypes, CleanupTarget, TARGET_SPLIT_OPTIONS,
    AUGMENTATION_TYPES_OPTIONS, CLASS_WEIGHTS
)

class AugmentConfigHandler(ConfigHandler):
    """
    Configuration handler for augment module with validation and business logic.
    
    Features:
    - 🔧 Augment-specific configuration validation
    - 🎨 Form field mapping and extraction
    - 📊 Class balancing configuration
    - 🗑️ Cleanup target management
    - ✅ Schema-based validation
    """
    
    @handle_ui_errors(error_component_title="Augment Config Handler Error", log_error=True)
    def __init__(self, 
                 module_name: str = 'augment', 
                 parent_module: str = 'dataset',
                 default_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initialize augment config handler.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name  
            default_config: Optional default configuration
            **kwargs: Additional arguments for compatibility
        """
        # Use augment defaults if none provided
        if default_config is None:
            default_config = get_default_augment_config()
            
        super().__init__(
            module_name=module_name,
            parent_module=parent_module,
            default_config=default_config
        )
        
        # Augment-specific state
        self._schema = get_augment_config_schema()
        self._supported_types = [t.value for t in AugmentationTypes]
        self._cleanup_targets = [t.value for t in CleanupTarget]
        
        self.logger.info(f"🎨 AugmentConfigHandler initialized with {len(self._config)} config sections")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate augment configuration with business logic.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Basic structure validation
            if 'data' not in config:
                errors.append("Missing 'data' configuration section")
            elif 'dir' not in config['data']:
                errors.append("Missing data directory path")
                
            if 'augmentation' not in config:
                errors.append("Missing 'augmentation' configuration section")
                return False, errors
            
            aug_config = config['augmentation']
            
            # Validate required fields
            required_fields = ['num_variations', 'target_count', 'intensity', 'target_split']
            for field in required_fields:
                if field not in aug_config:
                    errors.append(f"Missing required field: {field}")
            
            # Validate ranges
            if 'num_variations' in aug_config:
                if not (1 <= aug_config['num_variations'] <= 10):
                    errors.append("num_variations must be between 1 and 10")
            
            if 'target_count' in aug_config:
                if not (10 <= aug_config['target_count'] <= 10000):
                    errors.append("target_count must be between 10 and 10000")
                    
            if 'intensity' in aug_config:
                if not (0.0 <= aug_config['intensity'] <= 1.0):
                    errors.append("intensity must be between 0.0 and 1.0")
            
            # Validate target split
            if 'target_split' in aug_config:
                valid_splits = [opt[1] for opt in TARGET_SPLIT_OPTIONS]
                if aug_config['target_split'] not in valid_splits:
                    errors.append(f"target_split must be one of: {valid_splits}")
            
            # Validate augmentation types
            if 'types' in aug_config:
                for aug_type in aug_config['types']:
                    if aug_type not in self._supported_types:
                        errors.append(f"Unsupported augmentation type: {aug_type}")
            
            # Validate cleanup configuration
            if 'cleanup' in config:
                cleanup_config = config['cleanup']
                if 'default_target' in cleanup_config:
                    if cleanup_config['default_target'] not in self._cleanup_targets:
                        errors.append(f"Invalid cleanup target: {cleanup_config['default_target']}")
            
            # Validate class weights if balancing is enabled
            if aug_config.get('balance_classes', False):
                if 'balancing' in config and 'layer_weights' in config['balancing']:
                    weights = config['balancing']['layer_weights']
                    for layer, weight in weights.items():
                        if not isinstance(weight, (int, float)) or weight < 0:
                            errors.append(f"Invalid weight for {layer}: must be non-negative number")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
            return False, errors
    
    def extract_ui_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Extracted configuration dictionary
        """
        try:
            config = self.get_default_config()
            
            # Extract basic augmentation settings
            aug_config = config['augmentation']
            
            # Basic form fields
            if 'num_variations_slider' in ui_components:
                aug_config['num_variations'] = ui_components['num_variations_slider'].value
            
            if 'target_count_slider' in ui_components:
                aug_config['target_count'] = ui_components['target_count_slider'].value
                
            if 'intensity_slider' in ui_components:
                aug_config['intensity'] = ui_components['intensity_slider'].value
                
            if 'balance_classes_checkbox' in ui_components:
                aug_config['balance_classes'] = ui_components['balance_classes_checkbox'].value
                
            if 'target_split_dropdown' in ui_components:
                aug_config['target_split'] = ui_components['target_split_dropdown'].value
                
            if 'augmentation_types_select' in ui_components:
                aug_config['types'] = list(ui_components['augmentation_types_select'].value)
            
            # Position parameters
            position_config = aug_config['position']
            if 'horizontal_flip_slider' in ui_components:
                position_config['horizontal_flip'] = ui_components['horizontal_flip_slider'].value
            if 'rotation_limit_slider' in ui_components:
                position_config['rotation_limit'] = ui_components['rotation_limit_slider'].value
            if 'translate_limit_slider' in ui_components:
                position_config['translate_limit'] = ui_components['translate_limit_slider'].value
            if 'scale_limit_slider' in ui_components:
                position_config['scale_limit'] = ui_components['scale_limit_slider'].value
            
            # Lighting parameters
            lighting_config = aug_config['lighting']
            if 'brightness_limit_slider' in ui_components:
                lighting_config['brightness_limit'] = ui_components['brightness_limit_slider'].value
            if 'contrast_limit_slider' in ui_components:
                lighting_config['contrast_limit'] = ui_components['contrast_limit_slider'].value
            if 'hsv_hue_slider' in ui_components:
                lighting_config['hsv_hue'] = ui_components['hsv_hue_slider'].value
            if 'hsv_saturation_slider' in ui_components:
                lighting_config['hsv_saturation'] = ui_components['hsv_saturation_slider'].value
            
            # Update combined parameters
            aug_config['combined'] = {**position_config, **lighting_config}
            
            # Extract cleanup settings
            if 'cleanup_target_dropdown' in ui_components:
                config['cleanup']['default_target'] = ui_components['cleanup_target_dropdown'].value
            
            # Extract data directory
            if 'data_dir_input' in ui_components:
                config['data']['dir'] = ui_components['data_dir_input'].value
            
            self.logger.info("✅ Configuration extracted from UI components")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract UI config: {e}")
            return self.get_default_config()
    
    def update_ui_from_config(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Update UI components from configuration.
        
        Args:
            ui_components: Dictionary of UI components to update
            config: Configuration to apply
        """
        try:
            aug_config = config.get('augmentation', {})
            
            # Update basic form fields
            if 'num_variations_slider' in ui_components and 'num_variations' in aug_config:
                ui_components['num_variations_slider'].value = aug_config['num_variations']
                
            if 'target_count_slider' in ui_components and 'target_count' in aug_config:
                ui_components['target_count_slider'].value = aug_config['target_count']
                
            if 'intensity_slider' in ui_components and 'intensity' in aug_config:
                ui_components['intensity_slider'].value = aug_config['intensity']
                
            if 'balance_classes_checkbox' in ui_components and 'balance_classes' in aug_config:
                ui_components['balance_classes_checkbox'].value = aug_config['balance_classes']
                
            if 'target_split_dropdown' in ui_components and 'target_split' in aug_config:
                ui_components['target_split_dropdown'].value = aug_config['target_split']
                
            if 'augmentation_types_select' in ui_components and 'types' in aug_config:
                ui_components['augmentation_types_select'].value = aug_config['types']
            
            # Update position parameters
            position_config = aug_config.get('position', {})
            for param in ['horizontal_flip', 'rotation_limit', 'translate_limit', 'scale_limit']:
                widget_name = f'{param}_slider'
                if widget_name in ui_components and param in position_config:
                    ui_components[widget_name].value = position_config[param]
            
            # Update lighting parameters
            lighting_config = aug_config.get('lighting', {})
            for param in ['brightness_limit', 'contrast_limit', 'hsv_hue', 'hsv_saturation']:
                widget_name = f'{param}_slider'
                if widget_name in ui_components and param in lighting_config:
                    ui_components[widget_name].value = lighting_config[param]
            
            # Update cleanup settings
            cleanup_config = config.get('cleanup', {})
            if 'cleanup_target_dropdown' in ui_components and 'default_target' in cleanup_config:
                ui_components['cleanup_target_dropdown'].value = cleanup_config['default_target']
            
            # Update data directory
            data_config = config.get('data', {})
            if 'data_dir_input' in ui_components and 'dir' in data_config:
                ui_components['data_dir_input'].value = data_config['dir']
            
            self.logger.info("✅ UI components updated from configuration")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update UI from config: {e}")
    
    def get_config_summary(self) -> Dict[str, Any]:
        """
        Get configuration summary for display.
        
        Returns:
            Dictionary containing configuration summary
        """
        config = self.config
        aug_config = config.get('augmentation', {})
        
        return {
            'augmentation_type': ', '.join(aug_config.get('types', ['combined'])),
            'variations': aug_config.get('num_variations', 2),
            'target_count': aug_config.get('target_count', 500),
            'intensity': f"{aug_config.get('intensity', 0.7):.1f}",
            'target_split': aug_config.get('target_split', 'train'),
            'balance_classes': aug_config.get('balance_classes', True),
            'data_directory': config.get('data', {}).get('dir', 'data'),
            'cleanup_target': config.get('cleanup', {}).get('default_target', 'both')
        }

# Factory function for creating config handler
def get_augment_config_handler(**kwargs) -> AugmentConfigHandler:
    """
    Factory function to create an AugmentConfigHandler instance.
    
    Args:
        **kwargs: Arguments to pass to AugmentConfigHandler constructor
        
    Returns:
        AugmentConfigHandler instance
    """
    return AugmentConfigHandler(**kwargs)