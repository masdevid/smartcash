"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Augmentation initializer yang inherit dari CommonInitializer dengan SRP handlers integration
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import register_augmentation_handlers

class AugmentationInitializer(CommonInitializer):
    """Initializer untuk augmentation module dengan CommonInitializer inheritance."""
    
    def __init__(self):
        super().__init__(
            module_name='dataset_augmentation',
            logger_namespace='smartcash.ui.dataset.augmentation'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Create UI components untuk augmentation module.
        
        Args:
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of UI components
        """
        # Create augmentation UI dengan config dan env
        ui_components = create_augmentation_ui(env=env, config=config)
        
        # Ensure critical components exist
        critical_components = self._get_critical_components()
        for component in critical_components:
            if component not in ui_components or ui_components[component] is None:
                raise ValueError(f"Critical component missing: {component}")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], 
                             config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Setup handlers specific untuk augmentation module.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Updated UI components dictionary dengan handlers
        """
        try:
            # Register augmentation-specific handlers
            ui_components = register_augmentation_handlers(ui_components)
            
            # Log handler registration success
            if 'logger' in ui_components:
                registered_info = ui_components.get('registered_handlers', {})
                ui_components['logger'].info(
                    f"✅ {registered_info.get('total', 0)} handlers berhasil didaftarkan"
                )
            
            return ui_components
            
        except Exception as e:
            # Log error tapi jangan fail completely
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Error setup handlers: {str(e)}")
            else:
                self.logger.error(f"❌ Error setup handlers: {str(e)}")
            
            return ui_components
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration untuk augmentation module.
        
        Returns:
            Default configuration dictionary dengan nilai research-optimized
        """
        return {
            'data_dir': 'data',
            'augmented_dir': 'data/augmented',
            'preprocessed_dir': 'data/preprocessed',
            
            # Basic augmentation parameters dengan nilai moderat
            'num_variations': 2,
            'target_count': 500,
            'output_prefix': 'aug',
            'balance_classes': False,
            
            # Advanced parameters - research optimized dan tidak ekstrim
            'fliplr': 0.5,
            'degrees': 10,          # Reduced dari 15
            'translate': 0.1,       # Reduced dari 0.15  
            'scale': 0.1,           # Reduced dari 0.15
            'hsv_h': 0.015,         # Reduced dari 0.025
            'hsv_s': 0.7,
            'brightness': 0.2,      # Reduced dari 0.3
            'contrast': 0.2,        # Reduced dari 0.3
            
            # Pipeline research types
            'augmentation_types': ['combined'],  # Default untuk penelitian
            'target_split': 'train',
            'intensity': 0.7        # Moderate intensity
        }
    
    def _get_critical_components(self) -> List[str]:
        """
        Get list of critical component keys yang harus ada.
        
        Returns:
            List of critical component keys
        """
        return [
            'ui',                    # Main UI widget
            'augment_button',        # Primary action button
            'check_button',          # Check dataset button
            'save_button',           # Save config button
            'reset_button',          # Reset config button
            
            # Configuration widgets
            'num_variations',        # Basic config
            'target_count',
            'output_prefix',
            'balance_classes',
            'augmentation_types',    # Aug types selector
            'target_split',          # Split selector
            
            # Advanced parameter widgets
            'fliplr', 'degrees', 'translate', 'scale',
            'hsv_h', 'hsv_s', 'brightness', 'contrast',
            
            # Progress dan status components
            'tracker',               # Progress tracker
            'confirmation_area'      # Confirmation dialog area
        ]
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Additional validation untuk augmentation module.
        
        Args:
            ui_components: UI components dictionary
            
        Returns:
            Validation result dictionary
        """
        # Validate button functionality
        button_keys = ['augment_button', 'check_button', 'save_button', 'reset_button']
        non_functional_buttons = []
        
        for button_key in button_keys:
            button = ui_components.get(button_key)
            if not button or not hasattr(button, 'on_click'):
                non_functional_buttons.append(button_key)
        
        if non_functional_buttons:
            return {
                'valid': False,
                'message': f"Non-functional buttons: {', '.join(non_functional_buttons)}"
            }
        
        # Validate widget values
        widget_validations = [
            ('num_variations', lambda w: getattr(w, 'value', 0) > 0, 'Jumlah variasi harus > 0'),
            ('target_count', lambda w: getattr(w, 'value', 0) > 0, 'Target count harus > 0'),
            ('augmentation_types', lambda w: len(getattr(w, 'value', [])) > 0, 'Minimal 1 jenis augmentasi harus dipilih')
        ]
        
        for widget_key, validator, error_msg in widget_validations:
            widget = ui_components.get(widget_key)
            if widget and not validator(widget):
                return {
                    'valid': False,
                    'message': error_msg
                }
        
        return {'valid': True}
    
    def _handle_save_button(self, ui_components: Dict[str, Any], button) -> None:
        """Override save button handler untuk augmentation-specific behavior."""
        from .handlers.augmentation_handler import handle_save_config_button_click
        handle_save_config_button_click(ui_components, button)
    
    def _handle_reset_button(self, ui_components: Dict[str, Any], button) -> None:
        """Override reset button handler untuk augmentation-specific behavior."""
        from .handlers.augmentation_handler import handle_reset_config_button_click
        handle_reset_config_button_click(ui_components, button)
    
    def _update_cached_config(self, new_config: Dict[str, Any]) -> None:
        """Update cached UI components dengan new config."""
        if not self._cached_components:
            return
            
        try:
            # Update basic config
            self._cached_components['config'].update(new_config)
            
            # Apply config ke UI widgets jika ada config handler
            from .handlers.config_handler import create_config_handler
            config_handler = create_config_handler(self._cached_components)
            
            # Create full config structure
            full_config = {
                'augmentation': new_config,
                'data': {'dir': new_config.get('data_dir', 'data')},
                'preprocessing': {'output_dir': new_config.get('preprocessed_dir', 'data/preprocessed')}
            }
            
            config_handler.apply_config_to_ui(full_config)
            
        except Exception as e:
            if self._cached_components and 'logger' in self._cached_components:
                self._cached_components['logger'].warning(f"⚠️ Error updating cached config: {str(e)}")

# Global instance untuk module-level access
_augmentation_initializer = None

def get_augmentation_initializer() -> AugmentationInitializer:
    """Get singleton augmentation initializer."""
    global _augmentation_initializer
    if _augmentation_initializer is None:
        _augmentation_initializer = AugmentationInitializer()
    return _augmentation_initializer

def initialize_augmentation_ui(env=None, config=None, force_refresh=False, **kwargs):
    """
    Initialize augmentation UI dengan CommonInitializer pattern.
    
    Args:
        env: Environment context
        config: Custom configuration
        force_refresh: Force refresh UI components
        **kwargs: Additional parameters
        
    Returns:
        UI widget atau error fallback
    """
    initializer = get_augmentation_initializer()
    return initializer.initialize(env=env, config=config, force_refresh=force_refresh, **kwargs)

# def reset_augmentation_module():
#     """Reset augmentation module untuk debugging."""
#     initializer = get_augmentation_initializer()
#     initializer.reset_module()
    
# def get_augmentation_status() -> Dict[str, Any]:
#     """Get status augmentation module untuk debugging."""
#     initializer = get_augmentation_initializer()
#     return initializer.get_module_status()

# # Convenience functions untuk backward compatibility
# create_augmentation_ui_safe = initialize_augmentation_ui
# get_augmentation_ui_status = get_augmentation_status

# # One-liner utilities
# init_augmentation = lambda env=None, config=None: initialize_augmentation_ui(env, config)
# reset_augmentation = lambda: reset_augmentation_module()
# status_augmentation = lambda: get_augmentation_status()