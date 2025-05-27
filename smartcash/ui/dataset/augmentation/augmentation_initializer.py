"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Fixed augmentation initializer dengan proper communicator setup dan SRP handlers
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer

class AugmentationInitializer(CommonInitializer):
    """Fixed initializer dengan proper communicator integration dan handlers"""
    
    def __init__(self):
        super().__init__(
            module_name='dataset_augmentation',
            logger_namespace='smartcash.ui.dataset.augmentation'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan proper communicator setup"""
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        # Create UI dengan enhanced config
        ui_components = create_augmentation_ui(env=env, config=config)
        
        # Setup communicator integration
        ui_components.update({
            'logger_namespace': self.logger_namespace,
            'module_initialized': True,
            'communicator_setup': 'pending'
        })
        
        # Validate critical components
        critical_missing = [comp for comp in self._get_critical_components() 
                           if comp not in ui_components or ui_components[comp] is None]
        
        if critical_missing:
            raise ValueError(f"Critical components missing: {', '.join(critical_missing)}")
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], 
                             config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan proper communicator integration"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import register_augmentation_handlers
            
            # Register handlers dengan communicator support
            ui_components = register_augmentation_handlers(ui_components)
            
            # Setup communicator jika belum ada
            if not ui_components.get('communicator_ready', False):
                self._setup_communicator_integration(ui_components)
            
            # Log handler registration
            if 'logger' in ui_components:
                registered_info = ui_components.get('registered_handlers', {})
                ui_components['logger'].info(
                    f"✅ Handlers registered: {registered_info.get('total', 0)} dengan communicator integration"
                )
            
            return ui_components
            
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Handler setup error: {str(e)}")
            else:
                self.logger.error(f"❌ Handler setup error: {str(e)}")
            
            return ui_components
    
    def _setup_communicator_integration(self, ui_components: Dict[str, Any]):
        """Setup communicator integration untuk UI components"""
        try:
            from smartcash.dataset.augmentor.communicator import create_communicator
            
            # Create communicator dengan UI components
            communicator = create_communicator(ui_components)
            ui_components['communicator'] = communicator
            ui_components['communicator_ready'] = True
            ui_components['communicator_setup'] = 'success'
            
            # Log success
            if 'logger' in ui_components:
                ui_components['logger'].info("🔗 Communicator integration berhasil")
                
        except ImportError:
            ui_components['communicator_ready'] = False
            ui_components['communicator_setup'] = 'failed'
            if 'logger' in ui_components:
                ui_components['logger'].warning("⚠️ Communicator tidak tersedia")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dengan research-optimized values"""
        return {
            'data_dir': 'data',
            'augmented_dir': 'data/augmented',
            'preprocessed_dir': 'data/preprocessed',
            
            # Research-optimized parameters
            'num_variations': 2,
            'target_count': 500,
            'output_prefix': 'aug',
            'balance_classes': False,
            
            # Moderate augmentation parameters
            'fliplr': 0.5,
            'degrees': 10,
            'translate': 0.1,
            'scale': 0.1,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'brightness': 0.2,
            'contrast': 0.2,
            
            # Research pipeline configuration
            'augmentation_types': ['combined'],
            'target_split': 'train',
            'intensity': 0.7
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components untuk proper operation"""
        return [
            'ui', 'augment_button', 'check_button', 'save_button', 'reset_button',
            'num_variations', 'target_count', 'output_prefix', 'balance_classes',
            'augmentation_types', 'target_split', 'tracker', 'confirmation_area',
            'fliplr', 'degrees', 'translate', 'scale', 'hsv_h', 'hsv_s', 'brightness', 'contrast'
        ]
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Additional validation dengan communicator check"""
        # Validate button functionality
        non_functional_buttons = [key for key in ['augment_button', 'check_button', 'save_button', 'reset_button']
                                 if not (button := ui_components.get(key)) or not hasattr(button, 'on_click')]
        
        if non_functional_buttons:
            return {'valid': False, 'message': f"Non-functional buttons: {', '.join(non_functional_buttons)}"}
        
        # Validate widget values
        validations = [
            ('num_variations', lambda w: getattr(w, 'value', 0) > 0, 'Jumlah variasi harus > 0'),
            ('target_count', lambda w: getattr(w, 'value', 0) > 0, 'Target count harus > 0'),
            ('augmentation_types', lambda w: len(getattr(w, 'value', [])) > 0, 'Minimal 1 jenis augmentasi')
        ]
        
        for widget_key, validator, error_msg in validations:
            if widget := ui_components.get(widget_key):
                if not validator(widget):
                    return {'valid': False, 'message': error_msg}
        
        return {'valid': True}
    
    def _handle_save_button(self, ui_components: Dict[str, Any], button) -> None:
        """Fixed save button handler"""
        from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_save_config_button_click
        handle_save_config_button_click(ui_components, button)
    
    def _handle_reset_button(self, ui_components: Dict[str, Any], button) -> None:
        """Fixed reset button handler"""
        from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_reset_config_button_click
        handle_reset_config_button_click(ui_components, button)
    
    def _update_cached_config(self, new_config: Dict[str, Any]) -> None:
        """Update cached config dengan train split enforcement"""
        if not self._cached_components:
            return
            
        try:
            # Update basic config
            self._cached_components['config'].update(new_config)
            
            # Apply to UI dengan config handler
            from smartcash.ui.dataset.augmentation.handlers.config_handler import create_config_handler
            config_handler = create_config_handler(self._cached_components)
            
            # Create full config dengan train enforcement
            full_config = {
                'augmentation': {**new_config, 'target_split': 'train'},
                'data': {'dir': new_config.get('data_dir', 'data')},
                'preprocessing': {'output_dir': new_config.get('preprocessed_dir', 'data/preprocessed')}
            }
            
            config_handler.apply_config_to_ui(full_config)
            
        except Exception as e:
            if self._cached_components and 'logger' in self._cached_components:
                self._cached_components['logger'].warning(f"⚠️ Error updating cached config: {str(e)}")

# Global instance management
_augmentation_initializer = None

def get_augmentation_initializer() -> AugmentationInitializer:
    """Singleton augmentation initializer"""
    global _augmentation_initializer
    if _augmentation_initializer is None:
        _augmentation_initializer = AugmentationInitializer()
    return _augmentation_initializer

def initialize_augmentation_ui(env=None, config=None, force_refresh=False, **kwargs):
    """Initialize augmentation UI dengan proper communicator setup"""
    initializer = get_augmentation_initializer()
    return initializer.initialize(env=env, config=config, force_refresh=force_refresh, **kwargs)

# One-liner utilities
init_augmentation = lambda env=None, config=None: initialize_augmentation_ui(env, config)
reset_augmentation = lambda: get_augmentation_initializer().reset_module()
status_augmentation = lambda: get_augmentation_initializer().get_module_status()