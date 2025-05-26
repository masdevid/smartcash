from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE

# Import handlers
from smartcash.ui.dataset.augmentation.handlers.augmentation_handler import handle_augmentation_button_click
from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import handle_cleanup_button_click
from smartcash.ui.dataset.augmentation.handlers.reset_handler import handle_reset_button_click
from smartcash.ui.dataset.augmentation.handlers.save_handler import handle_save_button_click

# Import utils
from smartcash.ui.dataset.augmentation.utils.ui_observers import register_ui_observers

# Import komponen UI
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui


class AugmentationInitializer(CommonInitializer):
    """Initializer untuk UI dataset augmentation."""
    
    def __init__(self):
        super().__init__(
            module_name='dataset_augmentation',
            logger_namespace=AUGMENTATION_LOGGER_NAMESPACE
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components untuk augmentation."""
        return create_augmentation_ui(env, config)
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers khusus untuk augmentation."""
        
        # Setup observers untuk notifikasi
        try:
            observer_manager = register_ui_observers(ui_components)
            ui_components['observer_manager'] = observer_manager
        except Exception as e:
            ui_components['logger'].warning(f"Gagal setup observer: {str(e)}")
        
        # Setup augmentation-specific button handlers
        self._setup_augmentation_button_handlers(ui_components)
        
        # Setup shared progress tracking
        try:
            self._setup_shared_progress_tracking(ui_components)
        except Exception as e:
            ui_components['logger'].warning(f"Error setup progress: {str(e)}")
        
        # Load konfigurasi dan update UI
        try:
            if config and isinstance(config, dict):
                from smartcash.ui.dataset.augmentation.handlers.config_handler import update_ui_from_config
                update_ui_from_config(ui_components, config)
        except Exception as e:
            ui_components['logger'].warning(f"Gagal update UI dari config: {str(e)}")
        
        return ui_components
    
    def _setup_augmentation_button_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup handlers khusus untuk tombol augmentation."""
        logger = ui_components.get('logger')
        
        # Augmentation button
        augment_button = ui_components.get('augment_button')
        if augment_button is not None and hasattr(augment_button, 'on_click'):
            augment_button.on_click(lambda b: handle_augmentation_button_click(ui_components, b))
            logger.debug("✅ Augment button handler registered")
    
    def _setup_shared_progress_tracking(self, ui_components: Dict[str, Any]) -> None:
        """Setup shared progress tracking untuk augmentasi."""
        if 'tracker' in ui_components:
            ui_components['logger'].debug("✅ Shared progress tracking sudah tersedia")
            return
        
        # Setup fallback progress functions
        if not callable(ui_components.get('update_progress')):
            def fallback_update_progress(progress_type: str, value: int, message: str = ""):
                if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
                    ui_components['progress_bar'].value = value
                    ui_components['progress_bar'].description = f"Progress: {value}%"
                
                if message:
                    for label_key in ['progress_message', 'step_label', 'overall_label']:
                        if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
                            ui_components[label_key].value = message
            
            ui_components['update_progress'] = fallback_update_progress
        
        if not callable(ui_components.get('reset_all')):
            def fallback_reset_progress():
                if 'progress_bar' in ui_components:
                    if hasattr(ui_components['progress_bar'], 'value'):
                        ui_components['progress_bar'].value = 0
                    if hasattr(ui_components['progress_bar'], 'layout'):
                        ui_components['progress_bar'].layout.visibility = 'hidden'
            
            ui_components['reset_all'] = fallback_reset_progress
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default config untuk augmentation."""
        from smartcash.common.config import get_config_manager
        config_manager = get_config_manager()
        try:
            return config_manager.get_module_config('augmentation') or {}
        except Exception:
            return {}
    
    def _get_critical_components(self) -> List[str]:
        """Get critical components untuk augmentation."""
        return ['ui', 'augment_button']
    
    def _handle_reset_button(self, ui_components: Dict[str, Any], button) -> None:
        """Handle reset button untuk augmentation."""
        handle_reset_button_click(ui_components, button)
    
    def _handle_save_button(self, ui_components: Dict[str, Any], button) -> None:
        """Handle save button untuk augmentation."""
        handle_save_button_click(ui_components, button)
    
    def _handle_cleanup_button(self, ui_components: Dict[str, Any], button) -> None:
        """Handle cleanup button untuk augmentation."""
        handle_cleanup_button_click(ui_components, button)


# Global instance dan functions untuk backward compatibility
_augmentation_initializer = AugmentationInitializer()

def initialize_dataset_augmentation_ui(env=None, config=None):
    """Backward compatible function."""
    return _augmentation_initializer.initialize(env=env, config=config)