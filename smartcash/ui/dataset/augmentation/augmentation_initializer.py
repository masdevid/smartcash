"""
File: smartcash/ui/dataset/augmentation/augmentation_initializer.py
Deskripsi: Fixed initializer dengan cache management dan parameter alignment
"""

from typing import Dict, Any, List
from smartcash.ui.utils.common_initializer import CommonInitializer
from smartcash.ui.utils.ui_logger_namespace import AUGMENTATION_LOGGER_NAMESPACE, KNOWN_NAMESPACES
MODULE_LOGGER_NAME = KNOWN_NAMESPACES[AUGMENTATION_LOGGER_NAMESPACE]

class AugmentationInitializer(CommonInitializer):
    """Fixed initializer dengan cache management dan log suppression"""
    
    def __init__(self):
        super().__init__(
            module_name=MODULE_LOGGER_NAME,
            logger_namespace=AUGMENTATION_LOGGER_NAMESPACE
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Create UI components dengan aligned parameters"""
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        ui_components = create_augmentation_ui(env=env, config=config)
        ui_components.update({
            'logger_namespace': self.logger_namespace,
            'module_initialized': True,
            'augmentation_initialized': True  # Flag untuk UI detection
        })
        
        return ui_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], 
                             config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers dengan cache invalidation"""
        try:
            from smartcash.ui.dataset.augmentation.handlers.main_handler import register_all_handlers
            
            # Clear existing handlers untuk prevent double registration
            self._clear_existing_handlers(ui_components)
            
            ui_components = register_all_handlers(ui_components)
            
            # Cache management flags
            ui_components['handlers_cache_valid'] = True
            ui_components['last_config_hash'] = hash(str(config))
            
            if 'logger' in ui_components:
                ui_components['logger'].info(f"✅ Handlers registered: {ui_components.get('handlers_registered', 0)} total")
            
            return ui_components
            
        except Exception as e:
            if 'logger' in ui_components:
                ui_components['logger'].error(f"❌ Handler setup error: {str(e)}")
            return ui_components
    
    def _clear_existing_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Clear existing handlers untuk prevent conflicts"""
        button_keys = ['augment_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
        
        for key in button_keys:
            button = ui_components.get(key)
            if button and hasattr(button, '_click_handlers'):
                try:
                    button._click_handlers.callbacks.clear()
                except Exception:
                    pass
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default config dengan aligned parameters"""
        return {
            'data': {'dir': 'data'},
            'augmentation': {
                'num_variations': 2, 'target_count': 500, 'output_prefix': 'aug_', 'balance_classes': False,
                'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2,
                'types': ['combined'], 'target_split': 'train', 'intensity': 0.7,
                'output_dir': 'data/augmented'
            },
            'preprocessing': {'output_dir': 'data/preprocessed'}
        }
    
    def _get_critical_components(self) -> List[str]:
        """Critical components dengan aligned names"""
        return [
            'ui', 'augment_button', 'check_button', 'save_button', 'reset_button',
            'num_variations', 'target_count', 'augmentation_types', 'target_split',
            'tracker', 'log_output'
        ]
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation dengan cache check"""
        # Parent validation
        parent_result = super()._additional_validation(ui_components)
        if not parent_result.get('valid', True):
            return parent_result
        
        # Cache validation
        cache_issues = []
        
        # Check handler cache validity
        if not ui_components.get('handlers_cache_valid', False):
            cache_issues.append("Handler cache invalid")
        
        # Check config hash consistency
        current_config = ui_components.get('config', {})
        stored_hash = ui_components.get('last_config_hash')
        current_hash = hash(str(current_config))
        
        if stored_hash and stored_hash != current_hash:
            cache_issues.append("Config hash mismatch")
            # Auto-fix: update cache
            ui_components['last_config_hash'] = current_hash
        
        if cache_issues:
            return {
                'valid': True,
                'message': f'Cache inconsistencies detected: {", ".join(cache_issues)} - auto-fixing',
                'cache_issues': cache_issues
            }
        
        return {'valid': True, 'message': 'All validations passed'}
    
    def _update_cached_config(self, new_config: Dict[str, Any]) -> None:
        """Enhanced cache update dengan invalidation"""
        if not self._cached_components:
            return
        
        try:
            # Update config dengan parameter alignment
            merged_config = self._get_merged_config(new_config)
            self._cached_components['config'] = merged_config
            
            # Invalidate handler cache jika config berubah significantly
            old_hash = self._cached_components.get('last_config_hash')
            new_hash = hash(str(merged_config))
            
            if old_hash != new_hash:
                self._cached_components['handlers_cache_valid'] = False
                self._cached_components['last_config_hash'] = new_hash
                
                # Re-setup handlers dengan new config
                self._setup_module_handlers(self._cached_components, merged_config)
            
            # Apply config ke UI widgets
            self._apply_config_to_widgets(merged_config)
            
        except Exception as e:
            logger = self._cached_components.get('logger', self.logger)
            logger.warning(f"⚠️ Cache refresh error: {str(e)}")
    
    def _apply_config_to_widgets(self, config: Dict[str, Any]) -> None:
        """Apply config ke UI widgets dengan aligned parameters"""
        if not self._cached_components:
            return
        
        aug_config = config.get('augmentation', {})
        
        # Widget mapping dengan safe updates
        widget_mappings = {
            'num_variations': aug_config.get('num_variations', 2),
            'target_count': aug_config.get('target_count', 500),
            'fliplr': aug_config.get('fliplr', 0.5),
            'degrees': aug_config.get('degrees', 10),
            'translate': aug_config.get('translate', 0.1),
            'scale': aug_config.get('scale', 0.1),
            'hsv_h': aug_config.get('hsv_h', 0.015),
            'hsv_s': aug_config.get('hsv_s', 0.7),
            'brightness': aug_config.get('brightness', 0.2),
            'contrast': aug_config.get('contrast', 0.2)
        }
        
        for widget_key, value in widget_mappings.items():
            widget = self._cached_components.get(widget_key)
            if widget and hasattr(widget, 'value'):
                try:
                    widget.value = value
                except Exception:
                    pass
        
        # Multi-select widgets
        aug_types_widget = self._cached_components.get('augmentation_types')
        if aug_types_widget and hasattr(aug_types_widget, 'value'):
            try:
                aug_types_widget.value = list(aug_config.get('types', ['combined']))
            except Exception:
                pass
        
        target_split_widget = self._cached_components.get('target_split')
        if target_split_widget and hasattr(target_split_widget, 'value'):
            try:
                target_split_widget.value = aug_config.get('target_split', 'train')
            except Exception:
                pass
    
    def _setup_log_suppression(self) -> None:
        """Enhanced log suppression untuk augmentation"""
        super()._setup_log_suppression()
        
        # Augmentor-specific suppressions
        augmentor_loggers = [
            'smartcash.dataset.augmentor', 'smartcash.dataset.augmentor.core',
            'smartcash.dataset.augmentor.utils', 'smartcash.dataset.augmentor.strategies',
            'smartcash.dataset.augmentor.communicator', 'albumentations', 'cv2'
        ]
        
        import logging
        for logger_name in augmentor_loggers:
            try:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.CRITICAL)
                logger.propagate = False
                # Clear handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
            except Exception:
                pass
    
    def invalidate_cache(self) -> None:
        """Invalidate semua cache untuk force refresh"""
        if self._cached_components:
            self._cached_components['handlers_cache_valid'] = False
            self._cached_components.pop('last_config_hash', None)
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status untuk debugging"""
        if not self._cached_components:
            return {'cached': False}
        
        return {
            'cached': True,
            'handlers_valid': self._cached_components.get('handlers_cache_valid', False),
            'config_hash': self._cached_components.get('last_config_hash'),
            'components_count': len(self._cached_components),
            'critical_widgets': [key for key in self._get_critical_components() if key in self._cached_components]
        }

# Global instance dengan cache management
_aug_initializer = None

def get_aug_initializer():
    global _aug_initializer
    if _aug_initializer is None:
        _aug_initializer = AugmentationInitializer()
    return _aug_initializer

# Public API dengan cache control
init_augmentation = lambda env=None, config=None, force=False: get_aug_initializer().initialize(env=env, config=config, force_refresh=force)
reset_augmentation = lambda: get_aug_initializer().reset_module()
invalidate_aug_cache = lambda: get_aug_initializer().invalidate_cache()
get_aug_cache_status = lambda: get_aug_initializer().get_cache_status()