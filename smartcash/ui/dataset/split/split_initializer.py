"""
File: smartcash/ui/dataset/split/split_initializer.py
Deskripsi: Split initializer menggunakan CommonInitializer base class
"""

from typing import Dict, Any, Optional, List
import ipywidgets as widgets
from IPython.display import display

# Import base class
from smartcash.ui.utils.common_initializer import CommonInitializer

# Import handlers dan components
from smartcash.ui.dataset.split.handlers.button_handlers import setup_button_handlers
from smartcash.ui.dataset.split.handlers.slider_handlers import setup_slider_handlers
from smartcash.ui.dataset.split.components.split_form import create_split_form
from smartcash.ui.dataset.split.components.split_layout import create_split_layout
from smartcash.ui.dataset.split.handlers.defaults import get_default_split_config


class SplitInitializer(CommonInitializer):
    """
    Dataset split UI initializer menggunakan CommonInitializer base class.
    """
    
    def __init__(self):
        super().__init__(
            module_name='dataset_split',
            logger_namespace='smartcash.ui.dataset.split'
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration untuk split module."""
        try:
            return get_default_split_config()
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting default split config: {str(e)}")
            # Fallback minimal config
            return {
                'data': {
                    'train_ratio': 0.7,
                    'val_ratio': 0.2,
                    'test_ratio': 0.1,
                    'random_seed': 42,
                    'stratify': True
                }
            }
    
    def _get_critical_components(self) -> List[str]:
        """Get list of critical component keys yang harus ada."""
        return ['main_container', 'train_slider', 'val_slider', 'test_slider']
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """
        Create UI components specific untuk split module.
        
        Args:
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of UI components
        """
        try:
            # Buat form komponen
            form_components = create_split_form(config)
            
            # Buat layout utama
            layout_components = create_split_layout(form_components)
            
            # Combine semua komponen
            ui_components = {**form_components, **layout_components}
            
            # Ensure main UI component
            if 'main_container' not in ui_components:
                ui_components['main_container'] = widgets.VBox([
                    widgets.HTML("<h3>Split Dataset Configuration</h3>")
                ])
            
            # Set as main UI
            ui_components['ui'] = ui_components['main_container']
            
            return ui_components
            
        except Exception as e:
            self.logger.error(f"💥 Error membuat komponen UI: {str(e)}")
            # Return minimal UI
            error_widget = widgets.HTML(f"<div style='color:red'>Error: {str(e)}</div>")
            return {
                'main_container': error_widget,
                'ui': error_widget
            }
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], 
                             config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """
        Setup handlers specific untuk split module.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
            env: Environment context
            **kwargs: Additional parameters
            
        Returns:
            Updated UI components dictionary
        """
        setup_results = {
            'button_handlers': False,
            'slider_handlers': False
        }
        
        logger = ui_components.get('logger', self.logger)
        
        # Setup button handlers
        try:
            setup_button_handlers(ui_components)
            setup_results['button_handlers'] = True
            logger.debug("🔗 Button handlers berhasil dipasang")
        except Exception as e:
            logger.error(f"💥 Error setup button handlers: {str(e)}")
        
        # Setup slider handlers untuk auto-adjustment
        try:
            setup_slider_handlers(ui_components)
            setup_results['slider_handlers'] = True
            logger.debug("🔗 Slider handlers berhasil dipasang")
        except Exception as e:
            logger.error(f"💥 Error setup slider handlers: {str(e)}")
        
        # Store setup results untuk debugging
        ui_components['_setup_results'] = setup_results
        
        return ui_components
    
    def _additional_validation(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Additional validation specific untuk split module."""
        
        # Validate slider components
        slider_keys = ['train_slider', 'val_slider', 'test_slider']
        functional_sliders = []
        
        for slider_key in slider_keys:
            if (slider_key in ui_components and 
                ui_components[slider_key] is not None and
                hasattr(ui_components[slider_key], 'value')):
                functional_sliders.append(slider_key)
        
        # Minimal requirement: at least train slider harus ada
        if 'train_slider' not in functional_sliders:
            return {
                'valid': False,
                'message': 'Train slider tidak ditemukan atau tidak functional'
            }
        
        # Validate split ratios if sliders are available
        if len(functional_sliders) == 3:
            try:
                train_val = ui_components['train_slider'].value
                val_val = ui_components['val_slider'].value
                test_val = ui_components['test_slider'].value
                
                total = train_val + val_val + test_val
                if abs(total - 1.0) > 0.01:  # Allow small floating point errors
                    return {
                        'valid': False,
                        'message': f'Split ratios tidak valid (total: {total:.3f})'
                    }
            except Exception as e:
                return {
                    'valid': False,
                    'message': f'Error validating split ratios: {str(e)}'
                }
        
        return {
            'valid': True,
            'functional_sliders': functional_sliders,
            'total_functional': len(functional_sliders)
        }
    
    def _update_cached_config(self, new_config: Dict[str, Any]) -> None:
        """Update cached UI components dengan config baru."""
        if not self._cached_components:
            return
        
        try:
            # Update base config
            super()._update_cached_config(new_config)
            
            # Update slider values jika component ada dan config berisi data split
            if 'data' in new_config:
                data_config = new_config['data']
                
                slider_mapping = {
                    'train_ratio': 'train_slider',
                    'val_ratio': 'val_slider',
                    'test_ratio': 'test_slider'
                }
                
                for config_key, slider_key in slider_mapping.items():
                    if (config_key in data_config and 
                        slider_key in self._cached_components and
                        hasattr(self._cached_components[slider_key], 'value')):
                        self._cached_components[slider_key].value = data_config[config_key]
                        
        except Exception as e:
            # Silent fail untuk config update
            self.logger.debug(f"Config update failed: {str(e)}")
    
    def _get_return_value(self, ui_components: Dict[str, Any]) -> Any:
        """Get return value from UI components - return full components dict for split."""
        # Split module returns full components dict instead of just UI
        return ui_components
    
    def _finalize_setup(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Finalize setup dengan display UI."""
        # Call parent finalize
        super()._finalize_setup(ui_components, config)
        
        # Display UI jika ada main_container
        if 'main_container' in ui_components:
            try:
                display(ui_components['main_container'])
            except Exception as e:
                self.logger.warning(f"⚠️ Error displaying UI: {str(e)}")


# Global initializer instance
_split_initializer = None

def get_split_initializer() -> SplitInitializer:
    """Get atau create split initializer instance."""
    global _split_initializer
    if _split_initializer is None:
        _split_initializer = SplitInitializer()
    return _split_initializer
