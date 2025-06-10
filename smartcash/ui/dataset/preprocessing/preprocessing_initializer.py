"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Fixed preprocessing initializer dengan critical components yang tepat dan error handling yang robust
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler
from smartcash.ui.dataset.preprocessing.components.ui_components import create_preprocessing_main_ui
from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import setup_preprocessing_handlers

class PreprocessingInitializer(CommonInitializer):
    """🎯 Fixed preprocessing initializer dengan critical components validation yang robust"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """🏗️ Create UI components dengan validation yang proper"""
        try:
            ui_components = create_preprocessing_main_ui(config)
            
            # PENTING: Validate critical components ada dan accessible
            critical_components = self._get_critical_components()
            missing_components = []
            
            for component_name in critical_components:
                if component_name not in ui_components:
                    missing_components.append(component_name)
                elif ui_components[component_name] is None:
                    missing_components.append(f"{component_name} (None)")
            
            if missing_components:
                error_msg = f"❌ Missing critical components: {', '.join(missing_components)}"
                self.logger.error(error_msg)
                # Return fallback dengan semua required components
                return self._create_fallback_components(config, error_msg)
            
            # Update dengan metadata
            ui_components.update({
                'preprocessing_initialized': True,
                'module_name': 'preprocessing',
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'components_validated': True
            })
            
            self.logger.info(f"✅ UI components created successfully dengan {len(ui_components)} komponen")
            return ui_components
            
        except Exception as e:
            error_msg = f"❌ Error creating UI components: {str(e)}"
            self.logger.error(error_msg)
            return self._create_fallback_components(config, error_msg)
    
    def _create_fallback_components(self, config: Dict[str, Any], error_msg: str) -> Dict[str, Any]:
        """🚨 Create fallback components dengan semua critical components yang diperlukan"""
        import ipywidgets as widgets
        
        fallback_components = {
            # CRITICAL COMPONENTS
            'ui': widgets.VBox([
                widgets.HTML(f"<div style='padding: 15px; background: #ffebee; border-left: 4px solid #f44336;'>"
                           f"<h4 style='color: #c62828; margin: 0 0 10px 0;'>⚠️ UI Initialization Error</h4>"
                           f"<p style='margin: 0; color: #d32f2f;'>{error_msg}</p>"
                           f"<small style='color: #666;'>Menggunakan fallback mode</small></div>")
            ]),
            'preprocess_button': widgets.Button(description="🚀 Preprocessing (Disabled)", disabled=True, button_style='danger'),
            'check_button': widgets.Button(description="🔍 Check (Disabled)", disabled=True, button_style='warning'),
            'cleanup_button': widgets.Button(description="🧹 Cleanup (Disabled)", disabled=True, button_style='warning'),
            'save_button': widgets.Button(description="💾 Save (Disabled)", disabled=True),
            'reset_button': widgets.Button(description="🔄 Reset (Disabled)", disabled=True),
            'log_output': widgets.Output(),
            'status_panel': widgets.HTML(f"<div style='color: #d32f2f;'>❌ {error_msg}</div>"),
            
            # ADDITIONAL COMPONENTS
            'confirmation_area': widgets.Output(),
            'progress_tracker': None,
            
            # METADATA
            'module_name': 'preprocessing',
            'data_dir': config.get('data', {}).get('dir', 'data'),
            'fallback_mode': True,
            'error_message': error_msg
        }
        
        return fallback_components
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """🔧 Setup handlers dengan validation dan fallback handling"""
        try:
            # Validate components sebelum setup handlers
            if ui_components.get('fallback_mode'):
                self.logger.warning("⚠️ Running in fallback mode, handlers setup skipped")
                return ui_components
            
            result = setup_preprocessing_handlers(ui_components, config, env)
            self._load_and_update_ui(ui_components)
            
            self.logger.info("✅ Module handlers setup completed")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error setup handlers: {str(e)}")
            # Don't fail completely, return components as-is
            return ui_components
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]):
        """📂 Load config dan update UI dengan error handling yang robust"""
        try:
            if ui_components.get('fallback_mode'):
                self.logger.warning("⚠️ Fallback mode active, skipping config load")
                return
            
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                self.logger.warning("⚠️ No config handler available")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            loaded_config = config_handler.load_config()
            if loaded_config:
                config_handler.update_ui(ui_components, loaded_config)
                ui_components['config'] = loaded_config
                self.logger.info("✅ Config loaded and UI updated successfully")
            else:
                self.logger.warning("⚠️ No config loaded, using defaults")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading config: {str(e)}")
            # Don't fail completely, just log the warning
    
    def _get_default_config(self) -> Dict[str, Any]:
        """📋 Get default config dengan error handling"""
        try:
            from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
            return get_default_preprocessing_config()
        except Exception as e:
            self.logger.error(f"❌ Error loading default config: {str(e)}")
            return {
                'preprocessing': {
                    'enabled': True,
                    'target_splits': ['train', 'valid'],
                    'normalization': {'method': 'minmax', 'target_size': [640, 640]},
                    'validation': {'enabled': True}
                },
                'performance': {'batch_size': 32}
            }
    
    def _get_critical_components(self) -> List[str]:
        """📝 Define critical components yang HARUS ada untuk proper functioning"""
        return [
            'ui',                    # Main UI container
            'preprocess_button',     # Primary operation button
            'check_button',         # Validation button
            'cleanup_button',       # Cleanup button
            'save_button',          # Config save button
            'reset_button',         # Config reset button
            'log_output',           # Log output widget
            'status_panel'          # Status display
        ]

# Global instance
_preprocessing_initializer = PreprocessingInitializer()

def initialize_preprocessing_ui(env=None, config=None, **kwargs):
    """🏭 Factory function untuk preprocessing UI dengan robust error handling"""
    try:
        return _preprocessing_initializer.initialize(env=env, config=config, **kwargs)
    except Exception as e:
        print(f"❌ Fatal error initializing preprocessing UI: {str(e)}")
        # Return minimal fallback
        import ipywidgets as widgets
        return {
            'ui': widgets.VBox([
                widgets.HTML(f"<div style='color: #d32f2f; padding: 20px;'>"
                           f"<h3>❌ Preprocessing UI Initialization Failed</h3>"
                           f"<p>Error: {str(e)}</p>"
                           f"<p>Please check the console for more details.</p></div>")
            ]),
            'error': str(e),
            'fallback_mode': True
        }