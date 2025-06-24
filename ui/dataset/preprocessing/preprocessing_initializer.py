"""
File: smartcash/ui/dataset/preprocessing/preprocessing_initializer.py
Deskripsi: Fixed preprocessing initializer dengan implementasi abstract methods yang benar
"""

from typing import Dict, Any, List
from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.dataset.preprocessing.handlers.config_handler import PreprocessingConfigHandler


class PreprocessingInitializer(CommonInitializer):
    """Fixed preprocessing initializer dengan proper abstract method implementation"""
    
    def __init__(self):
        super().__init__(
            module_name='preprocessing',
            config_handler_class=PreprocessingConfigHandler,
            parent_module='dataset'
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Implementasi wajib: Create UI components dengan proper error handling"""
        try:
            from .components.ui_components import create_preprocessing_main_ui
            
            # Create UI components dengan config
            ui_components = create_preprocessing_main_ui(config or {})
            
            # Validate bahwa UI components berupa dict dan tidak kosong
            if not isinstance(ui_components, dict) or not ui_components:
                raise ValueError("create_preprocessing_main_ui mengembalikan nilai invalid")
            
            # Tambahkan metadata yang diperlukan
            ui_components.update({
                'module_name': 'preprocessing',
                'data_dir': config.get('data', {}).get('dir', 'data'),
                'api_integration_enabled': True,
                'progress_tracking_enabled': True
            })
            
            self.logger.debug(f"✅ UI components created: {list(ui_components.keys())}")
            return ui_components
            
        except ImportError as e:
            self.logger.error(f"❌ Import error: {str(e)}")
            raise ImportError(f"Tidak dapat import UI components: {str(e)}")
        except Exception as e:
            self.logger.error(f"❌ Error creating UI components: {str(e)}")
            raise
    
    def _get_critical_components(self) -> List[str]:
        """Implementasi wajib: Daftar komponen kritis untuk preprocessing"""
        return [
            'header',
            'status_panel', 
            'log_output',
            'action_buttons',
            'input_form'
        ]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Implementasi wajib: Default config dari handlers/defaults.py"""
        from .handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                             env=None, **kwargs) -> Dict[str, Any]:
        """Setup handlers khusus preprocessing dengan error handling"""
        try:
            from .handlers.preprocessing_handlers import setup_preprocessing_handlers
            
            # Setup handlers dengan proper error handling
            result = setup_preprocessing_handlers(ui_components, config, env)
            
            # Load dan update UI jika berhasil
            self._load_and_update_ui(ui_components)
            
            self.logger.debug("✅ Preprocessing handlers setup berhasil")
            return result
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Import error pada handlers: {str(e)}")
            return ui_components  # Return original jika handlers gagal
        except Exception as e:
            self.logger.warning(f"⚠️ Error setting up handlers: {str(e)}")
            return ui_components  # Return original jika handlers gagal
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], 
                                env=None, **kwargs) -> None:
        """Hook setelah initialization selesai"""
        try:
            # Setup auto-save jika diperlukan
            config_handler = ui_components.get('config_handler')
            if config_handler and hasattr(config_handler, 'setup_auto_save'):
                config_handler.setup_auto_save(ui_components)
            
            # Log final status
            self.logger.info(f"🎉 Preprocessing UI siap digunakan dengan {len(ui_components)} komponen")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error dalam post-initialization: {str(e)}")
    
    def _load_and_update_ui(self, ui_components: Dict[str, Any]) -> None:
        """Load config dan update UI components"""
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                self.logger.debug("ℹ️ Config handler tidak tersedia")
                return
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            # Load dan update config
            loaded_config = config_handler.load_config()
            if loaded_config:
                config_handler.update_ui(ui_components, loaded_config)
                self.logger.debug("✅ Config loaded dan UI updated")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading/updating UI: {str(e)}")


# Entry point function
def initialize_preprocessing_ui(config=None, env=None, **kwargs):
    """
    Entry point untuk inisialisasi preprocessing UI.
    
    Args:
        config: Konfigurasi awal (optional)
        env: Environment info seperti 'colab' (optional)
        **kwargs: Parameter tambahan
        
    Returns:
        UI components dict atau fallback UI jika gagal
    """
    return _preprocessing_initializer.initialize(config=config, env=env, **kwargs)


# Global instance
_preprocessing_initializer = PreprocessingInitializer()