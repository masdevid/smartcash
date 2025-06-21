# File: smartcash/ui/pretrained/pretrained_init.py
"""
File: smartcash/ui/pretrained/pretrained_init.py
Deskripsi: Initializer untuk pretrained models - Fixed version berdasarkan preprocessing pattern
"""

import logging
import sys
from typing import Dict, Any, List, Optional

from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class PretrainedInitializer(CommonInitializer):
    """🤖 Pretrained models initializer dengan fixed UI creation pattern"""
    
    def __init__(self):
        super().__init__(
            module_name='pretrained_models',
            config_handler_class=PretrainedConfigHandler
        )
    
    def _create_ui_components(self, config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Buat komponen UI untuk modul pretrained
        
        Args:
            config: Konfigurasi untuk inisialisasi UI
            env: Environment (opsional)
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary berisi komponen UI yang dibuat
        """
        try:
            from smartcash.ui.pretrained.components.ui_components import create_pretrained_ui_components
            
            # Buat komponen UI dengan konfigurasi yang sesuai
            ui_components = create_pretrained_ui_components(
                env=env, 
                config={'pretrained_models': config.get('pretrained_models', {})}
            )
            
            # Validasi komponen kritis
            critical_components = self._get_critical_components()
            missing_components = [comp for comp in critical_components if comp not in ui_components]
            
            if missing_components:
                logger.warning(f"Komponen kritis tidak ditemukan: {missing_components}")
            
            # Setup event handlers
            try:
                self._setup_event_handlers(ui_components, config)
                ui_components['ui_initialized'] = True
            except Exception as e:
                error_msg = f"Gagal menyiapkan event handlers: {str(e)}"
                logger.exception(error_msg, exc_info=True)
                ui_components['ui_initialized'] = False
                ui_components['error'] = error_msg
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Gagal membuat komponen UI pretrained: {str(e)}"
            logger.exception(error_msg, exc_info=True)
            return self._create_fallback_ui(
                error_msg,
                exc_info=sys.exc_info()
            )
    
    def _setup_event_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Setup event handlers untuk UI components"""
        try:
            from smartcash.ui.pretrained.handlers.pretrained_handlers import (
                _handle_download_sync,
                _setup_config_handlers,
                _setup_operation_handlers
            )
            
            # Setup config handlers (save/reset)
            _setup_config_handlers(ui_components)
            
            # Setup operation handlers (download/sync)
            _setup_operation_handlers(ui_components)
            
            # Connect download/sync button
            if download_sync_button := ui_components.get('download_sync_button'):
                download_sync_button.on_click(lambda _: _handle_download_sync(ui_components))
            
            logger.info("Event handlers setup successfully")
            
        except Exception as e:
            logger.warning(f"Error setting up handlers: {str(e)}")
            raise  # Re-raise to be handled by the caller
    
    def _create_config_handler(self, module_name: str = None, parent_module: str = None):
        """Create config handler dengan proper parameters"""
        try:
            from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
            return PretrainedConfigHandler(
                module_name=module_name or self.module_name,
                parent_module=parent_module
            )
        except Exception as e:
            logger.warning(f"Error creating config handler: {str(e)}")
            return None
    
    def _load_initial_config(self, ui_components: Dict[str, Any], config_handler) -> None:
        """Load dan apply initial config"""
        try:
            if not config_handler:
                logger.warning("No config handler, skipping config load")
                return
            
            # Load config dengan fallback ke defaults
            try:
                loaded_config = config_handler.get_default_config()
                logger.info("Default config loaded")
                
                # Ensure pretrained_type is a string, not a list
                if 'pretrained_models' in loaded_config and 'pretrained_type' in loaded_config['pretrained_models']:
                    pretrained_type = loaded_config['pretrained_models']['pretrained_type']
                    if isinstance(pretrained_type, (list, tuple)) and len(pretrained_type) > 0:
                        loaded_config['pretrained_models']['pretrained_type'] = str(pretrained_type[0])
                        logger.debug(f"Converted pretrained_type from list to string: {loaded_config['pretrained_models']['pretrained_type']}")
                    
            except Exception as e:
                logger.warning(f"Error loading defaults: {str(e)}")
                loaded_config = self._get_default_config()
            
            # Update UI dengan loaded config
            if loaded_config and ui_components.get('ui_initialized'):
                try:
                    config_handler.update_ui(ui_components, loaded_config)
                    ui_components['config'] = loaded_config
                    logger.info("Config loaded dan UI updated")
                except Exception as e:
                    logger.warning(f"Error updating UI with config: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error loading initial config: {str(e)}", exc_info=True)
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Mengembalikan konfigurasi default untuk pretrained models"""
        return {
            'pretrained_models': {
                'models_dir': '/content/models',
                'drive_models_dir': '/content/drive/MyDrive/SmartCash/models',
                'pretrained_type': 'yolov5s',
                'auto_download': False,
                'sync_drive': True
            }
        }
    
    def _setup_module_handlers(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs) -> Dict[str, Any]:
        """Setup module-specific handlers"""
        try:
            from smartcash.ui.pretrained.handlers.pretrained_handlers import setup_pretrained_handlers
            return setup_pretrained_handlers(ui_components, config, env=env, **kwargs)
        except Exception as e:
            logger.error(f"Gagal setup module handlers: {str(e)}")
            return ui_components
    
    def _get_critical_components(self) -> List[str]:
        """Critical components yang harus ada"""
        return [
            'ui', 'download_sync_button', 'save_button', 'reset_button',
            'log_output', 'confirmation_area', 'status'
        ]
    
    def _post_initialization_hook(self, ui_components: Dict[str, Any], config: Dict[str, Any], env=None, **kwargs):
        """Post-init hook untuk additional setup"""
        super()._post_initialization_hook(ui_components, config, env, **kwargs)
        
        # Auto-check model availability jika enabled
        try:
            pretrained_config = config.get('pretrained_models', {})
            if pretrained_config.get('auto_download', False):
                logger.info("Auto-download enabled, checking models...")
                # Implement model checking logic here
        except Exception as e:
            logger.warning(f"Post-init hook error: {str(e)}")

    # Method _create_fallback_ui sudah dipindahkan ke CommonInitializer
    # untuk memastikan konsistensi di semua modul

# Global instance
_pretrained_initializer = PretrainedInitializer()

def initialize_pretrained_ui(env=None, config=None, **kwargs):
    """Inisialisasi UI untuk modul pretrained
    
    Args:
        env: Informasi environment (opsional)
        config: Konfigurasi awal (opsional)
        **kwargs: Parameter tambahan
        
    Returns:
        Dictionary berisi komponen UI yang diinisialisasi
    """
    try:
        result = _pretrained_initializer.initialize(env=env, config=config, **kwargs)
        # Pastikan kita hanya mengembalikan UI components
        if isinstance(result, dict) and 'ui' in result:
            return result['ui']
        return result
    except Exception as e:
        error_msg = f"Gagal menginisialisasi UI pretrained: {str(e)}"
        logger.exception(error_msg, exc_info=True)
        
        # Buat fallback UI dengan traceback lengkap
        fallback_ui = _pretrained_initializer._create_fallback_ui(
            error_msg,
            exc_info=sys.exc_info()
        )
        
        # Kembalikan hanya UI-nya, bukan dictionary lengkap
        return fallback_ui.get('ui', fallback_ui)