# File: smartcash/ui/hyperparameters/handlers/config_handler.py
# Deskripsi: Handler untuk konfigurasi hyperparameters yang disederhanakan

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import BaseConfigHandler
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class HyperparametersConfigHandler(BaseConfigHandler):
    """Handler untuk konfigurasi hyperparameters dengan simplified structure"""
    
    def __init__(self, module_name: str = 'hyperparameters', 
                 config_filename: str = 'hyperparameters_config.yaml'):
        super().__init__(module_name, config_filename)
        self.config_type = 'hyperparameters'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari UI components"""
        from .config_extractor import extract_hyperparameters_config
        return extract_hyperparameters_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        from .config_updater import update_hyperparameters_ui
        update_hyperparameters_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Ambil default configuration untuk hyperparameters"""
        from .defaults import get_default_hyperparameters_config
        return get_default_hyperparameters_config()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validasi konfigurasi hyperparameters"""
        try:
            if not isinstance(config, dict):
                return False
                
            # Validasi section yang diperlukan
            required_sections = ['training', 'optimizer', 'scheduler', 'loss', 'early_stopping', 'checkpoint', 'model']
            
            for section in required_sections:
                if section not in config:
                    logger.warning(f"⚠️ Missing section: {section}")
                    return False
                    
            # Validasi training parameters
            training = config.get('training', {})
            if not all(key in training for key in ['epochs', 'batch_size', 'learning_rate']):
                logger.warning("⚠️ Missing required training parameters")
                return False
                
            logger.info("✅ Config hyperparameters valid")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating config: {str(e)}")
            return False