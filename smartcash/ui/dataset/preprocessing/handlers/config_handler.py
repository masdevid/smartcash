"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Enhanced config handler dengan service layer integration dan validation
"""

from typing import Dict, Any
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.common.config.manager import get_config_manager

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup config handlers dengan service integration."""
    config_extractor = get_config_extractor(ui_components)
    config_manager = get_config_manager()
    logger = ui_components.get('logger')
    
    def save_config_action(button=None) -> None:
        """Save current config dengan validation."""
        try:
            # Extract current parameters
            params = config_extractor.extract_processing_parameters()
            
            if params['valid']:
                # Save config
                preprocessing_config = {'preprocessing': params['config']}
                save_success = config_manager.save_config(preprocessing_config, 'preprocessing')
                
                if save_success:
                    logger and logger.success("💾 Konfigurasi preprocessing tersimpan")
                else:
                    logger and logger.error("❌ Gagal menyimpan konfigurasi")
            else:
                logger and logger.warning("⚠️ Konfigurasi tidak valid, tidak disimpan")
                
        except Exception as e:
            logger and logger.error(f"💥 Error saving config: {str(e)}")
    
    def reset_config_action(button=None) -> None:
        """Reset config ke default values."""
        try:
            # Load default config
            default_config = {
                'preprocessing': {
                    'img_size': [640, 640],
                    'normalize': True,
                    'normalization_method': 'minmax',
                    'num_workers': 4,
                    'split': 'all'
                }
            }
            
            # Apply ke UI
            config_extractor.apply_config_to_ui(default_config)
            logger and logger.info("🔄 Konfigurasi direset ke default")
            
        except Exception as e:
            logger and logger.error(f"💥 Error resetting config: {str(e)}")
    
    # Register handlers
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(save_config_action)
    
    if 'reset_button' in ui_components:
        ui_components['reset_button'].on_click(reset_config_action)
    
    # Apply initial config jika ada
    if config:
        try:
            config_extractor.apply_config_to_ui(config)
            logger and logger.info("⚙️ Konfigurasi awal diterapkan ke UI")
        except Exception as e:
            logger and logger.warning(f"⚠️ Error applying initial config: {str(e)}")
    
    ui_components['save_config'] = save_config_action
    ui_components['reset_config'] = reset_config_action
    
    return ui_components