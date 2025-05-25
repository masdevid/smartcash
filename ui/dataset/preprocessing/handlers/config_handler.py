"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Fixed config handler dengan status panel updates dan clear outputs
"""

from typing import Dict, Any
from smartcash.ui.dataset.preprocessing.utils.config_extractor import get_config_extractor
from smartcash.common.config.manager import get_config_manager

def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Setup config handlers dengan status panel integration."""
    config_extractor = get_config_extractor(ui_components)
    config_manager = get_config_manager()
    logger = ui_components.get('logger')
    
    def save_config_action(button=None) -> None:
        """Save current config dengan validation dan status update."""
        try:
            # Clear outputs
            _clear_ui_outputs(ui_components)
            
            logger and logger.info("💾 Menyimpan konfigurasi preprocessing")
            
            # Extract current parameters
            params = config_extractor.extract_processing_parameters()
            
            if params['valid']:
                # Save config
                preprocessing_config = {'preprocessing': params['config']}
                save_success = config_manager.save_config(preprocessing_config, 'preprocessing')
                
                if save_success:
                    logger and logger.success("✅ Konfigurasi preprocessing tersimpan")
                    _update_status_panel(ui_components, "💾 Konfigurasi berhasil disimpan", "success")
                else:
                    logger and logger.error("❌ Gagal menyimpan konfigurasi")
                    _update_status_panel(ui_components, "❌ Gagal menyimpan konfigurasi", "error")
            else:
                logger and logger.warning("⚠️ Konfigurasi tidak valid, tidak disimpan")
                _update_status_panel(ui_components, "⚠️ Konfigurasi tidak valid", "warning")
                
        except Exception as e:
            error_msg = f"Error saving config: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            _update_status_panel(ui_components, f"❌ {error_msg}", "error")
    
    def reset_config_action(button=None) -> None:
        """Reset config ke default values dengan status update."""
        try:
            # Clear outputs
            _clear_ui_outputs(ui_components)
            
            logger and logger.info("🔄 Mereset konfigurasi ke default")
            
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
            logger and logger.success("✅ Konfigurasi direset ke default")
            _update_status_panel(ui_components, "🔄 Konfigurasi direset ke default", "info")
            
        except Exception as e:
            error_msg = f"Error resetting config: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            _update_status_panel(ui_components, f"❌ {error_msg}", "error")
    
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

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs untuk fresh display."""
    for output_key in ['log_output', 'status', 'confirmation_area']:
        if output_key in ui_components and hasattr(ui_components[output_key], 'clear_output'):
            ui_components[output_key].clear_output(wait=True)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan pesan."""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components:
        update_status_panel(ui_components['status_panel'], message, status_type)