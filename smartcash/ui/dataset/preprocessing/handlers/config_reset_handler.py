"""
File: smartcash/ui/dataset/preprocessing/handlers/config_reset_handler.py
Deskripsi: Handler khusus untuk reset konfigurasi preprocessing
"""

from typing import Dict, Any
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.dataset.preprocessing.utils import get_config_extractor

def setup_config_reset_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk reset konfigurasi preprocessing."""
    logger = ui_components.get('logger')
    config_manager = ui_components.get('config_manager')
    config_extractor = get_config_extractor(ui_components)
    
    def _on_reset_click(b):
        """Handler untuk reset konfigurasi dengan fallback handling."""
        if not config_manager:
            logger and logger.error("❌ Config manager tidak tersedia")
            update_status_panel(ui_components['status_panel'], 
                              "Config manager tidak tersedia", "error")
            return
        
        try:
            # Load saved config
            saved_config = config_manager.load_config('preprocessing')
            
            if saved_config and saved_config.get('preprocessing'):
                errors = config_extractor.apply_config_to_ui(saved_config)
                
                if errors:
                    logger and logger.warning(f"⚠️ Reset warnings: {', '.join(errors)}")
                
                config_summary = config_extractor.get_config_summary(saved_config)
                logger and logger.success("🔄 Konfigurasi berhasil direset dari file tersimpan")
                logger and logger.info(f"📋 {config_summary}")
                
                update_status_panel(ui_components['status_panel'], 
                                  "Konfigurasi berhasil direset", "success")
            else:
                # Fallback ke default config
                default_config = {
                    'preprocessing': {
                        'img_size': [640, 640],
                        'normalization': 'minmax',
                        'num_workers': 4,
                        'split': 'all'
                    }
                }
                config_extractor.apply_config_to_ui(default_config)
                
                logger and logger.info("🔄 Konfigurasi direset ke default (file tersimpan tidak ditemukan)")
                update_status_panel(ui_components['status_panel'], 
                                  "Konfigurasi direset ke default", "info")
                
        except Exception as e:
            error_msg = f"Error saat reset konfigurasi: {str(e)}"
            logger and logger.error(f"❌ {error_msg}")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['reset_button'].on_click(_on_reset_click)
    
    logger and logger.debug("✅ Config reset handler setup selesai")
    
    return ui_components