"""
File: smartcash/ui/dataset/preprocessing/handlers/config_save_handler.py
Deskripsi: Handler khusus untuk save konfigurasi preprocessing
"""

from typing import Dict, Any
from smartcash.common.environment import get_environment_manager
from smartcash.ui.components.status_panel import update_status_panel
from smartcash.ui.dataset.preprocessing.utils import get_config_extractor

def setup_config_save_handler(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup handler untuk save konfigurasi preprocessing."""
    logger = ui_components.get('logger')
    config_manager = ui_components.get('config_manager')
    env_manager = get_environment_manager()
    config_extractor = get_config_extractor(ui_components)
    
    def _on_save_click(b):
        """Handler untuk save konfigurasi dengan comprehensive validation."""
        if not config_manager:
            logger and logger.error("❌ Config manager tidak tersedia")
            update_status_panel(ui_components['status_panel'], 
                              "Config manager tidak tersedia", "error")
            return
        
        try:
            current_config = config_extractor.get_current_ui_config()
            
            # Validasi konfigurasi sebelum save
            validation_errors = config_extractor.validate_config(current_config)
            if validation_errors:
                error_msg = f"❌ Konfigurasi tidak valid: {', '.join(validation_errors)}"
                logger and logger.error(error_msg)
                update_status_panel(ui_components['status_panel'], error_msg, "error")
                return
            
            # Save konfigurasi
            success = config_manager.save_config(current_config, 'preprocessing')
            
            if success:
                # Log konfigurasi yang disimpan
                config_summary = config_extractor.get_config_summary(current_config)
                logger and logger.success(f"💾 Konfigurasi preprocessing berhasil disimpan")
                logger and logger.info(f"📋 {config_summary}")
                
                update_status_panel(ui_components['status_panel'], 
                                  "Konfigurasi preprocessing berhasil disimpan", "success")
                
                # Show save confirmation di Drive jika mounted
                if env_manager.is_drive_mounted:
                    logger and logger.info("☁️ Konfigurasi tersinkronisasi dengan Google Drive")
            else:
                logger and logger.error("❌ Gagal menyimpan konfigurasi")
                update_status_panel(ui_components['status_panel'], 
                                  "Gagal menyimpan konfigurasi", "error")
                
        except Exception as e:
            error_msg = f"Error saat menyimpan konfigurasi: {str(e)}"
            logger and logger.error(f"❌ {error_msg}")
            update_status_panel(ui_components['status_panel'], error_msg, "error")
    
    # Setup event handler
    ui_components['save_button'].on_click(_on_save_click)
    
    logger and logger.debug("✅ Config save handler setup selesai")
    
    return ui_components