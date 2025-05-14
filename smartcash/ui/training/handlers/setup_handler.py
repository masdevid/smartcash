"""
File: smartcash/ui/training/handlers/setup_handler.py
Deskripsi: Setup handler untuk komponen UI training
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.common.config.manager import get_config_manager
from smartcash.common.logger import get_logger
from smartcash.ui.training.handlers.training_handler_utils import ensure_ui_persistence
from smartcash.ui.training.handlers.training_info_handler import update_training_info
from smartcash.ui.training.handlers.button_event_handlers import register_button_handlers

def setup_training_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI training.
    
    Args:
        ui_components: Komponen UI
        env: Environment manager
        config: Konfigurasi model
        
    Returns:
        Dict berisi komponen UI dengan handler terpasang
    """
    try:
        # Dapatkan logger
        logger = ui_components.get('logger', None) or get_logger("training_ui")
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager()
        
        # Log informasi setup
        logger.info("🔧 Memulai setup handler training...")
        
        # Daftarkan handler untuk tombol
        register_button_handlers(ui_components, logger)
        
        # Update informasi training saat pertama kali
        update_training_info(ui_components, logger)
        
        # Pastikan persistensi UI components
        ensure_ui_persistence(ui_components, 'training')
        
        logger.info("✅ Setup handler training berhasil")
        return ui_components
    
    except Exception as e:
        # Fallback jika terjadi error
        logger = get_logger("training_ui")
        logger.error(f"❌ Error saat setup training handlers: {str(e)}")
        
        # Tampilkan pesan error
        if 'status_panel' in ui_components:
            with ui_components['status_panel']:
                ui_components['status_panel'].clear_output()
                display(HTML(f"""
                <div style="color:#e74c3c">
                    ❌ Error saat setup training handlers: {str(e)}
                </div>
                """))
        
        return ui_components
