"""
File: smartcash/ui/dataset/preprocessing/handlers/event_handlers.py
Deskripsi: Main coordinator untuk semua event handlers dengan SRP pattern
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.config_handlers import ConfigHandler

def setup_all_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], 
                      config_handler: Optional[ConfigHandler] = None) -> Dict[str, Any]:
    """Setup semua handlers dengan SRP pattern dan proper error handling
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi preprocessing
        config_handler: Handler konfigurasi (opsional)
        
    Returns:
        Dictionary handlers yang telah disetup
        
    Raises:
        ValueError: Jika logger bridge tidak tersedia
    """
    logger_bridge = ui_components.get('logger_bridge')
    if not logger_bridge:
        raise ValueError("Logger bridge belum diinisialisasi di UI components")
    
    handlers = {}
    
    try:
        # Setup config handlers (save/reset)
        from .config_event_handlers import setup_config_handlers
        handlers['config'] = setup_config_handlers(ui_components, config_handler)
        
        # Setup operation handlers (preprocess/check/cleanup)
        from .operation_handlers import setup_operation_handlers
        handlers['operations'] = setup_operation_handlers(ui_components, config)
        
        # Setup confirmation handlers (dialog management)
        from .confirmation_handlers import setup_confirmation_handlers
        handlers['confirmations'] = setup_confirmation_handlers(ui_components)
        
        logger_bridge.info("✅ Semua preprocessing handlers berhasil disetup")
        return handlers
        
    except Exception as e:
        logger_bridge.error(f"❌ Error setup handlers: {str(e)}")
        raise ValueError(f"Gagal setup handlers: {str(e)}") from e