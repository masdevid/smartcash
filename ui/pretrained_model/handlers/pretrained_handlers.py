"""
File: smartcash/ui/pretrained_model/handlers/pretrained_handlers.py
Deskripsi: Unified handlers untuk pretrained model dengan UI integration
"""

from typing import Dict, Any

def setup_pretrained_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan UI integration"""
    
    # Import handlers
    from smartcash.ui.pretrained_model.handlers.download_handler import setup_download_handler
    from smartcash.ui.pretrained_model.handlers.check_handler import setup_check_handler
    
    # Setup handlers
    setup_download_handler(ui_components, config)
    
    # Auto-check handler jika diaktifkan
    ui_components.get('auto_check_enabled', False) and setup_check_handler(ui_components, config)
    
    return ui_components