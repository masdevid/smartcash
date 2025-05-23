"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Updated download action dengan observer progress tracking
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.ui_validator import validate_download_params
from smartcash.ui.dataset.download.utils.confirmation_dialog import show_download_confirmation
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi aksi download dengan observer progress."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🚀 Memulai proses download dataset")
    
    disable_download_buttons(ui_components, True)
    
    try:
        # Reset log output
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        # Validasi parameter
        validation_result = validate_download_params(ui_components)
        if not validation_result['valid']:
            if logger:
                logger.error(f"❌ {validation_result['message']}")
            disable_download_buttons(ui_components, False)
            return
        
        # Tampilkan konfirmasi download
        show_download_confirmation(ui_components, validation_result['params'])
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat persiapan download: {str(e)}")
        disable_download_buttons(ui_components, False)