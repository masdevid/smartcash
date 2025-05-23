"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Fixed download action dengan validasi dan error handling yang robust
"""

from typing import Dict, Any

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi aksi download dengan validasi dan error handling yang diperbaiki."""
    logger = ui_components.get('logger')
    if logger:
        logger.info("🚀 Memulai proses download dataset")
    
    # Disable buttons
    _disable_all_buttons(ui_components, True)
    
    try:
        # Reset log output
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        
        # Clear confirmation area
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
        
        # Validasi parameter dengan error handling
        validation_result = _validate_params_safe(ui_components)
        if not validation_result['valid']:
            if logger:
                logger.error(f"❌ {validation_result['message']}")
            _disable_all_buttons(ui_components, False)
            return
        
        # Log parameter yang akan digunakan
        params = validation_result['params']
        if logger:
            logger.info(f"📋 Parameter download:")
            logger.info(f"   • Workspace: {params['workspace']}")
            logger.info(f"   • Project: {params['project']}")
            logger.info(f"   • Version: {params['version']}")
            logger.info(f"   • Output: {params['output_dir']}")
        
        # Tampilkan konfirmasi download
        _show_download_confirmation_safe(ui_components, params)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat persiapan download: {str(e)}")
        _disable_all_buttons(ui_components, False)

def _validate_params_safe(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi parameter dengan error handling yang aman."""
    try:
        from smartcash.ui.dataset.download.utils.ui_validator import validate_download_params
        return validate_download_params(ui_components)
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Error validasi: {str(e)}")
        return {
            'valid': False,
            'message': f"Error validasi parameter: {str(e)}",
            'params': {}
        }

def _show_download_confirmation_safe(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Tampilkan konfirmasi dengan error handling."""
    try:
        from smartcash.ui.dataset.download.utils.confirmation_dialog import show_download_confirmation
        show_download_confirmation(ui_components, params)
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Error konfirmasi dialog: {str(e)}")
        # Fallback: langsung execute download tanpa konfirmasi
        _execute_download_direct(ui_components, params)

def _execute_download_direct(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Execute download langsung tanpa konfirmasi jika dialog error."""
    try:
        from smartcash.ui.dataset.download.utils.download_executor import execute_roboflow_download
        
        logger = ui_components.get('logger')  
        if logger:
            logger.info("🚀 Memulai download (fallback mode)")
        
        result = execute_roboflow_download(ui_components, params)
        
        if result.get('status') == 'success':
            if logger:
                logger.success("✅ Download berhasil")
        else:
            if logger:
                logger.error(f"❌ Download gagal: {result.get('message', 'Unknown error')}")
                
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Error fallback download: {str(e)}")
    finally:
        _disable_all_buttons(ui_components, False)

def _disable_all_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """Enable/disable semua tombol dengan error handling."""
    try:
        from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons
        disable_download_buttons(ui_components, disabled)
    except Exception:
        # Fallback manual disable
        button_keys = ['download_button', 'check_button', 'reset_button', 'cleanup_button', 'save_button']
        for key in button_keys:
            if key in ui_components and hasattr(ui_components[key], 'disabled'):
                ui_components[key].disabled = disabled