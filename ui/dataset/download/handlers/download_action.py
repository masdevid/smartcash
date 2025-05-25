"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Fixed download action dengan integrasi latest progress_tracking dan button_state_manager
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager
from smartcash.ui.dataset.download.handlers.validation_handler import validate_download_parameters
from smartcash.ui.dataset.download.handlers.confirmation_handler import handle_download_confirmation
from smartcash.ui.dataset.download.handlers.execution_handler import execute_download_process

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """
    Handler utama untuk aksi download dataset dengan latest progress tracking integration.
    
    Args:
        ui_components: Dictionary komponen UI
        button: Button widget yang diklik
    """
    logger = ui_components.get('logger')
    button_manager = get_button_state_manager(ui_components)
    
    with button_manager.operation_context('download'):
        try:
            logger and logger.info("🚀 Memulai proses download dataset")
            
            # Step 1: Clear UI dan setup progress dengan latest ProgressTracker
            _clear_ui_outputs(ui_components)
            _setup_enhanced_download_progress(ui_components)
            
            # Step 2: Validasi parameter (10% progress)
            _update_download_progress(ui_components, 10, "🔍 Validasi parameter...")
            validation_result = validate_download_parameters(ui_components)
            
            if not validation_result['valid']:
                error_msg = f"❌ Validasi gagal: {validation_result['message']}"
                logger and logger.error(error_msg)
                _error_download_progress(ui_components, validation_result['message'])
                raise Exception(validation_result['message'])
            
            # Step 3: Check existing dataset (20% progress)
            _update_download_progress(ui_components, 20, "📊 Memeriksa dataset yang ada...")
            params = validation_result['params']
            
            # Step 4: Handle confirmation atau langsung execute
            handle_download_confirmation(ui_components, params, _execute_confirmed_download)
            
        except Exception as e:
            error_msg = f"Error download: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            _error_download_progress(ui_components, str(e))
            raise

def _execute_confirmed_download(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """
    Execute download setelah konfirmasi user dengan latest progress tracking.
    
    Args:
        ui_components: Dictionary komponen UI
        params: Parameter download yang sudah divalidasi
    """
    logger = ui_components.get('logger')
    
    try:
        _update_download_progress(ui_components, 30, "🚀 Memulai download...")
        
        if logger:
            logger.info("✅ Parameter valid - memulai download:")
            for key, value in params.items():
                if key != 'api_key':  # Don't log sensitive data
                    logger.info(f"   • {key}: {value}")
        
        # Execute download process dengan enhanced progress tracking
        result = execute_download_process(ui_components, params)
        
        # Handle hasil download
        if result.get('status') == 'success':
            _complete_download_success(ui_components, result)
        else:
            error_msg = result.get('message', 'Unknown error')
            _error_download_progress(ui_components, error_msg)
            logger and logger.error(f"❌ Download gagal: {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        logger and logger.error(f"💥 Error saat execute download: {str(e)}")
        raise

def _setup_enhanced_download_progress(ui_components: Dict[str, Any]) -> None:
    """Setup enhanced progress tracking untuk download operation."""
    # Use latest ProgressTracker integration
    if 'show_for_operation' in ui_components:
        ui_components['show_for_operation']('download')
    elif 'tracker' in ui_components:
        ui_components['tracker'].show('download')

def _update_download_progress(ui_components: Dict[str, Any], progress: int, message: str, color: str = None) -> None:
    """Update download progress dengan latest ProgressTracker integration."""
    # Use latest progress tracking methods
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, f"📊 {message}", color or 'info')
    elif 'tracker' in ui_components:
        ui_components['tracker'].update('overall', progress, message, color)

def _error_download_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state dengan latest ProgressTracker integration."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](f"❌ {message}")
    elif 'tracker' in ui_components:
        ui_components['tracker'].error(f"❌ {message}")

def _complete_download_success(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Handle successful download completion dengan latest progress tracking."""
    logger = ui_components.get('logger')
    stats = result.get('stats', {})
    duration = result.get('duration', 0)
    storage_type = "Google Drive" if result.get('drive_storage', False) else "Local Storage"
    
    # Complete progress tracking dengan latest integration
    if 'complete_operation' in ui_components:
        ui_components['complete_operation']("✅ Download selesai!")
    elif 'tracker' in ui_components:
        ui_components['tracker'].complete("✅ Download selesai!")
    
    # Log success details
    if logger:
        logger.success(f"🎉 Download berhasil dalam {duration:.1f} detik!")
        logger.info(f"📁 Storage: {storage_type}")
        logger.info(f"📂 Path: {result.get('output_dir', 'Unknown')}")
        logger.info(f"🖼️ Total gambar: {stats.get('total_images', 0)}")
        
        # Log breakdown per split
        for split in ['train', 'valid', 'test']:
            split_key = f'{split}_images'
            if split_key in stats and stats[split_key] > 0:
                logger.info(f"   • {split}: {stats[split_key]} gambar")

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs untuk memulai fresh."""
    try:
        if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
            ui_components['log_output'].clear_output(wait=True)
        if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
            ui_components['confirmation_area'].clear_output()
    except Exception:
        pass