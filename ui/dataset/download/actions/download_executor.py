"""
File: smartcash/ui/dataset/download/actions/download_executor.py
Deskripsi: Download action executor dengan enhanced validation dan progress tracking integration
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.services.download_validation_service import DownloadValidationService
from smartcash.ui.dataset.download.services.download_execution_service import DownloadExecutionService

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Execute download action dengan enhanced validation dan progress tracking."""
    logger = ui_components.get('logger')
    
    try:
        logger and logger.info("🚀 Memulai proses download dataset")
        
        # Clear UI dan setup progress
        _clear_ui_outputs(ui_components)
        _setup_download_progress(ui_components)
        
        # Step 1: Comprehensive validation (20% progress)
        _update_download_progress(ui_components, 10, "🔍 Validasi parameter komprehensif...")
        validation_service = DownloadValidationService(ui_components)
        validation_result = validation_service.validate_comprehensive()
        
        if not validation_result['valid']:
            error_msg = f"❌ Validasi gagal: {validation_result['message']}"
            logger and logger.error(error_msg)
            _error_download_progress(ui_components, validation_result['message'])
            return
        
        # Step 2: Execution dengan progress tracking
        _update_download_progress(ui_components, 20, "📊 Mempersiapkan eksekusi download...")
        execution_service = DownloadExecutionService(ui_components)
        
        # Execute dengan confirmation handling
        params = validation_result['params']
        execution_service.execute_with_confirmation(params, _download_completion_callback)
        
    except Exception as e:
        error_msg = f"Error download: {str(e)}"
        logger and logger.error(f"💥 {error_msg}")
        _error_download_progress(ui_components, str(e))

def _download_completion_callback(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Callback untuk download completion dengan result processing."""
    logger = ui_components.get('logger')
    
    if result.get('status') == 'success':
        _complete_download_success(ui_components, result)
    else:
        error_msg = result.get('message', 'Unknown error')
        _error_download_progress(ui_components, error_msg)
        logger and logger.error(f"❌ Download gagal: {error_msg}")

def _setup_download_progress(ui_components: Dict[str, Any]) -> None:
    """Setup progress tracking untuk download operation."""
    if 'show_for_operation' in ui_components:
        ui_components['show_for_operation']('download')
    elif 'tracker' in ui_components:
        ui_components['tracker'].show('download')

def _update_download_progress(ui_components: Dict[str, Any], progress: int, message: str, color: str = None) -> None:
    """Update download progress dengan latest integration."""
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', progress, f"📊 {message}", color or 'info')
    elif 'tracker' in ui_components:
        ui_components['tracker'].update('overall', progress, message, color)

def _error_download_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Set error state dengan latest integration."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](f"❌ {message}")
    elif 'tracker' in ui_components:
        ui_components['tracker'].error(f"❌ {message}")

def _complete_download_success(ui_components: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Handle successful download completion."""
    logger = ui_components.get('logger')
    stats = result.get('stats', {})
    duration = result.get('duration', 0)
    storage_type = "Google Drive" if result.get('drive_storage', False) else "Local Storage"
    
    # Complete progress tracking
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

# Helper functions untuk service integration
def get_download_executor_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get status download executor untuk debugging."""
    return {
        'validation_service_available': True,  # Always available
        'execution_service_available': True,   # Always available
        'progress_integration': {
            'show_for_operation': 'show_for_operation' in ui_components,
            'update_progress': 'update_progress' in ui_components,
            'complete_operation': 'complete_operation' in ui_components,
            'error_operation': 'error_operation' in ui_components,
            'tracker': 'tracker' in ui_components
        },
        'ui_outputs': {
            'log_output': 'log_output' in ui_components,
            'confirmation_area': 'confirmation_area' in ui_components
        },
        'logger_available': 'logger' in ui_components
    }