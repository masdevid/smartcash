"""
File: smartcash/ui/dataset/download/handlers/download_action.py
Deskripsi: Handler download yang diperbaiki dengan koneksi service yang tepat dan error handling yang robust
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.ui_validator import validate_download_params
from smartcash.ui.dataset.download.utils.download_executor import execute_roboflow_download
from smartcash.ui.dataset.download.utils.button_state import disable_download_buttons

def execute_download_action(ui_components: Dict[str, Any], button: Any = None) -> None:
    """Eksekusi download dengan validasi lengkap dan progress tracking."""
    logger = ui_components.get('logger')
    
    # 🚀 Start download process
    if logger:
        logger.info("🚀 Memulai proses download dataset")
    
    # Disable semua buttons untuk mencegah double click
    disable_download_buttons(ui_components, True)
    
    try:
        # 🧹 Clear outputs sebelum mulai
        _clear_ui_outputs(ui_components)
        
        # ✅ Validasi parameter
        if logger:
            logger.info("📋 Memvalidasi parameter download...")
        
        validation_result = validate_download_params(ui_components)
        if not validation_result['valid']:
            if logger:
                logger.error(f"❌ Validasi gagal: {validation_result['message']}")
            return
        
        # 📊 Log parameter yang akan digunakan
        params = validation_result['params']
        if logger:
            logger.info("✅ Parameter valid:")
            logger.info(f"   • Workspace: {params['workspace']}")
            logger.info(f"   • Project: {params['project']}")
            logger.info(f"   • Version: {params['version']}")
            logger.info(f"   • Output: {params['output_dir']}")
        
        # 🎯 Execute download langsung (tanpa dialog konfirmasi yang bermasalah)
        _execute_download_direct(ui_components, params)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error persiapan download: {str(e)}")
        # Re-enable buttons jika error
        disable_download_buttons(ui_components, False)

def _clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear semua UI output sebelum mulai download."""
    # Clear log output
    if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'clear_output'):
        ui_components['log_output'].clear_output(wait=True)
    
    # Clear confirmation area
    if 'confirmation_area' in ui_components and hasattr(ui_components['confirmation_area'], 'clear_output'):
        ui_components['confirmation_area'].clear_output()
    
    # Reset progress indicators
    _reset_progress_indicators(ui_components)

def _reset_progress_indicators(ui_components: Dict[str, Any]) -> None:
    """Reset semua progress indicators."""
    progress_widgets = ['progress_bar', 'current_progress']
    for widget_key in progress_widgets:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Progress: 0%"
    
    # Show progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'block'
        ui_components['progress_container'].layout.visibility = 'visible'

def _execute_download_direct(ui_components: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Execute download langsung dengan progress tracking."""
    logger = ui_components.get('logger')
    
    try:
        if logger:
            logger.info("🚀 Memulai download dataset...")
        
        # Execute download dengan progress callback yang sudah terintegrasi
        result = execute_roboflow_download(ui_components, params)
        
        # Handle hasil download
        if result.get('status') == 'success':
            if logger:
                stats = result.get('stats', {})
                duration = result.get('duration', 0)
                storage_type = "Drive" if result.get('drive_storage', False) else "Local"
                
                logger.success(f"✅ Download berhasil ({duration:.1f}s)")
                logger.info(f"📁 Storage: {storage_type}")
                logger.info(f"📊 Dataset: {stats.get('total_images', 0)} gambar")
                logger.info(f"📍 Lokasi: {result.get('output_dir', '')}")
        else:
            error_msg = result.get('message', 'Unknown error')
            if logger:
                logger.error(f"❌ Download gagal: {error_msg}")
                
    except Exception as e:
        if logger:
            logger.error(f"❌ Error download: {str(e)}")
    finally:
        # Re-enable buttons setelah selesai
        disable_download_buttons(ui_components, False)