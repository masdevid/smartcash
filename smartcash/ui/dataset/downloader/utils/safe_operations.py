"""
File: smartcash/ui/dataset/downloader/utils/safe_operations.py
Deskripsi: Safe operations utility untuk logger, progress tracker, dan download operations
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.components.progress_tracker import create_dual_progress_tracker

def create_safe_logger(ui_components: Dict[str, Any], module_name: str = 'downloader'):
    """Create safe logger dengan UI integration dan one-liner setup"""
    try:
        from smartcash.ui.utils.ui_logger import create_ui_logger
        return create_ui_logger(ui_components, name=module_name, log_to_file=False, redirect_stdout=False)
    except Exception:
        # Fallback ke standard logger jika UI logger gagal
        return get_logger(f'downloader.{module_name}')

def setup_safe_progress_tracker(ui_components: Dict[str, Any]) -> None:
    """Setup safe progress tracker dengan fallback handling"""
    try:
        # Progress tracker sudah dibuat di ui_components, hanya validate
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            # Create new jika tidak ada
            tracker = create_dual_progress_tracker(operation="Download Operation")
            ui_components['progress_tracker'] = tracker
            ui_components['progress_container'] = tracker.container
    except Exception as e:
        # Create minimal progress tracker jika error
        ui_components['progress_tracker'] = _create_minimal_progress_tracker()

def _create_minimal_progress_tracker():
    """Create minimal progress tracker untuk fallback"""
    class MinimalProgressTracker:
        def start_operation(self, message: str = ""): pass
        def update_progress(self, value: int, status: str = ""): pass
        def complete_operation(self, success: bool = True, message: str = ""): pass
        def reset(self): pass
    
    return MinimalProgressTracker()

def safe_download_operation(downloader, progress_tracker, logger, ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute download operation dengan safe error handling dan progress tracking.
    
    Args:
        downloader: Downloader service instance
        progress_tracker: Progress tracker untuk UI feedback
        logger: Logger untuk output
        ui_components: UI components dictionary
        
    Returns:
        Dictionary berisi result dan status
    """
    try:
        # Validate downloader
        if not downloader:
            raise Exception("Downloader service tidak tersedia")
        
        # Update progress: preparation
        progress_tracker.update_progress(30, "Memvalidasi konfigurasi...")
        logger.info("🔧 Validating download configuration...")
        
        # Execute download
        progress_tracker.update_progress(50, "Downloading dataset...")
        logger.info("📥 Starting dataset download...")
        
        # Call download method dengan error handling
        result = _execute_download_with_progress(downloader, progress_tracker, logger)
        
        if result.get('success'):
            progress_tracker.update_progress(90, "Finalizing download...")
            logger.info("🔄 Finalizing download process...")
            
            return {
                'success': True,
                'message': 'Download completed successfully',
                'total_files': result.get('total_files', 0),
                'output_dir': result.get('output_dir', 'data')
            }
        else:
            raise Exception(result.get('error', 'Download failed'))
    
    except Exception as e:
        logger.error(f"❌ Download operation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': f'Download failed: {str(e)}'
        }

def _execute_download_with_progress(downloader, progress_tracker, logger) -> Dict[str, Any]:
    """Execute actual download dengan progress updates"""
    try:
        # Check if downloader has download_dataset method
        if not hasattr(downloader, 'download_dataset'):
            raise Exception("Downloader tidak memiliki method download_dataset")
        
        # Progress callback untuk downloader
        def progress_callback(current: int, total: int, status: str = ""):
            if total > 0:
                percentage = int((current / total) * 40) + 50  # 50-90%
                progress_tracker.update_progress(percentage, status)
        
        # Execute download
        result = downloader.download_dataset()
        
        # Check result
        if isinstance(result, dict) and result.get('success'):
            return result
        else:
            return {'success': False, 'error': 'Download returned unsuccessful result'}
    
    except Exception as e:
        return {'success': False, 'error': str(e)}

def safe_config_operation(operation_func, logger, operation_name: str = "config operation") -> Any:
    """
    Execute config operation dengan safe error handling.
    
    Args:
        operation_func: Function yang akan dijalankan
        logger: Logger untuk error reporting
        operation_name: Nama operasi untuk logging
        
    Returns:
        Result dari operation atau None jika error
    """
    try:
        return operation_func()
    except Exception as e:
        logger.warning(f"⚠️ {operation_name} warning: {str(e)}")
        return None

def safe_ui_update(ui_components: Dict[str, Any], widget_name: str, property_name: str, value: Any, logger=None) -> bool:
    """
    Safe UI widget update dengan error handling.
    
    Args:
        ui_components: UI components dictionary
        widget_name: Nama widget yang akan diupdate
        property_name: Nama property yang akan diubah
        value: Nilai baru
        logger: Optional logger
        
    Returns:
        True jika sukses, False jika error
    """
    try:
        widget = ui_components.get(widget_name)
        if widget and hasattr(widget, property_name):
            setattr(widget, property_name, value)
            return True
        return False
    except Exception as e:
        if logger:
            logger.debug(f"🔧 UI update warning {widget_name}.{property_name}: {str(e)}")
        return False

def safe_widget_access(ui_components: Dict[str, Any], widget_name: str, default_value: Any = None) -> Any:
    """
    Safe widget value access dengan default fallback.
    
    Args:
        ui_components: UI components dictionary
        widget_name: Nama widget
        default_value: Default value jika widget tidak ada atau error
        
    Returns:
        Widget value atau default_value
    """
    try:
        widget = ui_components.get(widget_name)
        if widget and hasattr(widget, 'value'):
            return widget.value
        return default_value
    except Exception:
        return default_value

def validate_ui_components(ui_components: Dict[str, Any], required_components: list, logger=None) -> Dict[str, Any]:
    """
    Validate UI components untuk memastikan semua required components ada.
    
    Args:
        ui_components: UI components dictionary
        required_components: List required component names
        logger: Optional logger
        
    Returns:
        Dictionary berisi validation result
    """
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        error_msg = f"Missing UI components: {', '.join(missing_components)}"
        if logger:
            logger.error(f"❌ {error_msg}")
        return {'valid': False, 'missing': missing_components, 'message': error_msg}
    
    return {'valid': True, 'missing': [], 'message': 'All required components available'}

# One-liner utilities untuk common operations
get_safe_widget_value = lambda ui_components, widget_name, default=None: safe_widget_access(ui_components, widget_name, default)
set_safe_widget_value = lambda ui_components, widget_name, value, logger=None: safe_ui_update(ui_components, widget_name, 'value', value, logger)
has_widget = lambda ui_components, widget_name: widget_name in ui_components and ui_components[widget_name] is not None
is_widget_enabled = lambda ui_components, widget_name: not getattr(ui_components.get(widget_name), 'disabled', True)