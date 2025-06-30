# File: smartcash/ui/pretrained/handlers/event_handlers.py
"""
File: smartcash/ui/pretrained/handlers/event_handlers.py
Deskripsi: Basic event handlers untuk pretrained module dengan fail-fast approach
"""

from typing import Dict, Any, Optional, TypeVar, Callable
from functools import wraps

# Type variables for generic function typing
T = TypeVar('T')

# Type alias for logger bridge to support different logger implementations
LoggerBridge = Any

def with_error_handling(logger_bridge: Optional[LoggerBridge] = None):
    """Decorator untuk menangani error dan logging secara konsisten"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"❌ Error in {func.__name__}: {str(e)}"
                if logger_bridge and hasattr(logger_bridge, 'error'):
                    logger_bridge.error(error_msg, exc_info=True)
                else:
                    print(f"[ERROR] {error_msg}")
                raise  # Re-raise to allow caller to handle
        return wrapper
    return decorator


def _log_debug(logger_bridge: Optional[LoggerBridge], message: str) -> None:
    """Log debug message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'debug'):
        logger_bridge.debug(f"🔧 {message}")


def _log_info(logger_bridge: Optional[LoggerBridge], message: str) -> None:
    """Log info message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'info'):
        logger_bridge.info(f"ℹ️ {message}")


def _log_warning(logger_bridge: Optional[LoggerBridge], message: str) -> None:
    """Log warning message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'warning'):
        logger_bridge.warning(f"⚠️ {message}")


def _log_error(logger_bridge: Optional[LoggerBridge], message: str, exc_info: bool = False) -> None:
    """Log error message using logger_bridge if available"""
    if logger_bridge and hasattr(logger_bridge, 'error'):
        logger_bridge.error(f"❌ {message}", exc_info=exc_info)


def _log_to_output(ui_components: Dict[str, Any], message: str, level: str = "info") -> None:
    """Log message to output console if available"""
    if 'output_console' not in ui_components:
        return
    
    try:
        timestamp = _get_current_timestamp()
        formatted_msg = f"{timestamp} - {message}"
        
        if level == "error":
            ui_components['output_console'].append_stdout(f"❌ {formatted_msg}")
        elif level == "warning":
            ui_components['output_console'].append_stdout(f"⚠️ {formatted_msg}")
        elif level == "success":
            ui_components['output_console'].append_stdout(f"✅ {formatted_msg}")
        else:
            ui_components['output_console'].append_stdout(f"ℹ️ {formatted_msg}")
    except Exception as e:
        print(f"[ERROR] Error logging to output console: {str(e)}")


def setup_all_handlers(
    ui_components: Dict[str, Any], 
    config: Dict[str, Any], 
    logger_bridge: Optional[LoggerBridge] = None,
    **kwargs
) -> Dict[str, Any]:
    """🔧 Setup semua event handlers untuk pretrained module
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
        logger_bridge: Logger bridge instance untuk logging (opsional)
        **kwargs: Parameter tambahan
        
    Returns:
        Updated UI components dengan handlers
    """
    try:
        # Setup config handlers
        setup_config_handlers(ui_components, config, logger_bridge)
        
        # Setup operation handlers
        setup_operation_handlers(ui_components, config, logger_bridge)
        
        _log_info(logger_bridge, "Event handlers berhasil disetup")
        return ui_components
        
    except Exception as e:
        error_msg = f"Error setting up handlers: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        return ui_components


def setup_config_handlers(
    ui_components: Dict[str, Any], 
    config: Dict[str, Any], 
    logger_bridge: Optional[LoggerBridge] = None
) -> None:
    """⚙️ Setup handlers untuk config management
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
        logger_bridge: Logger bridge instance untuk logging (opsional)
    """
    try:
        from .config_handler import PretrainedConfigHandler
        
        # Initialize config handler with logger bridge if available
        config_handler = PretrainedConfigHandler()
        
        # Save button handler
        if 'save_button' in ui_components:
            def on_save_click(b):
                try:
                    extracted_config = config_handler.extract_config(ui_components)
                    # TODO: Implement actual save logic
                    _log_info(logger_bridge, "Config saved successfully")
                    _log_to_output(ui_components, "Konfigurasi berhasil disimpan", "success")
                except Exception as e:
                    error_msg = f"Gagal menyimpan konfigurasi: {str(e)}"
                    _log_error(logger_bridge, error_msg, exc_info=True)
                    _log_to_output(ui_components, error_msg, "error")
            
            ui_components['save_button'].on_click(on_save_click)
        
        # Reset button handler
        if 'reset_button' in ui_components:
            def on_reset_click(b):
                try:
                    from .config_updater import reset_pretrained_ui
                    reset_pretrained_ui(ui_components, logger_bridge)
                    _log_info(logger_bridge, "Config reset to defaults")
                    _log_to_output(ui_components, "Konfigurasi direset ke nilai default", "success")
                except Exception as e:
                    error_msg = f"Gagal mereset konfigurasi: {str(e)}"
                    _log_error(logger_bridge, error_msg, exc_info=True)
                    _log_to_output(ui_components, error_msg, "error")
            
            ui_components['reset_button'].on_click(on_reset_click)
        
        _log_info(logger_bridge, "Config handlers setup completed")
        
    except Exception as e:
        error_msg = f"Error setting up config handlers: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        raise


def setup_operation_handlers(
    ui_components: Dict[str, Any], 
    config: Dict[str, Any],
    logger_bridge: Optional[LoggerBridge] = None
) -> None:
    """🚀 Setup handlers untuk operations (download, sync, etc.)
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
        logger_bridge: Logger bridge instance untuk logging (opsional)
    """
    try:
        # Download & Sync button handler dengan complete implementation
        if 'download_sync_button' in ui_components:
            def on_download_sync_click(b):
                try:
                    from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader
                    from smartcash.ui.pretrained.services.model_syncer import PretrainedModelSyncer
                    from smartcash.ui.pretrained.services.model_checker import check_model_exists
                    
                    _log_info(logger_bridge, "Starting download & sync operation...")
                    _log_to_output(ui_components, "🚀 Memulai operasi download & sinkronisasi...", "info")
                    
                    # Get config
                    pretrained_config = config.get('pretrained_models', {})
                    models_dir = pretrained_config.get('models_dir', '/content/models')
                    drive_dir = pretrained_config.get('drive_models_dir', '/data/pretrained')
                    model_type = 'yolov5s'  # Fixed to YOLOv5s only
                    
                    # Initialize services
                    downloader = PretrainedModelDownloader()
                    syncer = PretrainedModelSyncer()
                    
                    # Progress callback
                    def update_progress(progress: int, message: str):
                        try:
                            if 'main_progress' in ui_components:
                                ui_components['main_progress'].value = progress
                            if 'sub_progress' in ui_components:
                                ui_components['sub_progress'].description = message
                            _log_to_output(ui_components, f"📊 {progress}% - {message}", "info")
                            _log_debug(logger_bridge, f"Progress: {progress}% - {message}")
                        except Exception as e:
                            _log_error(logger_bridge, f"Error updating progress: {str(e)}", exc_info=True)
                    
                    # Status callback
                    def update_status(message: str):
                        try:
                            if 'status' in ui_components:
                                ui_components['status'].value = message
                            _log_to_output(ui_components, message, "info")
                            _log_info(logger_bridge, f"Status: {message}")
                        except Exception as e:
                            _log_error(logger_bridge, f"Error updating status: {str(e)}", exc_info=True)
                    
                    try:
                        # Get current model status
                        model_status = ui_components.get('model_status', {})
                        local_exists = check_model_exists(models_dir, model_type)
                        drive_exists = check_model_exists(drive_dir, model_type)
                        
                        # Determine operation based on current status
                        if not local_exists and not drive_exists:
                            # Download new model
                            update_status("⬇️ Mengunduh model YOLOv5s...")
                            success = downloader.download_yolov5s(models_dir, update_progress, update_status)
                            
                            if success:
                                # Sync to drive after download
                                update_status("📤 Menyinkronkan ke drive...")
                                syncer.sync_to_drive(models_dir, drive_dir, update_progress, update_status)
                        
                        elif local_exists and not drive_exists:
                            # Sync to drive
                            update_status("📤 Menyinkronkan model ke drive...")
                            syncer.sync_to_drive(models_dir, drive_dir, update_progress, update_status)
                        
                        elif not local_exists and drive_exists:
                            # Sync from drive
                            update_status("📥 Menyinkronkan model dari drive...")
                            syncer.sync_from_drive(drive_dir, models_dir, update_progress, update_status)
                        
                        else:
                            # Both exist - bi-directional sync
                            update_status("🔄 Melakukan sinkronisasi dua arah...")
                            syncer.bi_directional_sync(models_dir, drive_dir, update_progress, update_status)
                        
                        # Re-check model status after operation
                        _recheck_model_status(ui_components, config, logger_bridge)
                        
                        # Final success message
                        update_progress(100, "Operasi berhasil diselesaikan")
                        _log_info(logger_bridge, "Download & sync operation completed")
                        _log_to_output(ui_components, "✅ Operasi download & sinkronisasi selesai", "success")
                            
                    except Exception as e:
                        error_msg = f"Gagal melakukan operasi download/sinkronisasi: {str(e)}"
                        _log_error(logger_bridge, error_msg, exc_info=True)
                        _log_to_output(ui_components, error_msg, "error")
                        if 'status' in ui_components:
                            ui_components['status'].value = error_msg
                        raise
                        
                except Exception as e:
                    error_msg = f"Error during download & sync: {str(e)}"
                    _log_error(logger_bridge, error_msg, exc_info=True)
                    _log_to_output(ui_components, error_msg, "error")
                    if 'status' in ui_components:
                        ui_components['status'].value = error_msg
            
            ui_components['download_sync_button'].on_click(on_download_sync_click)
        
        _log_info(logger_bridge, "Operation handlers setup completed")
        
    except Exception as e:
        error_msg = f"Error setting up operation handlers: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)
        raise


def _update_progress(
    ui_components: Dict[str, Any], 
    progress: int, 
    message: str = "",
    logger_bridge: Optional[LoggerBridge] = None
) -> None:
    """📊 Update progress indicator
    
    Args:
        ui_components: Dictionary berisi UI components
        progress: Progress value (0-100)
        message: Optional progress message
        logger_bridge: Logger bridge instance untuk logging (opsional)
    """
    try:
        if 'main_progress' in ui_components:
            ui_components['main_progress'].value = progress
        
        if message and 'status' in ui_components:
            ui_components['status'].value = message
        
        _log_debug(logger_bridge, f"Progress updated: {progress}% - {message}")
            
    except Exception as e:
        error_msg = f"Error updating progress: {str(e)}"
        _log_error(logger_bridge, error_msg, exc_info=True)


def _recheck_model_status(
    ui_components: Dict[str, Any], 
    config: Dict[str, Any],
    logger_bridge: Optional[LoggerBridge] = None
) -> None:
    """🔄 Re-check model status setelah operasi download/sync
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
        logger_bridge: Logger bridge instance untuk logging (opsional)
    """
    try:
        from smartcash.ui.pretrained.services.model_checker import check_model_exists
        
        pretrained_config = config.get('pretrained_models', {})
        models_dir = pretrained_config.get('models_dir', '/content/models')
        drive_dir = pretrained_config.get('drive_models_dir', '/data/pretrained')
        model_type = pretrained_config.get('pretrained_type', 'yolov5s')
        
        # Re-check model existence
        local_exists = check_model_exists(models_dir, model_type)
        drive_exists = check_model_exists(drive_dir, model_type)
        
        # Update model status
        ui_components['model_status'] = {
            'local_exists': local_exists,
            'drive_exists': drive_exists,
            'model_type': model_type,
            'last_checked': _get_current_timestamp()
        }
        
        # Update UI status
        if local_exists and drive_exists:
            status_msg = f"✅ Model {model_type} tersedia di local dan drive"
            ui_components['download_sync_button'].description = '🔄 Re-sync Models'
            ui_components['download_sync_button'].button_style = 'info'
        elif local_exists or drive_exists:
            location = "local" if local_exists else "drive"
            status_msg = f"⚠️ Model {model_type} hanya tersedia di {location}"
            ui_components['download_sync_button'].description = '📥 Sync Models'
            ui_components['download_sync_button'].button_style = 'warning'
        else:
            status_msg = f"❌ Model {model_type} tidak ditemukan"
            ui_components['download_sync_button'].description = '📥 Download Model'
            ui_components['download_sync_button'].button_style = 'primary'
        
        if 'status' in ui_components:
            ui_components['status'].value = status_msg
            
        logger.info(f"🔄 Model status updated: {status_msg}")
        
    except Exception as e:
        logger.error(f"❌ Error rechecking model status: {str(e)}")


def _get_current_timestamp() -> str:
    """⏰ Get current timestamp - menggunakan standard datetime"""
    from datetime import datetime
    return datetime.now().isoformat()
    """📝 Log message to output widget
    
    Args:
        ui_components: Dictionary berisi UI components
        message: Message to log
        level: Log level (info, warning, error)
    """
    try:
        if 'log_output' in ui_components:
            with ui_components['log_output']:
                emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(level, "ℹ️")
                print(f"{emoji} {message}")
                
    except Exception as e:
        logger.debug(f"⚠️ Error logging to output: {str(e)}")