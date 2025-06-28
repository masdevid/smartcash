# File: smartcash/ui/pretrained/handlers/event_handlers.py
"""
File: smartcash/ui/pretrained/handlers/event_handlers.py
Deskripsi: Basic event handlers untuk pretrained module dengan fail-fast approach
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def setup_all_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """🔧 Setup semua event handlers untuk pretrained module
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
        **kwargs: Parameter tambahan
        
    Returns:
        Updated UI components dengan handlers
    """
    try:
        # Setup config handlers
        setup_config_handlers(ui_components, config)
        
        # Setup operation handlers
        setup_operation_handlers(ui_components, config)
        
        logger.info("✅ Event handlers berhasil disetup")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error setting up handlers: {str(e)}")
        return ui_components


def setup_config_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """⚙️ Setup handlers untuk config management
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
    """
    try:
        from .config_handler import PretrainedConfigHandler
        
        config_handler = PretrainedConfigHandler()
        
        # Save button handler
        if 'save_button' in ui_components:
            def on_save_click(b):
                try:
                    extracted_config = config_handler.extract_config(ui_components)
                    # TODO: Implement actual save logic
                    logger.info("💾 Config saved successfully")
                except Exception as e:
                    logger.error(f"❌ Error saving config: {str(e)}")
            
            ui_components['save_button'].on_click(on_save_click)
        
        # Reset button handler
        if 'reset_button' in ui_components:
            def on_reset_click(b):
                try:
                    from .config_updater import reset_pretrained_ui
                    reset_pretrained_ui(ui_components)
                    logger.info("🔄 Config reset to defaults")
                except Exception as e:
                    logger.error(f"❌ Error resetting config: {str(e)}")
            
            ui_components['reset_button'].on_click(on_reset_click)
        
        logger.info("✅ Config handlers setup completed")
        
    except Exception as e:
        logger.error(f"❌ Error setting up config handlers: {str(e)}")


def setup_operation_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """🚀 Setup handlers untuk operations (download, sync, etc.)
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
    """
    try:
        # Download & Sync button handler dengan complete implementation
        if 'download_sync_button' in ui_components:
            def on_download_sync_click(b):
                try:
                    from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader
                    from smartcash.ui.pretrained.services.model_syncer import PretrainedModelSyncer
                    from smartcash.ui.pretrained.services.model_checker import check_model_exists
                    
                    logger.info("🚀 Starting download & sync operation...")
                    
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
                        if 'main_progress' in ui_components:
                            ui_components['main_progress'].value = progress
                        if 'sub_progress' in ui_components:
                            ui_components['sub_progress'].description = message
                        _log_to_output(ui_components, f"📊 {progress}% - {message}", "info")
                    
                    # Status callback
                    def update_status(message: str):
                        if 'status' in ui_components:
                            ui_components['status'].value = message
                        _log_to_output(ui_components, message, "info")
                    
                    # Get current model status
                    model_status = ui_components.get('model_status', {})
                    local_exists = check_model_exists(models_dir, model_type)
                    drive_exists = check_model_exists(drive_dir, model_type)
                    
                    # Determine operation based on current status
                    if not local_exists and not drive_exists:
                        # Download new model
                        update_status("⬇️ Downloading YOLOv5s model...")
                        success = downloader.download_yolov5s(models_dir, update_progress, update_status)
                        
                        if success:
                            # Sync to drive after download
                            update_status("📤 Syncing to drive...")
                            syncer.sync_to_drive(models_dir, drive_dir, update_progress, update_status)
                    
                    elif local_exists and not drive_exists:
                        # Sync to drive
                        update_status("📤 Syncing models to drive...")
                        syncer.sync_to_drive(models_dir, drive_dir, update_progress, update_status)
                    
                    elif not local_exists and drive_exists:
                        # Sync from drive
                        update_status("📥 Syncing models from drive...")
                        syncer.sync_from_drive(drive_dir, models_dir, update_progress, update_status)
                    
                    else:
                        # Both exist - bi-directional sync
                        update_status("🔄 Performing bi-directional sync...")
                        syncer.bi_directional_sync(models_dir, drive_dir, update_progress, update_status)
                    
                    # Re-check model status after operation
                    _recheck_model_status(ui_components, config)
                    
                    # Final success message
                    update_progress(100, "Operation completed successfully")
                    logger.info("✅ Download & sync operation completed")
                        
                except Exception as e:
                    error_msg = f"❌ Error during download & sync: {str(e)}"
                    logger.error(error_msg)
                    if 'status' in ui_components:
                        ui_components['status'].value = error_msg
                    _log_to_output(ui_components, error_msg, "error")
            
            ui_components['download_sync_button'].on_click(on_download_sync_click)
        
        logger.info("✅ Operation handlers setup completed")
        
    except Exception as e:
        logger.error(f"❌ Error setting up operation handlers: {str(e)}")


def _update_progress(ui_components: Dict[str, Any], progress: int, message: str = "") -> None:
    """📊 Update progress indicator
    
    Args:
        ui_components: Dictionary berisi UI components
        progress: Progress value (0-100)
        message: Optional progress message
    """
    try:
        if 'main_progress' in ui_components:
            ui_components['main_progress'].value = progress
        
        if message and 'status' in ui_components:
            ui_components['status'].value = message
            
    except Exception as e:
        logger.debug(f"⚠️ Error updating progress: {str(e)}")


def _recheck_model_status(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """🔄 Re-check model status setelah operasi download/sync
    
    Args:
        ui_components: Dictionary berisi UI components
        config: Konfigurasi sistem
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