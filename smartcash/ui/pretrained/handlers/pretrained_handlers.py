# File: smartcash/ui/pretrained/handlers/pretrained_handlers.py
"""
File: smartcash/ui/pretrained/handlers/pretrained_handlers.py
Deskripsi: Complete handlers untuk pretrained module dengan DRY patterns
"""

from typing import Dict, Any
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.ui.pretrained.services.model_checker import PretrainedModelChecker
from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader
from smartcash.ui.pretrained.services.model_syncer import PretrainedModelSyncer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def setup_pretrained_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """🔧 Setup handlers dengan complete workflow patterns"""
    try:
        # Initialize handlers dan services
        config_handler = PretrainedConfigHandler()
        model_checker = PretrainedModelChecker()
        model_downloader = PretrainedModelDownloader()
        model_syncer = PretrainedModelSyncer()
        
        # Setup config handlers (save/reset)
        _setup_config_handlers(ui_components)
        
        # Setup operation handlers
        _setup_operation_handlers(ui_components)
        
        # Add ke ui_components
        ui_components.update({
            'config_handler': config_handler,
            'model_checker': model_checker,
            'model_downloader': model_downloader,
            'model_syncer': model_syncer,
            'handlers': {
                'download_sync': _handle_download_sync
            }
        })
        
        logger.info("✅ Pretrained handlers setup berhasil")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error setup pretrained handlers: {str(e)}")
        return ui_components

def _setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset handlers dengan UI logging"""
    
    def save_config(button=None):
        _clear_outputs(ui_components)
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.save_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error save: {str(e)}", "error")
    
    def reset_config(button=None):
        _clear_outputs(ui_components)
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _log_to_ui(ui_components, "❌ Config handler tidak tersedia", "error")
                return
            
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.reset_config(ui_components)
        except Exception as e:
            _log_to_ui(ui_components, f"❌ Error reset: {str(e)}", "error")
    
    # Bind handlers
    if save_button := ui_components.get('save_button'):
        save_button.on_click(save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(reset_config)

def _setup_operation_handlers(ui_components: Dict[str, Any]):
    """Setup operation handlers dengan confirmation workflow"""
    
    def download_sync_handler(button=None):
        return _handle_download_sync_with_confirmation(ui_components)
    
    if download_sync_button := ui_components.get('download_sync_button'):
        download_sync_button.on_click(download_sync_handler)

def _handle_download_sync_with_confirmation(ui_components: Dict[str, Any]) -> bool:
    """Handle download/sync dengan confirmation pattern"""
    try:
        _clear_outputs(ui_components)
        
        if _should_execute_download_sync(ui_components):
            return _execute_download_sync_with_progress(ui_components)
        
        if not _is_confirmation_pending(ui_components):
            _show_confirmation_area(ui_components)
            _log_to_ui(ui_components, "⏳ Menunggu konfirmasi download/sync...", "info")
            _show_download_sync_confirmation(ui_components)
        
        return True
        
    except Exception as e:
        _handle_error(ui_components, f"❌ Error download/sync: {str(e)}")
        return False

def _execute_download_sync_with_progress(ui_components: Dict[str, Any]) -> bool:
    """Execute download/sync dengan progress tracking"""
    try:
        _disable_buttons(ui_components)
        
        # Setup progress
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'start'):
            progress_tracker.start("🚀 Memulai download/sync pretrained models...")
        else:
            _show_fallback_progress("🚀 Memulai download/sync...")
        
        # Get config untuk processing
        config_handler = ui_components.get('config_handler')
        current_config = config_handler.extract_config(ui_components) if config_handler else {}
        pretrained_config = current_config.get('pretrained_models', {})
        
        # Process models
        models_to_process = _get_models_to_process(pretrained_config)
        total_count = len(models_to_process)
        success_count = 0
        
        for i, model in enumerate(models_to_process):
            try:
                # Update progress
                if progress_tracker and hasattr(progress_tracker, 'update'):
                    progress_tracker.update(f"📦 Processing {model}...", i + 1, total_count)
                else:
                    _show_fallback_progress(f"📦 Processing {model}... ({i+1}/{total_count})")
                
                # Process model (implement actual logic)
                _process_model(model, pretrained_config, ui_components)
                success_count += 1
                
                _log_to_ui(ui_components, f"✅ {model} berhasil diproses", "success")
                
            except Exception as model_error:
                logger.warning(f"⚠️ Error processing {model}: {str(model_error)}")
                _log_to_ui(ui_components, f"⚠️ Error {model}: {str(model_error)}", "warning")
        
        # Final status
        if success_count == total_count:
            if progress_tracker and hasattr(progress_tracker, 'complete'):
                progress_tracker.complete(f"✅ Setup selesai! {success_count}/{total_count} models ready")
            _log_to_ui(ui_components, f"🎉 Download/sync selesai! {success_count}/{total_count} berhasil", "success")
        else:
            if progress_tracker and hasattr(progress_tracker, 'error'):
                progress_tracker.error(f"⚠️ Partial success: {success_count}/{total_count} models")
            _log_to_ui(ui_components, f"⚠️ Partial success: {success_count}/{total_count} models", "warning")
        
        _hide_confirmation_area(ui_components)
        return True
        
    except Exception as e:
        error_msg = f"Error download/sync: {str(e)}"
        logger.error(f"💥 {error_msg}")
        
        if progress_tracker and hasattr(progress_tracker, 'error'):
            progress_tracker.error(f"❌ {error_msg}")
        else:
            _show_fallback_progress(f"❌ {error_msg}")
        
        _handle_error(ui_components, f"💥 {error_msg}")
        return False
    finally:
        _enable_buttons(ui_components)

# === Helper Functions ===

def _get_models_to_process(pretrained_config: Dict[str, Any]) -> list:
    """Get list models untuk diproses"""
    # Default models atau dari config
    return ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']

def _process_model(model: str, config: Dict[str, Any], ui_components: Dict[str, Any]):
    """Process individual model - implement actual logic"""
    # Placeholder untuk actual implementation
    import time
    time.sleep(0.5)  # Simulate processing

def _handle_download_sync(ui_components: Dict[str, Any]) -> bool:
    """Direct download/sync handler"""
    return _handle_download_sync_with_confirmation(ui_components)

# === UI Utility Functions ===

def _log_to_ui(ui_components: Dict[str, Any], message: str, level: str = "info"):
    """Log ke UI dengan emoji context"""
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = message
    
    if log_output := ui_components.get('log_output'):
        with log_output:
            emoji_map = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}
            print(f"{emoji_map.get(level, 'ℹ️')} {message}")
    else:
        print(f"📝 {message}")

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear semua output widgets"""
    if log_output := ui_components.get('log_output'):
        log_output.clear_output()
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = ""

def _handle_error(ui_components: Dict[str, Any], error_message: str):
    """Comprehensive error handling"""
    _log_to_ui(ui_components, error_message, "error")
    if status_panel := ui_components.get('status_panel'):
        status_panel.value = error_message
    _enable_buttons(ui_components)
    _hide_confirmation_area(ui_components)

def _disable_buttons(ui_components: Dict[str, Any]):
    """Disable action buttons"""
    button_keys = [k for k in ui_components.keys() if k.endswith('_button')]
    for key in button_keys:
        if button := ui_components.get(key):
            button.disabled = True

def _enable_buttons(ui_components: Dict[str, Any]):
    """Enable action buttons"""
    button_keys = [k for k in ui_components.keys() if k.endswith('_button')]
    for key in button_keys:
        if button := ui_components.get(key):
            button.disabled = False

def _show_confirmation_area(ui_components: Dict[str, Any]):
    """Show confirmation area"""
    if confirmation_area := ui_components.get('confirmation_area'):
        confirmation_area.layout.display = 'block'

def _hide_confirmation_area(ui_components: Dict[str, Any]):
    """Hide confirmation area"""
    if confirmation_area := ui_components.get('confirmation_area'):
        confirmation_area.layout.display = 'none'

def _show_download_sync_confirmation(ui_components: Dict[str, Any]):
    """Show download/sync confirmation details"""
    config_handler = ui_components.get('config_handler')
    if config_handler:
        current_config = config_handler.extract_config(ui_components)
        pretrained_config = current_config.get('pretrained_models', {})
        models_count = len(_get_models_to_process(pretrained_config))
        summary = f"Download/sync {models_count} pretrained models"
        _log_to_ui(ui_components, f"📋 Konfirmasi: {summary}", "info")

def _should_execute_download_sync(ui_components: Dict[str, Any]) -> bool:
    """Check konfirmasi download/sync"""
    confirm_checkbox = ui_components.get('confirm_download_sync_checkbox')
    return confirm_checkbox and getattr(confirm_checkbox, 'value', False)

def _is_confirmation_pending(ui_components: Dict[str, Any]) -> bool:
    """Check confirmation dialog status"""
    confirmation_area = ui_components.get('confirmation_area')
    return confirmation_area and confirmation_area.layout.display != 'none'

def _show_fallback_progress(message: str):
    """Fallback progress display"""
    print(f"📊 {message}")