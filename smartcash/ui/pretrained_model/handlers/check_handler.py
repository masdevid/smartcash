"""
File: smartcash/ui/pretrained_model/handlers/check_handler.py
Deskripsi: Handler khusus untuk auto-check model dengan SRP approach
"""

from typing import Dict, Any

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup auto-check handler untuk pretrained model"""
    
    def execute_auto_check():
        """Execute auto-check untuk model pretrained"""
        logger = ui_components.get('logger')
        
        try:
            # Reset UI logger tapi jangan reset UI
            _reset_ui_logger(ui_components)
            
            logger and logger.info("🔍 Memulai auto-check model")
            
            # Gunakan progress_tracker jika tersedia
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'update'):
                ui_components['progress_tracker'].update('overall', 10, "Memeriksa model yang tersedia")
                ui_components['progress_tracker'].show_for_operation('check')
            else:
                # Fallback ke metode lama
                ui_components.get('show_for_operation', lambda x: None)('check')
                ui_components.get('update_progress', lambda *a: None)('overall', 10, "Memeriksa model yang tersedia")
            
            # Import model checker
            from smartcash.ui.pretrained_model.services.model_checker import ModelChecker
            
            # Check existing models
            checker = ModelChecker(config, logger)
            check_result = checker.check_all_models()
            
            models_to_download = check_result.get('missing_models', [])
            existing_models = check_result.get('existing_models', [])
            
            # Log existing models
            if existing_models:
                logger and logger.info(f"✅ Model tersedia: {', '.join(existing_models)}")
            
            # Log missing models
            if models_to_download:
                logger and logger.warning(f"⚠️ Model yang belum tersedia: {', '.join(models_to_download)}")
                _update_status_panel(ui_components, f"⚠️ {len(models_to_download)} model perlu diunduh", "warning")
            else:
                logger and logger.success("✅ Semua model sudah tersedia")
                _update_status_panel(ui_components, "✅ Semua model sudah tersedia", "success")
            
            # Complete operation
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'complete_operation'):
                ui_components['progress_tracker'].complete_operation("Auto-check selesai")
            else:
                ui_components.get('complete_operation', lambda x: None)("Auto-check selesai")
            
        except Exception as e:
            error_msg = f"Auto-check gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'error_operation'):
                ui_components['progress_tracker'].error_operation(error_msg)
            else:
                ui_components.get('error_operation', lambda x: None)(error_msg)
                
            _update_status_panel(ui_components, error_msg, "error")
    
    # Execute auto-check jika diaktifkan
    if ui_components.get('auto_check_enabled', False):
        # Tunggu sebentar sebelum auto-check untuk memastikan UI sudah dimuat
        import threading
        threading.Timer(1.0, execute_auto_check).start()
    
    return ui_components

def _reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger dan clear all outputs"""
    for key in ['log_output']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
        update_status_panel(ui_components['status_panel'], message, status_type)
