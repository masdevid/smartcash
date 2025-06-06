"""
File: smartcash/ui/pretrained_model/handlers/check_handler.py
Deskripsi: Handler untuk auto-check model dengan UI integration
"""

from typing import Dict, Any
from smartcash.ui.pretrained_model.services.model_checker import ModelChecker

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup auto-check handler untuk pretrained model"""
    
    def execute_auto_check():
        """Execute auto-check untuk model pretrained"""
        logger = ui_components.get('logger')
        progress_tracker = ui_components.get('progress_tracker')
        
        try:
            _reset_ui_logger(ui_components)
            if logger:
                logger.info("🔍 Memulai auto-check model")
            
            # Show progress tracker dengan API yang benar
            progress_tracker = ui_components.get('progress_tracker')
            if progress_tracker:
                # Gunakan metode show dengan parameter yang benar
                progress_tracker.show("Check Model", ["scan", "verify"], {"scan": 50, "verify": 50})
                # Update progress dengan API yang benar
                progress_tracker.update_overall(10, "🔍 Memeriksa model yang tersedia")
            else:
                # Fallback ke metode lama jika tersedia
                ui_components.get('show_progress', lambda x: None)("Check Model")
                ui_components.get('update_progress', lambda *a: None)('overall', 10, "🔍 Memeriksa model yang tersedia")
            
            # Check existing models
            checker = ModelChecker(ui_components, logger)
            check_result = checker.check_all_models()
            
            models_to_download = check_result.get('missing_models', [])
            existing_models = check_result.get('existing_models', [])
            
            # Log dan update status
            if existing_models and logger:
                logger.info(f"✅ Model tersedia: {', '.join(existing_models)}")
            
            if models_to_download:
                if logger:
                    logger.warning(f"⚠️ Model yang belum tersedia: {', '.join(models_to_download)}")
                _update_status_panel(ui_components, f"⚠️ {len(models_to_download)} model perlu diunduh", "warning")
            else:
                if logger:
                    logger.success("✅ Semua model sudah tersedia")
                _update_status_panel(ui_components, "✅ Semua model sudah tersedia", "success")
            
            # Complete operation dengan API yang benar
            if progress_tracker:
                progress_tracker.complete("✅ Auto-check selesai")
            else:
                # Fallback ke metode lama jika tersedia
                ui_components.get('complete_operation', lambda x: None)("Auto-check selesai")
            
        except Exception as e:
            error_msg = f"Auto-check gagal: {str(e)}"
            if logger:
                logger.error(f"💥 {error_msg}")
            
            # Error operation dengan API yang benar
            if progress_tracker:
                progress_tracker.error(error_msg)
            else:
                # Fallback ke metode lama jika tersedia
                ui_components.get('error_operation', lambda x: None)(error_msg)
                
            _update_status_panel(ui_components, error_msg, "error")
    
    # Execute auto-check jika diaktifkan
    ui_components.get('auto_check_enabled', False) and _schedule_auto_check(execute_auto_check)
    
    return ui_components

def _schedule_auto_check(execute_func):
    """Schedule auto-check dengan delay untuk memastikan UI sudah dimuat"""
    import threading
    threading.Timer(1.0, execute_func).start()

def _reset_ui_logger(ui_components: Dict[str, Any]) -> None:
    """Reset UI logger dan clear outputs"""
    [widget.clear_output(wait=True) for key in ['log_output'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    ui_components.get('status_panel') and hasattr(ui_components['status_panel'], 'value') and update_status_panel(ui_components['status_panel'], message, status_type)