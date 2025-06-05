"""
File: smartcash/ui/pretrained_model/handlers/download_handler.py
Deskripsi: Handler khusus untuk download dan sync model dengan SRP approach
"""

from typing import Dict, Any
from smartcash.ui.pretrained_model.services.model_downloader import ModelDownloader
from smartcash.ui.pretrained_model.services.model_checker import ModelChecker
from smartcash.ui.pretrained_model.services.model_syncer import ModelSyncer

def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download & sync handler dengan integrated state management"""
    
    def execute_download_sync(button=None):
        """Execute download and sync dengan comprehensive logging"""
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        # Reset UI logger dan clear outputs
        _reset_ui_logger(ui_components)
        button_manager.disable_buttons('download_sync_button')
        
        try:
            logger and logger.info("🚀 Memulai download dan sinkronisasi model")
            # Gunakan progress_tracker jika tersedia
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'show_for_operation'):
                ui_components['progress_tracker'].show_for_operation('download')
            else:
                # Fallback ke metode lama
                ui_components.get('show_for_operation', lambda x: None)('download')
            
            # Phase 1: Check existing models
            # Gunakan progress_tracker jika tersedia
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'update'):
                ui_components['progress_tracker'].update('overall', 10, "Memeriksa model yang tersedia")
            else:
                # Fallback ke metode lama
                ui_components.get('update_progress', lambda *a: None)('overall', 10, "Memeriksa model yang tersedia")
            checker = ModelChecker(config, logger)
            check_result = checker.check_all_models()
            
            models_to_download = check_result.get('missing_models', [])
            existing_models = check_result.get('existing_models', [])
            
            # Log existing models
            if existing_models:
                logger and logger.info(f"✅ Model tersedia: {', '.join(existing_models)}")
            
            # Phase 2: Download missing models
            if models_to_download:
                logger and logger.info(f"📥 Mengunduh {len(models_to_download)} model: {', '.join(models_to_download)}")
                # Gunakan progress_tracker jika tersedia
                if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'update'):
                    ui_components['progress_tracker'].update('overall', 30, "Mengunduh model")
                else:
                    # Fallback ke metode lama
                    ui_components.get('update_progress', lambda *a: None)('overall', 30, "Mengunduh model")
                
                downloader = ModelDownloader(config, logger)
                downloader.register_progress_callback(ui_components.get('progress_callback'))
                
                download_result = downloader.download_models(models_to_download)
                
                if not download_result.get('success', False):
                    raise Exception(download_result.get('message', 'Download gagal'))
                
                downloaded_count = download_result.get('downloaded_count', 0)
                logger and logger.success(f"✅ {downloaded_count} model berhasil diunduh")
            else:
                logger and logger.info("ℹ️ Semua model sudah tersedia, skip download")
            
            # Phase 3: Sync to Drive
            # Gunakan progress_tracker jika tersedia
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'update'):
                ui_components['progress_tracker'].update('overall', 80, "Sinkronisasi ke Google Drive")
            else:
                # Fallback ke metode lama
                ui_components.get('update_progress', lambda *a: None)('overall', 80, "Sinkronisasi ke Google Drive")
            syncer = ModelSyncer(config, logger)
            sync_result = syncer.sync_to_drive()
            
            if sync_result.get('success', False):
                sync_count = sync_result.get('synced_count', 0)
                logger and logger.success(f"☁️ {sync_count} model disinkronkan ke Drive")
            
            # Phase 4: Final status
            total_models = len(existing_models) + len(models_to_download)
            # Gunakan progress_tracker jika tersedia
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'complete_operation'):
                ui_components['progress_tracker'].complete_operation(f"Setup model selesai: {total_models} model siap digunakan")
            else:
                # Fallback ke metode lama
                ui_components.get('complete_operation', lambda x: None)(f"Setup model selesai: {total_models} model siap digunakan")
            _update_status_panel(ui_components, f"✅ Setup model selesai: {total_models} model siap", "success")
            
        except Exception as e:
            error_msg = f"Setup model gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            # Gunakan progress_tracker jika tersedia
            if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'error_operation'):
                ui_components['progress_tracker'].error_operation(error_msg)
            else:
                # Fallback ke metode lama
                ui_components.get('error_operation', lambda x: None)(error_msg)
            _update_status_panel(ui_components, error_msg, "error")
        
        finally:
            button_manager.enable_buttons()
    
    ui_components['download_sync_button'].on_click(execute_download_sync)

# Helper functions
def _get_button_manager(ui_components: Dict[str, Any]):
    """Get button manager dengan fallback"""
    from smartcash.ui.utils.button_state_manager import get_button_state_manager
    if 'button_manager' not in ui_components:
        ui_components['button_manager'] = get_button_state_manager(ui_components)
    return ui_components['button_manager']

def _reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger dan clear all outputs"""
    for key in ['log_output', 'status']:
        widget = ui_components.get(key)
        if widget and hasattr(widget, 'clear_output'):
            widget.clear_output(wait=True)
    
    # Reset progress container
    # Gunakan progress_tracker jika tersedia
    if 'progress_tracker' in ui_components and hasattr(ui_components['progress_tracker'], 'reset_all'):
        ui_components['progress_tracker'].reset_all()
    else:
        # Fallback ke metode lama
        ui_components.get('reset_all', lambda: None)()

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info"):
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    if 'status_panel' in ui_components and hasattr(ui_components['status_panel'], 'value'):
        update_status_panel(ui_components['status_panel'], message, status_type)