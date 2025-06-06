"""
File: smartcash/ui/pretrained_model/handlers/download_handler.py
Deskripsi: Handler untuk download dan sync model dengan UI integration
"""

from typing import Dict, Any
from smartcash.ui.pretrained_model.services.model_downloader import ModelDownloader
from smartcash.ui.pretrained_model.services.model_checker import ModelChecker
from smartcash.ui.pretrained_model.services.model_syncer import ModelSyncer

def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download & sync handler dengan UI integration"""
    
    def execute_download_sync(button=None):
        """Execute download and sync dengan comprehensive progress tracking"""
        logger = ui_components.get('logger')
        progress_tracker = ui_components.get('progress_tracker')
        
        # Reset UI dan disable button
        _reset_ui_logger(ui_components)
        button and setattr(button, 'disabled', True)
        ui_components.get('show_for_operation', lambda x: None)('download')
        
        try:
            logger and logger.info("🚀 Memulai download dan sinkronisasi model")
            
            # Phase 1: Check existing models
            ui_components.get('update_progress', lambda *a: None)('overall', 10, "🔍 Memeriksa model yang tersedia")
            checker = ModelChecker(ui_components, logger)
            check_result = checker.check_all_models()
            
            models_to_download = check_result.get('missing_models', [])
            existing_models = check_result.get('existing_models', [])
            
            # Log results
            existing_models and logger and logger.info(f"✅ Model tersedia: {', '.join(existing_models)}")
            
            # Phase 2: Download missing models
            if models_to_download:
                logger and logger.info(f"📥 Mengunduh {len(models_to_download)} model: {', '.join(models_to_download)}")
                ui_components.get('update_progress', lambda *a: None)('overall', 30, "📥 Mengunduh model")
                
                downloader = ModelDownloader(ui_components, logger)
                download_result = downloader.download_models(models_to_download)
                
                if not download_result.get('success', False):
                    raise Exception(download_result.get('message', 'Download gagal'))
                
                downloaded_count = download_result.get('downloaded_count', 0)
                logger and logger.success(f"✅ {downloaded_count} model berhasil diunduh")
            else:
                logger and logger.info("ℹ️ Semua model sudah tersedia, skip download")
            
            # Phase 3: Sync to Drive
            ui_components.get('update_progress', lambda *a: None)('overall', 80, "☁️ Sinkronisasi ke Google Drive")
            syncer = ModelSyncer(ui_components, logger)
            sync_result = syncer.sync_to_drive()
            
            sync_count = sync_result.get('synced_count', 0) if sync_result.get('success', False) else 0
            sync_count and logger and logger.success(f"☁️ {sync_count} model disinkronkan ke Drive")
            
            # Phase 4: Complete
            total_models = len(existing_models) + len(models_to_download)
            final_message = f"Setup model selesai: {total_models} model siap digunakan"
            ui_components.get('complete_operation', lambda x: None)(final_message)
            _update_status_panel(ui_components, f"✅ {final_message}", "success")
            
        except Exception as e:
            error_msg = f"Setup model gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            ui_components.get('error_operation', lambda x: None)(error_msg)
            _update_status_panel(ui_components, error_msg, "error")
        
        finally:
            button and setattr(button, 'disabled', False)
    
    ui_components['download_sync_button'].on_click(execute_download_sync)

def _reset_ui_logger(ui_components: Dict[str, Any]) -> None:
    """Reset UI logger dan clear all outputs"""
    [widget.clear_output(wait=True) for key in ['log_output', 'status'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]
    ui_components.get('reset_all', lambda: None)()

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    ui_components.get('status_panel') and hasattr(ui_components['status_panel'], 'value') and update_status_panel(ui_components['status_panel'], message, status_type)