"""
File: smartcash/ui/pretrained_model/handlers/download_handler.py
Deskripsi: Handler untuk download dan sync model dengan progress tracker integration
"""

from typing import Dict, Any
from smartcash.ui.pretrained_model.services.model_downloader import ModelDownloader
from smartcash.ui.pretrained_model.services.model_checker import ModelChecker
from smartcash.ui.pretrained_model.services.model_syncer import ModelSyncer

def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download & sync handler dengan progress tracker integration"""
    
    def execute_download_sync(button=None):
        """Execute download and sync dengan comprehensive progress tracking"""
        logger = ui_components.get('logger')
        
        # Reset UI dan disable button
        _reset_ui_logger(ui_components)
        button and setattr(button, 'disabled', True)
        
        try:
            logger and logger.info("🚀 Memulai download dan sinkronisasi model")
            
            # Phase 1: Check existing models (0-30%)
            checker = ModelChecker(ui_components, logger)
            check_result = checker.check_all_models()
            
            models_to_download = check_result.get('missing_models', [])
            existing_models = check_result.get('existing_models', [])
            
            # Log results
            existing_models and logger and logger.info(f"✅ Model tersedia: {', '.join(existing_models)}")
            
            # Phase 2: Download missing models (30-80%)
            if models_to_download:
                logger and logger.info(f"📥 Mengunduh {len(models_to_download)} model: {', '.join(models_to_download)}")
                
                downloader = ModelDownloader(ui_components, logger)
                download_result = downloader.download_models(models_to_download)
                
                if not download_result.get('success', False):
                    raise Exception(download_result.get('message', 'Download gagal'))
                
                downloaded_count = download_result.get('downloaded_count', 0)
                logger and logger.success(f"✅ {downloaded_count} model berhasil diunduh")
            else:
                logger and logger.info("ℹ️ Semua model sudah tersedia, skip download")
                # Update progress untuk skip download
                update_primary = ui_components.get('update_primary')
                update_primary and update_primary(80, "Skip download - semua model tersedia")
            
            # Phase 3: Sync to Drive (80-100%)
            syncer = ModelSyncer(ui_components, logger)
            sync_result = syncer.sync_to_drive()
            
            sync_count = sync_result.get('synced_count', 0) if sync_result.get('success', False) else 0
            sync_count and logger and logger.success(f"☁️ {sync_count} model disinkronkan ke Drive")
            
            # Phase 4: Complete (100%)
            total_models = len(existing_models) + len(models_to_download)
            final_message = f"Setup model selesai: {total_models} model siap digunakan"
            
            # Complete progress tracker
            tracker = ui_components.get('tracker')
            tracker and tracker.complete(final_message)
            _update_status_panel(ui_components, f"✅ {final_message}", "success")
            
        except Exception as e:
            error_msg = f"Setup model gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            _update_status_panel(ui_components, error_msg, "error")
            
            # Error progress tracker
            tracker = ui_components.get('tracker')
            tracker and tracker.error(error_msg)
        
        finally:
            button and setattr(button, 'disabled', False)
    
    ui_components['download_sync_button'].on_click(execute_download_sync)

def _reset_ui_logger(ui_components: Dict[str, Any]) -> None:
    """Reset UI logger dan clear all outputs"""
    [widget.clear_output(wait=True) for key in ['log_output', 'status'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    ui_components.get('status_panel') and hasattr(ui_components['status_panel'], 'value') and update_status_panel(ui_components['status_panel'], message, status_type)