"""
File: smartcash/ui/pretrained_model/handlers/pretrained_handlers.py
Deskripsi: Unified handlers yang terintegrasi dengan pretrained model services
"""

from typing import Dict, Any

def setup_pretrained_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup unified handlers dengan inline implementation"""
    
    # Progress callback
    def create_progress_callback():
        def progress_callback(**kwargs):
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            ui_components.get('update_progress', lambda *a: None)('overall', progress, message)
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Status handler
    def update_status(message: str, status_type: str = "info"):
        from smartcash.ui.components.status_panel import update_status_panel
        if 'status_panel' in ui_components:
            update_status_panel(ui_components['status_panel'], message, status_type)
    
    ui_components['update_status'] = update_status
    
    # Download handler
    def execute_download_sync(button=None):
        logger = ui_components.get('logger')
        
        _reset_ui_logger(ui_components)
        button and setattr(button, 'disabled', True) and setattr(button, 'description', "Processing...")
        
        try:
            logger and logger.info("🚀 Memulai download dan sinkronisasi model")
            ui_components.get('show_for_operation', lambda x: None)('download')
            
            from smartcash.ui.pretrained_model.services.model_checker import ModelChecker
            from smartcash.ui.pretrained_model.services.model_downloader import ModelDownloader
            from smartcash.ui.pretrained_model.services.model_syncer import ModelSyncer
            
            # Check existing models
            ui_components.get('update_progress', lambda *a: None)('overall', 10, "Memeriksa model yang tersedia")
            checker = ModelChecker(config, logger)
            check_result = checker.check_all_models()
            
            models_to_download = check_result.get('missing_models', [])
            existing_models = check_result.get('existing_models', [])
            
            existing_models and logger and logger.info(f"✅ Model tersedia: {', '.join(existing_models)}")
            
            # Download missing models
            if models_to_download:
                logger and logger.info(f"📥 Mengunduh {len(models_to_download)} model: {', '.join(models_to_download)}")
                ui_components.get('update_progress', lambda *a: None)('overall', 30, "Mengunduh model")
                
                downloader = ModelDownloader(config, logger)
                download_result = downloader.download_models(models_to_download)
                
                if not download_result.get('success', False):
                    raise Exception(download_result.get('message', 'Download gagal'))
                
                downloaded_count = download_result.get('downloaded_count', 0)
                logger and logger.success(f"✅ {downloaded_count} model berhasil diunduh")
            else:
                logger and logger.info("ℹ️ Semua model sudah tersedia, skip download")
            
            # Sync to Drive
            ui_components.get('update_progress', lambda *a: None)('overall', 80, "Sinkronisasi ke Google Drive")
            syncer = ModelSyncer(config, logger)
            sync_result = syncer.sync_to_drive()
            
            if sync_result.get('success', False):
                sync_count = sync_result.get('synced_count', 0)
                logger and logger.success(f"☁️ {sync_count} model disinkronkan ke Drive")
            
            # Final status
            total_models = len(existing_models) + len(models_to_download)
            ui_components.get('complete_operation', lambda x: None)(f"Setup model selesai: {total_models} model siap digunakan")
            update_status(f"✅ Setup model selesai: {total_models} model siap", "success")
            
        except Exception as e:
            error_msg = f"Setup model gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            ui_components.get('error_operation', lambda x: None)(error_msg)
            update_status(error_msg, "error")
        
        finally:
            button and setattr(button, 'disabled', False) and setattr(button, 'description', "Download & Sync Model")
    
    # Reset handler
    def execute_reset_ui(button=None):
        logger = ui_components.get('logger')
        
        try:
            _reset_ui_logger(ui_components)
            ui_components.get('reset_all', lambda: None)()
            update_status("UI berhasil direset", "success")
            logger and logger.info("🧹 UI berhasil direset - siap untuk operasi baru")
        except Exception as e:
            error_msg = f"Reset UI gagal: {str(e)}"
            logger and logger.error(f"💥 {error_msg}")
            update_status(error_msg, "error")
    
    # Attach handlers
    ui_components.get('download_sync_button') and ui_components['download_sync_button'].on_click(execute_download_sync)
    ui_components.get('reset_ui_button') and ui_components['reset_ui_button'].on_click(execute_reset_ui)
    
    return ui_components

def _reset_ui_logger(ui_components: Dict[str, Any]):
    """Reset UI logger dan clear all outputs"""
    [widget.clear_output(wait=True) for key in ['log_output', 'status'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]
    ui_components.get('reset_all', lambda: None)()