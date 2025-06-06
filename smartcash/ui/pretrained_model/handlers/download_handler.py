"""
File: smartcash/ui/pretrained_model/handlers/download_handler.py
Deskripsi: Optimized download handler dengan enhanced progress tracker integration
"""

from typing import Dict, Any
from smartcash.ui.pretrained_model.services.model_downloader import ModelDownloader
from smartcash.ui.pretrained_model.services.model_checker import ModelChecker
from smartcash.ui.pretrained_model.services.model_syncer import ModelSyncer

def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download & sync handler dengan enhanced progress tracker integration"""
    
    def execute_download_sync(button=None):
        """Execute download and sync dengan enhanced triple progress tracking"""
        logger = ui_components.get('logger')
        
        # Reset UI dan disable button
        _reset_ui_state(ui_components, button)
        
        # Setup enhanced progress dengan predefined steps
        steps = ui_components.get('progress_steps', ["Check Models", "Download Missing", "Sync to Drive"])
        step_weights = ui_components.get('progress_step_weights', {"Check Models": 20, "Download Missing": 60, "Sync to Drive": 20})
        
        # Show progress dengan configuration
        ui_components.get('show', lambda **kw: None)(operation_name="Model Setup", steps=steps, step_weights=step_weights)
        
        try:
            logger and logger.info("🚀 Memulai download dan sinkronisasi model")
            
            # Phase 1: Check existing models (20% weight)
            ui_components.get('update_step', lambda *a: None)(0, "🔍 Memeriksa model yang tersedia")
            checker = ModelChecker(ui_components, logger)
            check_result = checker.check_all_models()
            
            models_to_download = check_result.get('missing_models', [])
            existing_models = check_result.get('existing_models', [])
            
            # Log results dengan conditional logging
            existing_models and logger and logger.info(f"✅ Model tersedia: {', '.join(existing_models)}")
            ui_components.get('update_step', lambda *a: None)(100, f"Check selesai: {len(existing_models)}/{len(existing_models) + len(models_to_download)} tersedia")
            
            # Advance ke phase 2
            ui_components.get('advance_step', lambda *a: None)("📥 Mengunduh model yang hilang")
            
            # Phase 2: Download missing models (60% weight)
            if models_to_download:
                logger and logger.info(f"📥 Mengunduh {len(models_to_download)} model: {', '.join(models_to_download)}")
                downloader = ModelDownloader(ui_components, logger)
                download_result = downloader.download_models(models_to_download)
                
                if not download_result.get('success', False):
                    raise Exception(download_result.get('message', 'Download gagal'))
                
                downloaded_count = download_result.get('downloaded_count', 0)
                logger and logger.success(f"✅ {downloaded_count} model berhasil diunduh")
                ui_components.get('update_step', lambda *a: None)(100, f"Download selesai: {downloaded_count} model")
            else:
                logger and logger.info("ℹ️ Semua model sudah tersedia, skip download")
                ui_components.get('update_step', lambda *a: None)(100, "Skip download - semua model tersedia")
            
            # Advance ke phase 3
            ui_components.get('advance_step', lambda *a: None)("☁️ Sinkronisasi ke Google Drive")
            
            # Phase 3: Sync to Drive (20% weight)
            syncer = ModelSyncer(ui_components, logger)
            sync_result = syncer.sync_to_drive()
            
            sync_count = sync_result.get('synced_count', 0) if sync_result.get('success', False) else 0
            sync_count and logger and logger.success(f"☁️ {sync_count} model disinkronkan ke Drive")
            ui_components.get('update_step', lambda *a: None)(100, f"Sync selesai: {sync_count} model ke Drive")
            
            # Complete operation
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

def _reset_ui_state(ui_components: Dict[str, Any], button=None) -> None:
    """Reset UI state dan clear outputs dengan one-liner operations"""
    # Clear outputs
    [widget.clear_output(wait=True) for key in ['log_output', 'status'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]
    
    # Reset progress tracker
    ui_components.get('reset_all', lambda: None)()
    
    # Disable button during operation
    button and setattr(button, 'disabled', True)

def _update_status_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan consistent formatting"""
    from smartcash.ui.components.status_panel import update_status_panel
    ui_components.get('status_panel') and hasattr(ui_components['status_panel'], 'value') and update_status_panel(ui_components['status_panel'], message, status_type)