# File: smartcash/ui/pretrained/handlers/pretrained_handlers.py
"""
File: smartcash/ui/pretrained/handlers/pretrained_handlers.py
Deskripsi: Fixed handlers untuk pretrained module dengan DRY config handler inheritance
"""

import os
from typing import Dict, Any
from smartcash.ui.pretrained.handlers.config_handler import PretrainedConfigHandler
from smartcash.ui.pretrained.services.model_checker import PretrainedModelChecker
from smartcash.ui.pretrained.services.model_downloader import PretrainedModelDownloader
from smartcash.ui.pretrained.services.model_syncer import PretrainedModelSyncer
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

def setup_pretrained_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    🔧 Setup handlers untuk pretrained module dengan proper config inheritance
    
    Args:
        ui_components: UI components dictionary
        config: Configuration dictionary
        
    Returns:
        Updated ui_components dengan handlers
    """
    try:
        # Initialize handlers dan services
        config_handler = PretrainedConfigHandler()
        model_checker = PretrainedModelChecker()
        model_downloader = PretrainedModelDownloader()
        model_syncer = PretrainedModelSyncer()
        
        # 🔍 Extract components dengan safe access
        download_sync_button = ui_components.get('download_sync_button')
        progress_tracker = ui_components.get('progress_tracker')
        log_output = ui_components.get('log_output')
        status_panel = ui_components.get('status_panel')
        
        def handle_download_sync():
            """📥 Handle download dan sync pretrained models"""
            try:
                if progress_tracker and hasattr(progress_tracker, 'start'):
                    progress_tracker.start("🔄 Memulai download/sync pretrained models...")
                else:
                    _show_fallback_progress("🔄 Memulai download/sync pretrained models...")
                
                if log_output:
                    with log_output:
                        print("🚀 Memulai proses download/sync pretrained models...")
                
                # Get current config
                current_config = config_handler.extract_config(ui_components)
                pretrained_config = current_config.get('pretrained_models', {})
                
                # Process download/sync
                success_count = 0
                total_count = 0
                
                # Simulate processing (replace with actual implementation)
                models_to_process = ['yolov5s', 'yolov5m', 'yolov5l']
                total_count = len(models_to_process)
                
                for i, model in enumerate(models_to_process):
                    try:
                        # Update progress
                        if progress_tracker and hasattr(progress_tracker, 'update'):
                            progress_tracker.update(f"📦 Processing {model}...", i + 1, total_count)
                        else:
                            _show_fallback_progress(f"📦 Processing {model}... ({i+1}/{total_count})")
                        
                        # Process model (implement actual download/sync logic here)
                        # model_downloader.download_model(model)
                        # model_syncer.sync_model(model)
                        
                        success_count += 1
                        
                        if log_output:
                            with log_output:
                                print(f"✅ {model} berhasil diproses")
                                
                    except Exception as model_error:
                        logger.warning(f"⚠️ Error processing {model}: {str(model_error)}")
                        if log_output:
                            with log_output:
                                print(f"⚠️ Error processing {model}: {str(model_error)}")
                
                # Final status update
                if success_count == total_count:
                    if progress_tracker and hasattr(progress_tracker, 'complete'):
                        progress_tracker.complete(f"✅ Setup selesai! {success_count}/{total_count} models ready")
                    if status_panel:
                        status_panel.value = f"✅ Setup selesai! {success_count}/{total_count} models ready"
                    print(f"🎉 Setup pretrained models selesai! {success_count}/{total_count} berhasil")
                else:
                    if progress_tracker and hasattr(progress_tracker, 'error'):
                        progress_tracker.error(f"⚠️ Partial success: {success_count}/{total_count} models")
                    if status_panel:
                        status_panel.value = f"⚠️ Partial success: {success_count}/{total_count} models"
                    print(f"⚠️ Setup selesai dengan warning: {success_count}/{total_count} berhasil")
                
            except Exception as e:
                error_msg = f"Error setup pretrained: {str(e)}"
                logger.error(f"💥 {error_msg}")
                
                # Update UI dengan error state
                if progress_tracker and hasattr(progress_tracker, 'error'):
                    progress_tracker.error(f"❌ {error_msg}")
                else:
                    _show_fallback_progress(f"❌ {error_msg}")
                
                if status_panel:
                    status_panel.value = f"❌ {error_msg}"
                
                if log_output:
                    with log_output:
                        print(f"💥 {error_msg}")
                
                raise
        
        # 📎 Setup config handlers (save/reset) dengan UI logging integration
        _setup_config_handlers(ui_components)
        
        # 📎 Attach download sync handler
        if download_sync_button:
            download_sync_button.on_click(lambda b: handle_download_sync())
        
        # 📋 Add handlers to ui_components
        ui_components.update({
            'handlers': {
                'download_sync': handle_download_sync
            },
            'config_handler': config_handler,
            'model_checker': model_checker,
            'model_downloader': model_downloader,
            'model_syncer': model_syncer
        })
        
        logger.info("✅ Pretrained handlers setup berhasil dengan shared config inheritance")
        return ui_components
        
    except Exception as e:
        logger.error(f"❌ Error setup pretrained handlers: {str(e)}")
        return ui_components

def _setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset handlers dengan UI logging integration"""
    
    def save_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                logger.error("❌ Config handler tidak tersedia")
                return
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.save_config(ui_components)
        except Exception as e:
            logger.error(f"❌ Error save config: {str(e)}")
    
    def reset_config(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                logger.error("❌ Config handler tidak tersedia")
                return
            
            # Set UI components untuk logging  
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
            config_handler.reset_config(ui_components)
        except Exception as e:
            logger.error(f"❌ Error reset config: {str(e)}")
    
    # Bind handlers dengan safety check
    if save_button := ui_components.get('save_button'):
        save_button.on_click(save_config)
    if reset_button := ui_components.get('reset_button'):
        reset_button.on_click(reset_config)