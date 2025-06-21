"""
File: smartcash/ui/pretrained/handlers/pretrained_handlers.py
Deskripsi: Event handlers untuk pretrained module dengan progress integration
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
    Setup handlers untuk pretrained module dengan progress tracking.
    
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
        
        # Get components
        download_sync_button = ui_components.get('download_sync_button')
        progress_tracker = ui_components.get('progress_tracker')
        log_output = ui_components.get('log_output')
        status_panel = ui_components.get('status_panel')
        
        def handle_download_sync():
            """Handle download and sync operation dengan progress tracking"""
            try:
                if not progress_tracker:
                    logger.warning("⚠️ Progress tracker not found")
                    return
                
                # Start progress tracking
                progress_tracker.show("Pretrained Models Setup", ["Check", "Download", "Sync"])
                progress_tracker.update_overall(0, "🔍 Memulai setup pretrained models...")
                
                # Clear previous logs
                if log_output:
                    log_output.clear_output()
                
                with log_output if log_output else logger:
                    # Extract current config
                    current_config = config_handler.extract_config(ui_components)
                    pretrained_config = current_config.get('pretrained_models', {})
                    models_dir = pretrained_config.get('models_dir', '/data/pretrained')
                    models_config = pretrained_config.get('models', {})
                    
                    # Create models directory
                    os.makedirs(models_dir, exist_ok=True)
                    logger.info(f"📁 Models directory: {models_dir}")
                    
                    # Step 1: Check existing models
                    progress_tracker.update_overall(20, "🔍 Checking existing models...")
                    check_results = {}
                    for model_key, model_info in models_config.items():
                        model_path = os.path.join(models_dir, model_info['filename'])
                        exists = model_checker.check_model_exists(model_path)
                        check_results[model_key] = {
                            'exists': exists,
                            'path': model_path,
                            'info': model_info
                        }
                        logger.info(f"✅ {model_info['name']}: {'Found' if exists else 'Not found'}")
                    
                    # Step 2: Download missing models
                    progress_tracker.update_overall(40, "⬇️ Downloading missing models...")
                    download_results = {}
                    for model_key, result in check_results.items():
                        if not result['exists']:
                            model_info = result['info']
                            logger.info(f"📥 Downloading {model_info['name']}...")
                            
                            # Update progress per model
                            progress_tracker.update_current(0, f"Downloading {model_info['name']}...")
                            
                            download_success = model_downloader.download_model(
                                model_info['url'],
                                result['path'],
                                progress_callback=lambda p, msg: progress_tracker.update_current(p, msg)
                            )
                            
                            download_results[model_key] = download_success
                            if download_success:
                                logger.info(f"✅ Downloaded {model_info['name']}")
                            else:
                                logger.error(f"❌ Failed to download {model_info['name']}")
                        else:
                            download_results[model_key] = True
                            logger.info(f"⏭️ Skipped {result['info']['name']} (already exists)")
                    
                    # Step 3: Sync to drive (jika diperlukan)
                    progress_tracker.update_overall(80, "🔄 Syncing to drive...")
                    drive_dir = pretrained_config.get('drive_models_dir', '')
                    if drive_dir and os.path.exists('/content/drive'):
                        sync_success = model_syncer.sync_models_to_drive(models_dir, drive_dir)
                        if sync_success:
                            logger.info(f"✅ Models synced to drive: {drive_dir}")
                        else:
                            logger.warning("⚠️ Drive sync failed")
                    else:
                        logger.info("⏭️ Drive sync skipped (drive not mounted)")
                    
                    # Final results
                    progress_tracker.update_overall(100, "✅ Setup completed!")
                    
                    # Update status
                    total_models = len(models_config)
                    successful_downloads = sum(1 for success in download_results.values() if success)
                    
                    if successful_downloads == total_models:
                        status_message = f"✅ All {total_models} models ready"
                        if status_panel:
                            status_panel.update_status(status_message, 'success')
                        progress_tracker.complete("All pretrained models ready for training!")
                    else:
                        status_message = f"⚠️ {successful_downloads}/{total_models} models ready"
                        if status_panel:
                            status_panel.update_status(status_message, 'warning')
                        progress_tracker.complete(f"Setup completed with {total_models - successful_downloads} issues")
                    
                    logger.info(f"🎯 Final status: {status_message}")
                    
            except Exception as e:
                error_msg = f"Setup failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                if progress_tracker:
                    progress_tracker.error(error_msg)
                if status_panel:
                    status_panel.update_status(error_msg, 'error')
        
        # Attach event handler
        if download_sync_button:
            download_sync_button.on_click(lambda _: handle_download_sync())
            logger.info("🔗 Download sync handler attached")
        
        # Return updated components
        return {
            'config_handler': config_handler,
            'model_checker': model_checker,
            'model_downloader': model_downloader,
            'model_syncer': model_syncer,
            'handle_download_sync': handle_download_sync
        }
        
    except Exception as e:
        logger.error(f"❌ Error setting up handlers: {str(e)}")
        return {}