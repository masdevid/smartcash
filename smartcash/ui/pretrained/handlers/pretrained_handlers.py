"""
File: smartcash/ui/pretrained/handlers/pretrained_handlers.py
Deskripsi: Event handlers untuk pretrained module dengan progress integration - Fixed status panel update
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
                    
                    # Create models directory if not exists
                    os.makedirs(models_dir, exist_ok=True)
                    logger.info(f"📁 Models directory: {models_dir}")
                    
                    # Step 1: Check existing models
                    progress_tracker.update_overall(20, "🔍 Checking existing models...")
                    check_results = {}
                    download_results = {}
                    
                    for model_key, model_info in models_config.items():
                        logger.info(f"🔍 Checking {model_info['name']}...")
                        result = model_checker.check_model(
                            model_info['url'],
                            models_dir,
                            model_info.get('filename', f"{model_key}.pt"),
                            model_info.get('min_size_mb', 1)
                        )
                        check_results[model_key] = result
                        logger.info(f"📊 {model_info['name']}: {'✅ Ready' if result['available'] else '❌ Need download'}")
                    
                    # Step 2: Download missing models
                    progress_tracker.update_overall(50, "⬇️ Downloading missing models...")
                    for model_key, result in check_results.items():
                        if not result['available']:
                            model_info = models_config[model_key]
                            logger.info(f"⬇️ Downloading {model_info['name']}...")
                            
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
                    
                    # ✅ FIX: Update status panel menggunakan helper function
                    total_models = len(models_config)
                    successful_downloads = sum(1 for success in download_results.values() if success)
                    
                    if successful_downloads == total_models:
                        status_message = f"✅ All {total_models} models ready"
                        _update_status_panel_safe(status_panel, status_message, 'success')
                        progress_tracker.complete("All pretrained models ready for training!")
                    else:
                        status_message = f"⚠️ {successful_downloads}/{total_models} models ready"
                        _update_status_panel_safe(status_panel, status_message, 'warning')
                        progress_tracker.complete(f"Setup completed with {total_models - successful_downloads} issues")
                    
                    logger.info(f"🎯 Final status: {status_message}")
                    
            except Exception as e:
                error_msg = f"Setup failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                if progress_tracker:
                    progress_tracker.error(error_msg)
                _update_status_panel_safe(status_panel, error_msg, 'error')
        
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


def _update_status_panel_safe(status_panel, message: str, status_type: str = 'info') -> None:
    """
    🔧 Helper function untuk update status panel dengan safe error handling
    
    Args:
        status_panel: Widget HTML status panel
        message: Pesan status
        status_type: Tipe status ('info', 'success', 'warning', 'error')
    """
    if not status_panel:
        logger.warning(f"⚠️ Status panel not found, logging: {message}")
        return
    
    try:
        # Import dan gunakan helper function yang sama dengan preprocessing UI
        from smartcash.ui.utils.alert_utils import update_status_panel
        update_status_panel(status_panel, message, status_type)
        
    except ImportError:
        # Fallback: manual update HTML value
        _manual_status_panel_update(status_panel, message, status_type)
    except Exception as e:
        logger.warning(f"⚠️ Error updating status panel: {str(e)}")
        _manual_status_panel_update(status_panel, message, status_type)


def _manual_status_panel_update(status_panel, message: str, status_type: str) -> None:
    """Manual status panel update jika helper function tidak tersedia"""
    try:
        # Status type color mapping
        colors = {
            'info': {'bg': '#d1ecf1', 'text': '#0c5460', 'icon': 'ℹ️'},
            'success': {'bg': '#d4edda', 'text': '#155724', 'icon': '✅'},
            'warning': {'bg': '#fff3cd', 'text': '#856404', 'icon': '⚠️'},
            'error': {'bg': '#f8d7da', 'text': '#721c24', 'icon': '❌'}
        }
        
        style_info = colors.get(status_type, colors['info'])
        
        # Update HTML value directly
        status_panel.value = f"""
        <div style="
            padding: 10px;
            background-color: {style_info['bg']};
            color: {style_info['text']};
            border-radius: 4px;
            margin: 5px 0;
            border-left: 4px solid {style_info['text']};
        ">
            <p style="margin: 5px 0">{style_info['icon']} {message}</p>
        </div>"""
        
    except Exception as e:
        logger.error(f"❌ Error manual status update: {str(e)}")
        # Last resort: simple text update
        if hasattr(status_panel, 'value'):
            status_panel.value = f"<p>{message}</p>"