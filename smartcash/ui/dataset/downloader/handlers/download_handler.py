"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Fixed download handler dengan proper service integration dan error handling yang disederhanakan
"""

from typing import Dict, Any, Callable
from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.components.dialogs import confirm
from smartcash.dataset.downloader import get_downloader_instance


def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup handlers dengan fixed integration"""
    from smartcash.ui.dataset.downloader.handlers.check_handler import setup_check_handler
    from smartcash.ui.dataset.downloader.handlers.cleanup_handler import setup_cleanup_handler
    
    logger = ui_components.get('logger')
    
    # Setup individual handlers dan tambahkan ke ui_components
    ui_components['check_handler'] = setup_check_handler(ui_components, config, logger)
    
    ui_components['cleanup_handler'] = setup_cleanup_handler(ui_components, config, logger)
    
    ui_components['download_handler'] = _setup_download_handler(ui_components, config, logger)
    
    logger.debug("✅ Semua handlers berhasil disetup")
    return ui_components

def _setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger):
    """Setup download button handler dengan proper service integration"""
    download_button = ui_components.get('download_button')
    if not download_button:
        logger.error("❌ Download button tidak ditemukan")
        # Buat dummy handler untuk menghindari error
        class DummyHandler:
            def set_progress_callback(self, callback): pass
        return DummyHandler()
    
    # Buat objek handler dengan metode yang diperlukan
    class DownloadHandlerObj:
        def __init__(self):
            self.logger = logger
            
        def set_progress_callback(self, callback):
            # Dummy method untuk compatibility
            logger.debug(f"🔧 Progress callback diset pada download handler")
    
    # Buat instance handler
    handler = DownloadHandlerObj()
    
    def on_download_click(button):
        """Handle download dengan validation dan service call"""
        try:
            # Disable button during operation
            button.disabled = True
            
            # Extract dan validate config
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                show_status_safe("❌ Config handler tidak tersedia", "error", ui_components)
                button.disabled = False
                return
            
            current_config = config_handler.extract_config_from_ui(ui_components)
            validation = config_handler.validate_config(current_config)
            
            if not validation['valid']:
                error_msg = f"❌ Config tidak valid: {'; '.join(validation['errors'])}"
                show_status_safe(error_msg, "error", ui_components)
                logger.error(error_msg)
                return
            
            # Check existing dataset
            from pathlib import Path
            has_existing = any((Path('data') / split).exists() and 
                             len(list((Path('data') / split).rglob('*'))) > 0 
                             for split in ['train', 'valid', 'test'])
            
            if has_existing:
                _show_download_confirmation(ui_components, current_config, logger)
            else:
                _execute_download(ui_components, current_config, logger)
                
        except Exception as e:
            logger.error(f"❌ Error download handler: {str(e)}")
            show_status_safe(f"❌ Error: {str(e)}", "error", ui_components)
        finally:
            button.disabled = False
    
    # Clear existing handlers dan attach new one
    try:
        download_button._click_handlers.callbacks.clear()
    except:
        pass
    
    download_button.on_click(on_download_click)
    logger.debug("✅ Download handler berhasil disetup")
    
    # Return handler object untuk digunakan di ui_components
    return handler

def _show_download_confirmation(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Show confirmation dialog untuk download yang akan menimpa existing data"""
    roboflow = config.get('data', {}).get('roboflow', {})
    download_config = config.get('download', {})
    
    dataset_id = f"{roboflow.get('workspace', '')}/{roboflow.get('project', '')}:v{roboflow.get('version', '')}"
    
    message = f"""Dataset existing akan ditimpa!

🎯 Target: {dataset_id}
🔄 UUID Renaming: {'✅' if download_config.get('rename_files', True) else '❌'}
✅ Validasi: {'✅' if download_config.get('validate_download', True) else '❌'}
💾 Backup: {'✅' if download_config.get('backup_existing', False) else '❌'}

Lanjutkan download?"""
    
    confirm(
        "Konfirmasi Download Dataset", 
        message,
        on_yes=lambda btn: _execute_download(ui_components, config, logger),
        on_no=lambda btn: logger.info("🚫 Download dibatalkan")
    )

def _execute_download(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Execute download dengan service integration"""
    try:
        # Setup progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if not progress_tracker:
            logger.error("❌ Progress tracker tidak tersedia")
            return
        
        # Prepare download config
        roboflow = config.get('data', {}).get('roboflow', {})
        download_service_config = {
            'workspace': roboflow.get('workspace', ''),
            'project': roboflow.get('project', ''),
            'version': roboflow.get('version', ''),
            'api_key': roboflow.get('api_key', ''),
            'output_format': 'yolov5pytorch',
            'validate_download': config.get('download', {}).get('validate_download', True),
            'organize_dataset': True,
            'backup_existing': config.get('download', {}).get('backup_existing', False),
            'rename_files': True
        }
        
        # Validate required fields
        required = ['workspace', 'project', 'version', 'api_key']
        missing = [f for f in required if not download_service_config[f]]
        
        if missing:
            error_msg = f"❌ Field wajib kosong: {', '.join(missing)}"
            show_status_safe(error_msg, "error", ui_components)
            return
        
        # Show progress
        progress_tracker.show("Download Dataset")
        logger.info(f"📥 Memulai download {roboflow.get('workspace', '')}/{roboflow.get('project', '')}:v{roboflow.get('version', '')}")
        
        # Create downloader service
        try:
            downloader = get_downloader_instance(download_service_config, logger)
            if not downloader:
                error_msg = "❌ Gagal membuat download service"
                progress_tracker.error(error_msg)
                show_status_safe(error_msg, "error", ui_components)
                return
        except Exception as e:
            error_msg = f"❌ Gagal membuat download service: {str(e)}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            logger.error(error_msg)
            return
        
        # Setup progress callback
        def progress_callback(step: str, current: int, total: int, message: str):
            if total > 0:
                percentage = int((current / total) * 100)
                progress_tracker.update_progress(percentage, f"{step}: {message}")
        
        try:
            if hasattr(downloader, 'set_progress_callback') and callable(downloader.set_progress_callback):
                downloader.set_progress_callback(progress_callback)
            else:
                logger.debug("⚠️ Downloader tidak mendukung progress callback")
        except Exception as e:
            logger.debug(f"⚠️ Error saat setup progress callback: {str(e)}")  # Progress callback is optional
        
        # Execute download
        result = downloader.download_dataset()
        
        if result and result.get('status') == 'success':
            stats = result.get('stats', {})
            success_msg = f"✅ Dataset berhasil didownload: {stats.get('total_images', 0):,} gambar"
            progress_tracker.complete(success_msg)
            show_status_safe(success_msg, "success", ui_components)
            logger.success(success_msg)
        else:
            error_msg = f"❌ Download gagal: {result.get('message', 'Unknown error') if result else 'No response'}"
            progress_tracker.error(error_msg)
            show_status_safe(error_msg, "error", ui_components)
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"❌ Error saat download: {str(e)}"
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.error(error_msg)
        show_status_safe(error_msg, "error", ui_components)
        logger.error(error_msg)

# Export
__all__ = ['setup_download_handlers']