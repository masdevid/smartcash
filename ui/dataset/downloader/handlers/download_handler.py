"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Refactored download handler dengan backend integration
"""

from typing import Dict, Any
from smartcash.ui.dataset.downloader.utils.ui_utils import (
    log_download_config, display_check_results, show_download_success,
    clear_outputs, handle_ui_error, show_ui_success
)
from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
from smartcash.ui.dataset.downloader.utils.dialog_utils import (
    show_download_confirmation_dialog, show_cleanup_confirmation_dialog,
    create_confirm_callback, create_cancel_callback
)
from smartcash.ui.dataset.downloader.utils.progress_utils import create_progress_callback
from smartcash.ui.dataset.downloader.utils.backend_utils import (
    check_existing_dataset, get_cleanup_targets, create_backend_downloader, 
    create_backend_cleanup_service
)

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan utils separation"""
    ui_components['progress_callback'] = create_progress_callback(ui_components)
    
    setup_download_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    
    # Setup save/reset handlers
    setup_config_handlers(ui_components)
    
    return ui_components

def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download handler dengan backend service integration"""
    
    def execute_download(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('download_button')
        
        try:
            # Extract dan validate config
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "Config handler tidak tersedia", button_manager)
                return
            
            ui_config = config_handler.extract_config(ui_components)
            validation = config_handler.validate_config(ui_config)
            
            if not validation['valid']:
                error_msg = f"Konfigurasi tidak valid:\n• {chr(10).join(validation['errors'])}"
                handle_ui_error(ui_components, error_msg, button_manager)
                return
            
            # Setup progress tracker
            _setup_progress_tracker(ui_components, "Dataset Download")
            
            # Check existing dan konfirmasi
            _check_and_confirm_download(ui_config, ui_components, button_manager)
            
        except Exception as e:
            handle_ui_error(ui_components, f"Error download handler: {str(e)}", button_manager)
    
    download_button = ui_components.get('download_button')
    if download_button:
        download_button.on_click(execute_download)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup check handler dengan backend scanner integration"""
    
    def execute_check(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            _setup_progress_tracker(ui_components, "Dataset Check")
            
            # Use backend scanner
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(ui_components.get('logger'))
            scanner.set_progress_callback(ui_components['progress_callback'])
            
            result = scanner.scan_existing_dataset_parallel()
            
            if result.get('status') == 'success':
                display_check_results(ui_components, result)
                show_ui_success(ui_components, "Dataset check selesai", button_manager)
            else:
                handle_ui_error(ui_components, result.get('message', 'Scan failed'), button_manager)
                
        except Exception as e:
            handle_ui_error(ui_components, f"Error check handler: {str(e)}", button_manager)
    
    check_button = ui_components.get('check_button')
    if check_button:
        check_button.on_click(execute_check)

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan proper dialog confirmation"""
    
    def execute_cleanup(button=None):
        button_manager = get_button_manager(ui_components)
        
        clear_outputs(ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            # Get cleanup targets using backend utils
            targets_result = get_cleanup_targets(ui_components.get('logger'))
            
            if targets_result.get('status') != 'success':
                handle_ui_error(ui_components, "Gagal mendapatkan cleanup targets", button_manager)
                return
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            
            if total_files == 0:
                show_ui_success(ui_components, "Tidak ada file untuk dibersihkan", button_manager)
                return
            
            # Show confirmation dialog
            show_cleanup_confirmation_dialog(
                ui_components,
                targets_result,
                create_confirm_callback(ui_components, "cleanup", 
                    lambda: _execute_cleanup_with_backend(targets_result, ui_components, button_manager)),
                create_cancel_callback(ui_components, "cleanup")
            )
            
        except Exception as e:
            handle_ui_error(ui_components, f"Error cleanup handler: {str(e)}", button_manager)
    
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup)

def setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset handlers"""
    def save_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "Config handler tidak tersedia")
                return
            
            success = config_handler.save_config(ui_components)
            if success:
                show_ui_success(ui_components, "✅ Konfigurasi berhasil disimpan")
            else:
                handle_ui_error(ui_components, "❌ Gagal menyimpan konfigurasi")
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error saat save: {str(e)}")
    
    def reset_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                handle_ui_error(ui_components, "Config handler tidak tersedia")
                return
            
            success = config_handler.reset_config(ui_components)
            if success:
                show_ui_success(ui_components, "🔄 Konfigurasi berhasil direset")
            else:
                handle_ui_error(ui_components, "❌ Gagal reset konfigurasi")
        except Exception as e:
            handle_ui_error(ui_components, f"❌ Error saat reset: {str(e)}")
    
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config_handler)
    if reset_button:
        reset_button.on_click(reset_config_handler)

def _check_and_confirm_download(ui_config: Dict[str, Any], ui_components: Dict[str, Any], button_manager):
    """Check existing dataset menggunakan backend scanner"""
    try:
        logger = ui_components.get('logger')
        
        # Use backend scanner instead of manual check
        has_content, total_images, summary_data = check_existing_dataset(logger)
        
        if logger:
            logger.info(f"📊 Existing dataset check: {'Found' if has_content else 'Empty'} ({total_images} images)")
        
        if has_content:
            # Show confirmation dialog with backend data
            show_download_confirmation_dialog(
                ui_config,
                ui_components,
                total_images,
                create_confirm_callback(ui_components, "download", 
                    lambda: _execute_download_with_backend(ui_config, ui_components, button_manager)),
                create_cancel_callback(ui_components, "download")
            )
        else:
            # No existing content, proceed directly
            _execute_download_with_backend(ui_config, ui_components, button_manager)
            
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Error checking existing dataset: {str(e)}")
        # Proceed anyway
        _execute_download_with_backend(ui_config, ui_components, button_manager)

def _execute_download_with_backend(ui_config: Dict[str, Any], ui_components: Dict[str, Any], button_manager):
    """Execute download menggunakan backend utils"""
    try:
        logger = ui_components.get('logger')
        
        # Create downloader using backend utils
        downloader = create_backend_downloader(ui_config, logger)
        if not downloader:
            handle_ui_error(ui_components, "❌ Gagal membuat download service", button_manager)
            return
        
        # Setup progress callback
        if hasattr(downloader, 'set_progress_callback'):
            downloader.set_progress_callback(ui_components['progress_callback'])
        
        # Log config dan execute
        log_download_config(ui_components, ui_config)
        result = downloader.download_dataset()
        
        # Handle result
        if result and result.get('status') == 'success':
            show_download_success(ui_components, result)
            show_ui_success(ui_components, "Download dataset berhasil", button_manager)
        else:
            error_msg = result.get('message', 'Download gagal') if result else 'No response from backend service'
            handle_ui_error(ui_components, error_msg, button_manager)
            
    except Exception as e:
        handle_ui_error(ui_components, f"❌ Exception in download execution: {str(e)}", button_manager)

def _execute_cleanup_with_backend(targets_result: Dict[str, Any], ui_components: Dict[str, Any], button_manager):
    """Execute cleanup menggunakan backend utils"""
    try:
        logger = ui_components.get('logger')
        
        # Setup progress tracker
        _setup_progress_tracker(ui_components, "Dataset Cleanup")
        
        # Create cleanup service using backend utils
        cleanup_service = create_backend_cleanup_service(logger)
        if not cleanup_service:
            handle_ui_error(ui_components, "❌ Gagal membuat cleanup service", button_manager)
            return
        
        cleanup_service.set_progress_callback(ui_components['progress_callback'])
        
        # Execute cleanup
        result = cleanup_service.cleanup_dataset_files(targets_result.get('targets', {}))
        
        if result.get('status') == 'success':
            cleaned_count = len(result.get('cleaned_targets', []))
            show_ui_success(ui_components, f"Cleanup selesai: {cleaned_count} direktori dibersihkan", button_manager)
        else:
            handle_ui_error(ui_components, result.get('message', 'Cleanup failed'), button_manager)
            
    except Exception as e:
        handle_ui_error(ui_components, f"Error saat cleanup: {str(e)}", button_manager)

def _setup_progress_tracker(ui_components: Dict[str, Any], operation_name: str):
    """Setup progress tracker untuk operation"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.show(operation_name)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")
    
    logger = ui_components.get('logger')
    if logger:
        logger.info(f"🚀 Memulai {operation_name.lower()}")