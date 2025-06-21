"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Simplified download handler dengan shared dialog API seperti preprocessing
"""

from typing import Dict, Any
from smartcash.ui.dataset.downloader.utils.ui_utils import clear_outputs, handle_ui_error, show_ui_success, log_to_accordion
from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager
from smartcash.ui.dataset.downloader.utils.progress_utils import create_progress_callback
from smartcash.ui.dataset.downloader.utils.backend_utils import check_existing_dataset, create_backend_downloader, get_cleanup_targets, create_backend_cleanup_service

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan shared dialog API"""
    ui_components['progress_callback'] = create_progress_callback(ui_components)
    
    setup_download_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    setup_config_handlers(ui_components)
    
    return ui_components

def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download handler dengan dialog confirmation"""
    
    def execute_download(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('download_button')
        
        try:
            # Extract dan validate config
            config_handler = ui_components.get('config_handler')
            ui_config = config_handler.extract_config(ui_components)
            validation = config_handler.validate_config(ui_config)
            
            if not validation['valid']:
                handle_ui_error(ui_components, f"Config tidak valid: {', '.join(validation['errors'])}", button_manager)
                return
            
            # Check existing dataset
            has_existing, total_images, summary_data = check_existing_dataset(ui_components.get('logger'))
            
            if has_existing:
                _show_download_confirmation(ui_components, ui_config, total_images)
            else:
                _execute_download_operation(ui_components, ui_config, button_manager)
                
        except Exception as e:
            handle_ui_error(ui_components, f"Error download handler: {str(e)}", button_manager)
    
    download_button = ui_components.get('download_button')
    if download_button:
        download_button.on_click(execute_download)

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup check handler dengan backend scanner"""
    
    def execute_check(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            _setup_progress_tracker(ui_components, "Dataset Check")
            
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(ui_components.get('logger'))
            scanner.set_progress_callback(ui_components['progress_callback'])
            
            result = scanner.scan_existing_dataset_parallel()
            
            if result.get('status') == 'success':
                from smartcash.ui.dataset.downloader.utils.ui_utils import display_check_results
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
    """Setup cleanup handler dengan dialog confirmation"""
    
    def execute_cleanup(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            # Get cleanup targets
            targets_result = get_cleanup_targets(ui_components.get('logger'))
            
            if targets_result.get('status') != 'success':
                handle_ui_error(ui_components, "Gagal mendapatkan cleanup targets", button_manager)
                return
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            
            if total_files == 0:
                show_ui_success(ui_components, "Tidak ada file untuk dibersihkan", button_manager)
                return
            
            # Show confirmation dengan shared dialog
            _show_cleanup_confirmation(ui_components, targets_result)
            
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
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
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
            
            # Set UI components untuk logging
            if hasattr(config_handler, 'set_ui_components'):
                config_handler.set_ui_components(ui_components)
            
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

# === CONFIRMATION HANDLERS ===

def _show_download_confirmation(ui_components: Dict[str, Any], ui_config: Dict[str, Any], existing_count: int):
    """Show download confirmation menggunakan shared dialog API"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        # Show confirmation area dan log waiting
        _show_confirmation_area(ui_components)
        log_to_accordion(ui_components, "⏳ Menunggu konfirmasi download dari user...", "info")
        
        roboflow = ui_config.get('data', {}).get('roboflow', {})
        download = ui_config.get('download', {})
        backup_enabled = download.get('backup_existing', False)
        
        # Build message
        message_lines = [
            f"Dataset existing akan ditimpa! ({existing_count:,} file)",
            "",
            f"🎯 Target: {roboflow.get('workspace')}/{roboflow.get('project')}:v{roboflow.get('version')}",
            f"🔄 UUID Renaming: {'✅' if download.get('rename_files', True) else '❌'}",
            f"✅ Validasi: {'✅' if download.get('validate_download', True) else '❌'}",
            f"💾 Backup: {'✅' if backup_enabled else '❌'}",
            "",
            "Lanjutkan download?"
        ]
        
        show_confirmation_dialog(
            ui_components,
            title="⚠️ Konfirmasi Download Dataset",
            message="<br>".join(message_lines),
            on_confirm=lambda: _handle_download_confirm(ui_components, ui_config),
            on_cancel=lambda: _handle_download_cancel(ui_components),
            confirm_text="Ya, Download",
            cancel_text="Batal",
            danger_mode=True
        )
        
    except ImportError:
        log_to_accordion(ui_components, "⚠️ Dialog tidak tersedia, langsung execute", "warning")
        _hide_confirmation_area(ui_components)
        _execute_download_operation(ui_components, ui_config, get_button_manager(ui_components))
    except Exception as e:
        log_to_accordion(ui_components, f"⚠️ Error showing confirmation: {str(e)}", "warning")
        _hide_confirmation_area(ui_components)

def _show_cleanup_confirmation(ui_components: Dict[str, Any], targets_result: Dict[str, Any]):
    """Show cleanup confirmation menggunakan shared dialog API"""
    try:
        from smartcash.ui.components.dialog import show_confirmation_dialog
        
        # Show confirmation area dan log waiting
        _show_confirmation_area(ui_components)
        log_to_accordion(ui_components, "⏳ Menunggu konfirmasi cleanup dari user...", "info")
        
        summary = targets_result.get('summary', {})
        targets = targets_result.get('targets', {})
        
        message_lines = [
            f"Akan menghapus {summary.get('total_files', 0):,} file ({summary.get('size_formatted', '0 B')})",
            "",
            "📂 Target cleanup:"
        ]
        
        for target_name, target_info in targets.items():
            file_count = target_info.get('file_count', 0)
            size_formatted = target_info.get('size_formatted', '0 B')
            message_lines.append(f"  • {target_name}: {file_count:,} file ({size_formatted})")
        
        message_lines.extend(["", "⚠️ Direktori akan tetap dipertahankan", "Lanjutkan cleanup?"])
        
        show_confirmation_dialog(
            ui_components,
            title="⚠️ Konfirmasi Cleanup Dataset",
            message="<br>".join(message_lines),
            on_confirm=lambda: _handle_cleanup_confirm(ui_components, targets_result),
            on_cancel=lambda: _handle_cleanup_cancel(ui_components),
            confirm_text="Ya, Hapus",
            cancel_text="Batal",
            danger_mode=True
        )
        
    except ImportError:
        log_to_accordion(ui_components, "⚠️ Dialog tidak tersedia, langsung execute", "warning")
        _hide_confirmation_area(ui_components)
        _execute_cleanup_operation(ui_components, targets_result, get_button_manager(ui_components))
    except Exception as e:
        log_to_accordion(ui_components, f"⚠️ Error showing cleanup confirmation: {str(e)}", "warning")
        _hide_confirmation_area(ui_components)

def _handle_download_confirm(ui_components: Dict[str, Any], ui_config: Dict[str, Any]):
    """Handle download confirmation"""
    _hide_confirmation_area(ui_components)
    log_to_accordion(ui_components, "✅ Download dikonfirmasi, memulai...", "success")
    button_manager = get_button_manager(ui_components)
    _execute_download_operation(ui_components, ui_config, button_manager)

def _handle_download_cancel(ui_components: Dict[str, Any]):
    """Handle download cancellation"""
    _hide_confirmation_area(ui_components)
    log_to_accordion(ui_components, "🚫 Download dibatalkan oleh user", "info")
    button_manager = get_button_manager(ui_components)
    button_manager.enable_buttons()

def _handle_cleanup_confirm(ui_components: Dict[str, Any], targets_result: Dict[str, Any]):
    """Handle cleanup confirmation"""
    _hide_confirmation_area(ui_components)
    log_to_accordion(ui_components, "✅ Cleanup dikonfirmasi, memulai...", "success")
    button_manager = get_button_manager(ui_components)
    _execute_cleanup_operation(ui_components, targets_result, button_manager)

def _handle_cleanup_cancel(ui_components: Dict[str, Any]):
    """Handle cleanup cancellation"""
    _hide_confirmation_area(ui_components)
    log_to_accordion(ui_components, "🚫 Cleanup dibatalkan oleh user", "info")
    button_manager = get_button_manager(ui_components)
    button_manager.enable_buttons()

# === CONFIRMATION AREA MANAGEMENT ===

def _show_confirmation_area(ui_components: Dict[str, Any]):
    """Show confirmation area dengan visibility management"""
    from smartcash.ui.dataset.downloader.utils.ui_utils import show_confirmation_area
    show_confirmation_area(ui_components)

def _hide_confirmation_area(ui_components: Dict[str, Any]):
    """Hide confirmation area dengan visibility management"""
    from smartcash.ui.dataset.downloader.utils.ui_utils import hide_confirmation_area
    hide_confirmation_area(ui_components)

# === EXECUTION HELPERS ===

def _execute_download_operation(ui_components: Dict[str, Any], ui_config: Dict[str, Any], button_manager):
    """Execute download operation dengan backend service"""
    try:
        logger = ui_components.get('logger')
        _setup_progress_tracker(ui_components, "Dataset Download")
        
        # Create downloader
        downloader = create_backend_downloader(ui_config, logger)
        if not downloader:
            handle_ui_error(ui_components, "Gagal membuat download service", button_manager)
            return
        
        # Setup progress callback
        if hasattr(downloader, 'set_progress_callback'):
            downloader.set_progress_callback(ui_components['progress_callback'])
        
        # Log config
        from smartcash.ui.dataset.downloader.utils.ui_utils import log_download_config
        log_download_config(ui_components, ui_config)
        
        if logger:
            logger.info("🚀 Memulai download dataset")
        
        # Execute download
        result = downloader.download_dataset()
        
        if result and result.get('status') == 'success':
            from smartcash.ui.dataset.downloader.utils.ui_utils import show_download_success
            show_download_success(ui_components, result)
            show_ui_success(ui_components, "Download berhasil", button_manager)
        else:
            error_msg = result.get('message', 'Download gagal') if result else 'No response from service'
            handle_ui_error(ui_components, error_msg, button_manager)
            
    except Exception as e:
        handle_ui_error(ui_components, f"Error download operation: {str(e)}", button_manager)

def _execute_cleanup_operation(ui_components: Dict[str, Any], targets_result: Dict[str, Any], button_manager):
    """Execute cleanup operation dengan backend service"""
    try:
        logger = ui_components.get('logger')
        _setup_progress_tracker(ui_components, "Dataset Cleanup")
        
        # Create cleanup service
        cleanup_service = create_backend_cleanup_service(logger)
        if not cleanup_service:
            handle_ui_error(ui_components, "Gagal membuat cleanup service", button_manager)
            return
        
        # Setup progress callback
        cleanup_service.set_progress_callback(ui_components['progress_callback'])
        
        if logger:
            logger.info("🧹 Memulai cleanup dataset")
        
        # Execute cleanup
        result = cleanup_service.cleanup_dataset_files(targets_result.get('targets', {}))
        
        if result.get('status') == 'success':
            cleaned_count = len(result.get('cleaned_targets', []))
            show_ui_success(ui_components, f"Cleanup selesai: {cleaned_count} direktori dibersihkan", button_manager)
        else:
            handle_ui_error(ui_components, result.get('message', 'Cleanup failed'), button_manager)
            
    except Exception as e:
        handle_ui_error(ui_components, f"Error cleanup operation: {str(e)}", button_manager)

def _setup_progress_tracker(ui_components: Dict[str, Any], operation_name: str):
    """Setup progress tracker untuk operation"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.show(operation_name)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")