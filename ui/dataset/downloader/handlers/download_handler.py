"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Simplified download handler dengan shared dialog API seperti preprocessing
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.downloader.handlers.error_handling import (
    with_downloader_error_handling,
    handle_downloader_error,
    DOWNLOAD_CONTEXT,
    CLEANUP_CONTEXT
)
from smartcash.ui.dataset.downloader.utils.ui_utils import clear_outputs, log_to_accordion
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

@with_downloader_error_handling(operation="setup_download_handler")
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
                error_msg = f"Config tidak valid: {', '.join(validation['errors'])}"
                log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
                button_manager.enable_buttons()
                return
            
            # Check existing dataset
            has_existing, total_images, summary_data = check_existing_dataset(ui_components.get('logger_bridge'))
            
            if has_existing:
                _show_download_confirmation(ui_components, ui_config, total_images)
            else:
                _execute_download_operation(ui_components, ui_config, button_manager)
                
        except Exception as e:
            handle_downloader_error(
                e,
                DOWNLOAD_CONTEXT,
                logger=ui_components.get('logger_bridge'),
                ui_components=ui_components
            )
            button_manager.enable_buttons()
    
    download_button = ui_components.get('download_button')
    if download_button:
        download_button.on_click(execute_download)

@with_downloader_error_handling(operation="setup_check_handler")
def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup check handler dengan backend scanner"""
    
    def execute_check(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('check_button')
        
        try:
            # Check existing dataset
            has_existing, total_images, summary_data = check_existing_dataset(
                ui_components.get('logger_bridge')
            )
            
            if has_existing and summary_data:
                from smartcash.ui.dataset.downloader.utils.ui_utils import display_check_results
                display_check_results(ui_components, summary_data)
            else:
                log_to_accordion(ui_components, "ℹ️ Tidak ada dataset yang ditemukan", 'info')
                
        except Exception as e:
            handle_downloader_error(
                e,
                DOWNLOAD_CONTEXT._replace(operation="check_dataset"),
                logger=ui_components.get('logger_bridge'),
                ui_components=ui_components
            )
        finally:
            button_manager.enable_buttons()
    
    # Register handler
    if 'check_button' in ui_components and ui_components['check_button']:
        ui_components['check_button'].on_click(execute_check)

@with_downloader_error_handling(operation="setup_cleanup_handler")
def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan dialog confirmation"""
    
    def execute_cleanup(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            # Get cleanup targets
            targets_result = get_cleanup_targets(ui_components.get('logger_bridge'))
            
            if targets_result.get('has_targets', False):
                _show_cleanup_confirmation(ui_components, targets_result)
            else:
                log_to_accordion(ui_components, "ℹ️ Tidak ada target cleanup yang ditemukan", 'info')
                
        except Exception as e:
            handle_downloader_error(
                e,
                CLEANUP_CONTEXT._replace(operation="check_cleanup_targets"),
                logger=ui_components.get('logger_bridge'),
                ui_components=ui_components
            )
        finally:
            button_manager.enable_buttons()
    
    # Register handler
    if 'cleanup_button' in ui_components and ui_components['cleanup_button']:
        ui_components['cleanup_button'].on_click(execute_cleanup)

@with_downloader_error_handling(operation="setup_config_handlers")
def setup_config_handlers(ui_components: Dict[str, Any]):
    """Setup save/reset handlers"""
    
    def save_config_handler(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('save_button')
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                result = config_handler.save_config(ui_components)
                if result.get('success'):
                    log_to_accordion(ui_components, "✅ Konfigurasi berhasil disimpan", 'success')
                else:
                    error_msg = f"Gagal menyimpan konfigurasi: {result.get('error')}"
                    log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
            else:
                log_to_accordion(ui_components, "❌ Config handler tidak tersedia", 'error')
                
        except Exception as e:
            handle_downloader_error(
                e,
                DOWNLOAD_CONTEXT._replace(operation="save_config"),
                logger=ui_components.get('logger_bridge'),
                ui_components=ui_components
            )
        finally:
            button_manager.enable_buttons()
    
    def reset_config_handler(button=None):
        button_manager = get_button_manager(ui_components)
        clear_outputs(ui_components)
        button_manager.disable_buttons('reset_button')
        
        try:
            config_handler = ui_components.get('config_handler')
            if config_handler:
                result = config_handler.reset_config(ui_components)
                if result.get('success'):
                    log_to_accordion(ui_components, "✅ Konfigurasi berhasil direset ke default", 'success')
                else:
                    error_msg = f"Gagal mereset konfigurasi: {result.get('error')}"
                    log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
            else:
                log_to_accordion(ui_components, "❌ Config handler tidak tersedia", 'error')
                
        except Exception as e:
            handle_downloader_error(
                e,
                DOWNLOAD_CONTEXT._replace(operation="reset_config"),
                logger=ui_components.get('logger_bridge'),
                ui_components=ui_components
            )
        finally:
            button_manager.enable_buttons()
    
    # Register handlers
    if 'save_button' in ui_components and ui_components['save_button']:
        ui_components['save_button'].on_click(save_config_handler)
    
    if 'reset_button' in ui_components and ui_components['reset_button']:
        ui_components['reset_button'].on_click(reset_config_handler)

# === CONFIRMATION HANDLERS ===

def _show_download_confirmation(ui_components: Dict[str, Any], ui_config: Dict[str, Any], existing_count: int):
    """Show download confirmation menggunakan shared dialog API"""
    try:
        from smartcash.ui.components import show_confirmation_dialog
        
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
        from smartcash.ui.components import show_confirmation_dialog
        
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

@with_downloader_error_handling(operation="execute_download_operation")
def _execute_download_operation(ui_components: Dict[str, Any], ui_config: Dict[str, Any], button_manager):
    """Execute download operation dengan backend service"""
    try:
        # Setup progress tracker
        _setup_progress_tracker(ui_components, "download")
        
        # Create downloader instance
        downloader = create_backend_downloader(
            ui_config,
            ui_components.get('logger_bridge')
        )
        
        if not downloader:
            raise ValueError("Gagal membuat downloader instance")
        
        # Execute download
        result = downloader.download_dataset()
        
        if result.get('success'):
            log_to_accordion(ui_components, "✅ Download selesai", 'success')
            
            # Tampilkan summary
            if 'summary' in result:
                from smartcash.ui.dataset.downloader.utils.ui_utils import display_check_results
                display_check_results(ui_components, result['summary'])
        else:
            error_msg = result.get('error', 'Gagal mendownload dataset')
            log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
            raise Exception(error_msg)
            
    except Exception as e:
        handle_downloader_error(
            e,
            DOWNLOAD_CONTEXT._replace(operation="download_dataset"),
            logger=ui_components.get('logger_bridge'),
            ui_components=ui_components
        )
        raise
    finally:
        if button_manager:
            button_manager.enable_buttons()

@with_downloader_error_handling(operation="execute_cleanup_operation")
def _execute_cleanup_operation(ui_components: Dict[str, Any], targets_result: Dict[str, Any], button_manager):
    """Execute cleanup operation dengan backend service"""
    try:
        # Setup progress tracker
        _setup_progress_tracker(ui_components, "cleanup")
        
        # Create cleanup service
        cleanup_service = create_backend_cleanup_service(ui_components.get('logger_bridge'))
        
        if not cleanup_service:
            raise ValueError("Gagal membuat cleanup service")
        
        # Execute cleanup
        result = cleanup_service.cleanup(targets_result.get('targets', []))
        
        if result.get('success'):
            log_to_accordion(ui_components, "✅ Cleanup selesai", 'success')
            
            # Tampilkan summary
            if 'summary' in result:
                log_to_accordion(ui_components, result['summary'], 'info')
        else:
            error_msg = result.get('error', 'Gagal membersihkan dataset')
            log_to_accordion(ui_components, f"❌ {error_msg}", 'error')
            raise Exception(error_msg)
            
    except Exception as e:
        handle_downloader_error(
            e,
            CLEANUP_CONTEXT._replace(operation="cleanup_dataset"),
            logger=ui_components.get('logger_bridge'),
            ui_components=ui_components
        )
        raise
    finally:
        if button_manager:
            button_manager.enable_buttons()

def _setup_progress_tracker(ui_components: Dict[str, Any], operation_name: str):
    """Setup progress tracker untuk operation"""
    progress_tracker = ui_components.get('progress_tracker')
    if progress_tracker:
        progress_tracker.show(operation_name)
        progress_tracker.update_overall(0, f"🚀 Memulai {operation_name.lower()}...")