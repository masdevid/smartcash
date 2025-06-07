"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Fixed download handler dengan visible confirmation area
"""

from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import show_status_safe
from IPython.display import display, HTML, clear_output

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan dialog confirmation"""
    
    def execute_cleanup(button=None):
        button_manager = _get_button_manager(ui_components)
        logger = ui_components.get('logger')
        
        _clear_outputs(ui_components)
        button_manager.disable_buttons('cleanup_button')
        
        try:
            logger.info("🔍 Memulai scan untuk cleanup targets...")
            
            # Get cleanup targets dari backend
            from smartcash.dataset.downloader.dataset_scanner import create_dataset_scanner
            scanner = create_dataset_scanner(logger)
            targets_result = scanner.get_cleanup_targets()
            
            if targets_result.get('status') != 'success':
                _handle_ui_error(ui_components, "Gagal mendapatkan cleanup targets", button, button_manager)
                return
            
            summary = targets_result.get('summary', {})
            total_files = summary.get('total_files', 0)
            
            if total_files == 0:
                _show_ui_success(ui_components, "Tidak ada file untuk dibersihkan", button, button_manager)
                return
            
            # Log menunggu konfirmasi
            logger.info("⏳ Menunggu konfirmasi user untuk cleanup...")
            
            # Show confirmation dialog
            _show_cleanup_confirmation_dialog(ui_components, targets_result, button, button_manager)
            
        except Exception as e:
            _handle_ui_error(ui_components, f"Error saat cleanup: {str(e)}", button, button_manager)
    
    # Bind handler
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup)

def _show_cleanup_confirmation_dialog(ui_components: Dict[str, Any], targets_result: Dict[str, Any], button, button_manager):
    """Show cleanup confirmation menggunakan dialog component"""
    from smartcash.ui.components.dialogs import show_destructive_confirmation
    
    summary = targets_result.get('summary', {})
    targets = targets_result.get('targets', {})
    logger = ui_components.get('logger')
    
    # Build message dengan detail
    message_lines = [
        f"Akan menghapus {summary.get('total_files', 0):,} file ({summary.get('size_formatted', '0 B')})",
        "",
        "📂 Target cleanup:"
    ]
    
    # Add target details
    for target_name, target_info in targets.items():
        if target_info.get('file_count', 0) > 0:
            file_count = target_info.get('file_count', 0)
            size_formatted = target_info.get('size_formatted', '0 B')
            message_lines.append(f"  • {target_name}: {file_count:,} file ({size_formatted})")
    
    message_lines.extend([
        "",
        "⚠️ Direktori struktur akan tetap dipertahankan",
        "Lanjutkan cleanup?"
    ])
    
    def on_confirm(btn):
        """Handler saat konfirmasi cleanup"""
        if logger:
            logger.info("✅ User mengkonfirmasi cleanup")
        _execute_cleanup_with_backend(targets_result, ui_components, button, button_manager)
    
    def on_cancel(btn):
        """Handler saat batal cleanup - RESET SEMUA STATE"""
        if logger:
            logger.info("🚫 Cleanup dibatalkan oleh user")
        
        # Reset button state - enable semua buttons
        button_manager.enable_buttons()
        
        # Clear confirmation area
        _clear_confirmation_area(ui_components)
        
        # Reset progress tracker jika ada
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker and hasattr(progress_tracker, 'reset'):
            progress_tracker.reset()
        
        # Update status
        show_status_safe("🚫 Cleanup dibatalkan", "info", ui_components)
    
    # Show dialog menggunakan component
    show_destructive_confirmation(
        title="Konfirmasi Cleanup Dataset",
        message='\n'.join(message_lines),
        item_name="data preprocessed",
        on_confirm=on_confirm,
        on_cancel=on_cancel
    )

def _execute_cleanup_with_backend(targets_result: Dict[str, Any], ui_components: Dict[str, Any], button, button_manager):
    """Execute cleanup dengan backend service"""
    try:
        logger = ui_components.get('logger')
        
        if logger:
            logger.info("🧹 Memulai cleanup dataset...")
        
        # Show progress tracker
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            progress_tracker.show("Dataset Cleanup")
        
        # Use backend cleanup service
        from smartcash.dataset.downloader.cleanup_service import create_cleanup_service
        cleanup_service = create_cleanup_service(logger)
        cleanup_service.set_progress_callback(ui_components['progress_callback'])
        
        # Execute cleanup
        result = cleanup_service.cleanup_dataset_files(targets_result.get('targets', {}))
        
        if result.get('status') == 'success':
            cleaned_count = len(result.get('cleaned_targets', []))
            _show_ui_success(ui_components, f"Cleanup selesai: {cleaned_count} direktori dibersihkan", button, button_manager)
        else:
            _handle_ui_error(ui_components, result.get('message', 'Cleanup failed'), button, button_manager)
            
    except Exception as e:
        _handle_ui_error(ui_components, f"Error saat cleanup: {str(e)}", button, button_manager)

# Main setup function
def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan dialog confirmation"""
    
    def create_progress_callback():
        def progress_callback(step: str, current: int, total: int, message: str):
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    if hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(current, message)
                    if hasattr(progress_tracker, 'update_current'):
                        step_progress = _map_step_to_current_progress(step, current)
                        progress_tracker.update_current(step_progress, f"Step: {step}")
                
                if _is_milestone_step(step, current):
                    logger = ui_components.get('logger')
                    if logger:
                        logger.info(message)
            except Exception:
                pass
        
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Setup handlers
    setup_download_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)
    
    # Setup config handlers
    def save_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _handle_ui_error(ui_components, "Config handler tidak tersedia", button, None)
                return
            
            success = config_handler.save_config(ui_components)
            if success:
                _show_ui_success(ui_components, "✅ Konfigurasi berhasil disimpan", button, None)
            else:
                _handle_ui_error(ui_components, "❌ Gagal menyimpan konfigurasi", button, None)
        except Exception as e:
            _handle_ui_error(ui_components, f"❌ Error saat save: {str(e)}", button, None)
    
    def reset_config_handler(button=None):
        try:
            config_handler = ui_components.get('config_handler')
            if not config_handler:
                _handle_ui_error(ui_components, "Config handler tidak tersedia", button, None)
                return
            
            success = config_handler.reset_config(ui_components)
            if success:
                _show_ui_success(ui_components, "🔄 Konfigurasi berhasil direset", button, None)
            else:
                _handle_ui_error(ui_components, "❌ Gagal reset konfigurasi", button, None)
        except Exception as e:
            _handle_ui_error(ui_components, f"❌ Error saat reset: {str(e)}", button, None)
    
    # Bind handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config_handler)
    if reset_button:
        reset_button.on_click(reset_config_handler)
    
    return ui_components

# Placeholder functions for complete handlers
def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Download handler implementation exists in original file"""
    pass

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Check handler implementation exists in original file"""  
    pass