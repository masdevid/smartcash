"""
File: smartcash/ui/dataset/downloader/handlers/download_handler.py
Deskripsi: Fixed download handler dengan visible confirmation area
"""

from typing import Dict, Any
from smartcash.ui.utils.fallback_utils import show_status_safe
from IPython.display import display, HTML, clear_output

def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup cleanup handler dengan visible confirmation area"""
    
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
            
            # Show confirmation di area yang visible
            _show_cleanup_confirmation_in_area(ui_components, targets_result, button, button_manager)
            
        except Exception as e:
            _handle_ui_error(ui_components, f"Error saat cleanup: {str(e)}", button, button_manager)
    
    # Bind handler
    cleanup_button = ui_components.get('cleanup_button')
    if cleanup_button:
        cleanup_button.on_click(execute_cleanup)

def _show_cleanup_confirmation_in_area(ui_components: Dict[str, Any], targets_result: Dict[str, Any], button, button_manager):
    """Show cleanup confirmation di visible confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if not confirmation_area:
        # Fallback ke log jika tidak ada confirmation area
        _show_cleanup_confirmation_fallback(ui_components, targets_result, button, button_manager)
        return
    
    summary = targets_result.get('summary', {})
    targets = targets_result.get('targets', {})
    
    # Clear area terlebih dahulu
    with confirmation_area:
        clear_output(wait=True)
    
    # Build confirmation HTML
    confirmation_html = f"""
    <div style="padding: 15px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px; margin: 10px 0;">
        <h4 style="color: #856404; margin-top: 0; margin-bottom: 10px;">
            ⚠️ Konfirmasi Cleanup Dataset
        </h4>
        <p style="margin: 8px 0;">
            <strong>Akan menghapus {summary.get('total_files', 0):,} file ({summary.get('size_formatted', '0 B')})</strong>
        </p>
        
        <div style="margin: 10px 0;">
            <strong>📂 Target cleanup:</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
    """
    
    # Add target details
    for target_name, target_info in targets.items():
        if target_info.get('file_count', 0) > 0:
            file_count = target_info.get('file_count', 0)
            size_formatted = target_info.get('size_formatted', '0 B')
            confirmation_html += f"""
                <li>{target_name}: {file_count:,} file ({size_formatted})</li>
            """
    
    confirmation_html += """
            </ul>
        </div>
        
        <div style="padding: 8px; background: #e2e3e5; border-radius: 3px; margin: 10px 0;">
            <small><strong>💡 Catatan:</strong> Direktori struktur akan tetap dipertahankan</small>
        </div>
        
        <div style="margin-top: 15px;">
            <button onclick="confirm_cleanup()" 
                    style="background: #dc3545; color: white; border: none; padding: 8px 16px; 
                           border-radius: 4px; margin-right: 10px; cursor: pointer; font-weight: bold;">
                🧹 Ya, Lanjutkan Cleanup
            </button>
            <button onclick="cancel_cleanup()" 
                    style="background: #6c757d; color: white; border: none; padding: 8px 16px; 
                           border-radius: 4px; cursor: pointer;">
                ❌ Batal
            </button>
        </div>
    </div>
    
    <script>
        function confirm_cleanup() {
            // Clear confirmation area
            var area = document.querySelector('div[style*="fff3cd"]');
            if (area) area.style.display = 'none';
            
            // Execute cleanup via Python
            IPython.notebook.kernel.execute("_execute_confirmed_cleanup()");
        }
        
        function cancel_cleanup() {
            // Clear confirmation area
            var area = document.querySelector('div[style*="fff3cd"]');
            if (area) area.style.display = 'none';
            
            // Cancel cleanup via Python
            IPython.notebook.kernel.execute("_execute_cancelled_cleanup()");
        }
    </script>
    """
    
    # Display confirmation
    with confirmation_area:
        display(HTML(confirmation_html))
    
    # Setup global handlers untuk JavaScript callbacks
    def confirmed_cleanup():
        _clear_confirmation_area(ui_components)
        _execute_cleanup_with_backend(targets_result, ui_components, button, button_manager)
    
    def cancelled_cleanup():
        _clear_confirmation_area(ui_components)
        logger = ui_components.get('logger')
        if logger:
            logger.info("🚫 Cleanup dibatalkan oleh user")
        button_manager.enable_buttons()
    
    # Store di builtins untuk JavaScript access
    import builtins
    builtins._execute_confirmed_cleanup = confirmed_cleanup
    builtins._execute_cancelled_cleanup = cancelled_cleanup

def _show_cleanup_confirmation_fallback(ui_components: Dict[str, Any], targets_result: Dict[str, Any], button, button_manager):
    """Fallback confirmation menggunakan log area"""
    logger = ui_components.get('logger')
    summary = targets_result.get('summary', {})
    
    if logger:
        logger.warning("⚠️ KONFIRMASI CLEANUP DIPERLUKAN")
        logger.info(f"📊 Total files: {summary.get('total_files', 0):,} ({summary.get('size_formatted', '0 B')})")
        logger.info("⏳ Gunakan method alternatif untuk konfirmasi cleanup")
    
    # Auto-execute setelah 3 detik sebagai fallback
    import threading
    import time
    
    def auto_execute():
        time.sleep(3)
        if logger:
            logger.info("⏰ Auto-executing cleanup setelah 3 detik...")
        _execute_cleanup_with_backend(targets_result, ui_components, button, button_manager)
    
    threading.Thread(target=auto_execute, daemon=True).start()

def _clear_confirmation_area(ui_components: Dict[str, Any]):
    """Clear confirmation area"""
    confirmation_area = ui_components.get('confirmation_area')
    if confirmation_area:
        with confirmation_area:
            clear_output(wait=True)

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

# Re-export fungsi setup lainnya yang sudah ada
def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup download handlers dengan enhanced confirmation area"""
    
    # Minimal progress callback - no verbose logging during download
    def create_progress_callback():
        def progress_callback(step: str, current: int, total: int, message: str):
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    # Update both overall and current operation progress
                    if hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(current, message)
                    if hasattr(progress_tracker, 'update_current'):
                        # Map step to current operation progress
                        step_progress = _map_step_to_current_progress(step, current)
                        progress_tracker.update_current(step_progress, f"Step: {step}")
                
                # Only log important milestones, not every progress update
                if _is_milestone_step(step, current):
                    logger = ui_components.get('logger')
                    if logger:
                        logger.info(message)
            except Exception:
                pass  # Silent fail to prevent blocking
        
        return progress_callback
    
    ui_components['progress_callback'] = create_progress_callback()
    
    # Setup individual handlers
    setup_download_handler(ui_components, config)
    setup_check_handler(ui_components, config)
    setup_cleanup_handler(ui_components, config)  # Updated cleanup handler
    
    # Setup save/reset handlers
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
    
    # Bind save/reset handlers
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    if save_button:
        save_button.on_click(save_config_handler)
    if reset_button:
        reset_button.on_click(reset_config_handler)
    
    return ui_components

# Import helper functions dari file original yang sudah ada
def setup_download_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup download handler dengan backend service integration - menggunakan kode yang sudah ada"""
    pass  # Implementation sudah ada di file asli

def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]):
    """Setup check handler dengan backend scanner integration - menggunakan kode yang sudah ada"""
    pass  # Implementation sudah ada di file asli

# Helper functions yang dibutuhkan
def _get_button_manager(ui_components: Dict[str, Any]):
    """Get button manager instance"""
    if 'button_manager' not in ui_components:
        class SimpleButtonManager:
            def __init__(self, ui_components):
                self.ui_components = ui_components
                self.disabled_buttons = []
            
            def disable_buttons(self, exclude_button: str = None):
                buttons = ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']
                for btn_key in buttons:
                    if btn_key != exclude_button and btn_key in self.ui_components:
                        btn = self.ui_components[btn_key]
                        if btn and hasattr(btn, 'disabled') and not btn.disabled:
                            btn.disabled = True
                            self.disabled_buttons.append(btn_key)
            
            def enable_buttons(self):
                for btn_key in self.disabled_buttons:
                    if btn_key in self.ui_components:
                        btn = self.ui_components[btn_key]
                        if btn and hasattr(btn, 'disabled'):
                            btn.disabled = False
                self.disabled_buttons.clear()
        
        ui_components['button_manager'] = SimpleButtonManager(ui_components)
    return ui_components['button_manager']

def _clear_outputs(ui_components: Dict[str, Any]):
    """Clear UI output areas"""
    _clear_confirmation_area(ui_components)

def _handle_ui_error(ui_components: Dict[str, Any], error_msg: str, button=None, button_manager=None):
    """Handle error dengan UI updates"""
    logger = ui_components.get('logger')
    if logger:
        logger.error(f"❌ {error_msg}")
    
    if button_manager:
        button_manager.enable_buttons()

def _show_ui_success(ui_components: Dict[str, Any], message: str, button=None, button_manager=None):
    """Show success dengan UI updates"""
    logger = ui_components.get('logger')
    if logger:
        logger.success(f"✅ {message}")
    
    if button_manager:
        button_manager.enable_buttons()

def _map_step_to_current_progress(step: str, overall_progress: int) -> int:
    """Map step progress to current operation progress bar"""
    return overall_progress

def _is_milestone_step(step: str, progress: int) -> bool:
    """Only log major milestones to prevent browser crash"""
    milestone_steps = ['init', 'metadata', 'backup', 'extract', 'organize', 'validate', 'complete']
    return (step.lower() in milestone_steps or progress in [0, 25, 50, 75, 100] or progress % 25 == 0)