"""
File: smartcash/ui/dataset/preprocessing/handlers/stop_handler.py
Deskripsi: Handler untuk operasi stop/cancel preprocessing dengan cleanup futures dan resources
"""

from typing import Dict, Any, Optional
from concurrent.futures import Future
from smartcash.common.logger import get_logger

class StopHandler:
    """Handler untuk menghentikan operasi preprocessing yang sedang berjalan."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Inisialisasi stop handler dengan komponen UI."""
        self.ui_components = ui_components
        self.logger = get_logger('smartcash.ui.dataset.preprocessing.stop')
    
    def handle_stop_click(self, button: Any) -> None:
        """
        Handler untuk tombol stop preprocessing.
        
        Args:
            button: Widget tombol yang diklik
        """
        # Disable tombol untuk mencegah multiple click
        if button and hasattr(button, 'disabled'):
            button.disabled = True
        
        try:
            # Set flag stop_requested di UI components
            self.ui_components['stop_requested'] = True
            
            # Log stop request
            self.logger.warning("⏹️ Permintaan stop preprocessing diterima")
            self._log_to_ui("Menghentikan preprocessing...", "warning", "⏹️")
            
            # Update status panel
            self._update_status_panel("warning", "Menghentikan preprocessing...")
            
            # Stop preprocessing yang sedang berjalan
            self._stop_running_processes()
            
            # Sembunyikan tombol stop
            self._hide_stop_button()
            
            # Reset UI state setelah stop
            self._reset_ui_after_stop()
            
        except Exception as e:
            error_message = f"Error saat menghentikan preprocessing: {str(e)}"
            self.logger.error(f"❌ {error_message}")
            self._log_to_ui(error_message, "error", "❌")
        
        finally:
            # Re-enable tombol
            if button and hasattr(button, 'disabled'):
                button.disabled = False
    
    def _stop_running_processes(self) -> None:
        """Stop semua proses yang sedang berjalan."""
        try:
            # Stop preprocessing service jika ada
            self._stop_preprocessing_service()
            
            # Stop cleanup service jika ada
            self._stop_cleanup_service()
            
            # Cancel futures yang sedang berjalan
            self._cancel_running_futures()
            
            # Notify observer system
            self._notify_stop_to_observers()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error saat stop processes: {str(e)}")
    
    def _stop_preprocessing_service(self) -> None:
        """Stop preprocessing service yang sedang berjalan."""
        try:
            # Cek apakah ada service runner yang aktif
            if 'main_handler' in self.ui_components:
                main_handler = self.ui_components['main_handler']
                if hasattr(main_handler, 'stop_processing'):
                    main_handler.stop_processing()
                    self.logger.info("🛑 Preprocessing service dihentikan")
            
            # Set flag preprocessing_running ke False
            self.ui_components['preprocessing_running'] = False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error stop preprocessing service: {str(e)}")
    
    def _stop_cleanup_service(self) -> None:
        """Stop cleanup service yang sedang berjalan."""
        try:
            # Cek apakah ada cleanup handler yang aktif
            if 'cleanup_handler' in self.ui_components:
                cleanup_handler = self.ui_components['cleanup_handler']
                if hasattr(cleanup_handler, 'stop_cleanup'):
                    cleanup_handler.stop_cleanup()
                    self.logger.info("🧹 Cleanup service dihentikan")
            
            # Set flag cleanup_running ke False
            self.ui_components['cleanup_running'] = False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error stop cleanup service: {str(e)}")
    
    def _cancel_running_futures(self) -> None:
        """Cancel futures yang masih berjalan."""
        try:
            # List semua futures yang mungkin aktif
            future_keys = [
                'preprocessing_future',
                'cleanup_future', 
                'service_future',
                'main_future'
            ]
            
            cancelled_count = 0
            for future_key in future_keys:
                if future_key in self.ui_components:
                    future = self.ui_components[future_key]
                    if isinstance(future, Future) and not future.done():
                        try:
                            if future.cancel():
                                cancelled_count += 1
                                self.logger.debug(f"🔄 {future_key} berhasil dibatalkan")
                            else:
                                self.logger.debug(f"⚠️ {future_key} tidak dapat dibatalkan (mungkin sudah selesai)")
                        except Exception as e:
                            self.logger.debug(f"⚠️ Error cancel {future_key}: {str(e)}")
            
            if cancelled_count > 0:
                self.logger.info(f"🔄 {cancelled_count} futures berhasil dibatalkan")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error cancel futures: {str(e)}")
    
    def _notify_stop_to_observers(self) -> None:
        """Notify observer system tentang stop request."""
        try:
            # Notify observer manager
            observer_manager = self.ui_components.get('observer_manager')
            if observer_manager:
                # Set flag pada observer manager
                if hasattr(observer_manager, 'set_flag'):
                    observer_manager.set_flag('stop_requested', True)
                
                # Notify melalui observer system
                if hasattr(observer_manager, 'notify'):
                    try:
                        observer_manager.notify('preprocessing.stop', {
                            'reason': 'user_request',
                            'timestamp': self._get_current_timestamp()
                        })
                    except Exception as e:
                        self.logger.debug(f"Observer notify error: {str(e)}")
            
            # Notify notification manager jika ada
            if 'notification_manager' in self.ui_components:
                notification_manager = self.ui_components['notification_manager']
                if hasattr(notification_manager, 'notify_process_stop'):
                    notification_manager.notify_process_stop("Stop oleh pengguna")
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error notify observers: {str(e)}")
    
    def _hide_stop_button(self) -> None:
        """Sembunyikan tombol stop setelah operasi stop."""
        try:
            if 'stop_button' in self.ui_components:
                stop_button = self.ui_components['stop_button']
                if hasattr(stop_button, 'layout'):
                    stop_button.layout.display = 'none'
                if hasattr(stop_button, 'disabled'):
                    stop_button.disabled = True
                    
            self.logger.debug("👁️ Tombol stop disembunyikan")
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error hide stop button: {str(e)}")
    
    def _reset_ui_after_stop(self) -> None:
        """Reset UI state setelah operasi stop."""
        try:
            # Re-enable tombol utama
            buttons_to_enable = [
                'preprocess_button', 'save_button', 'reset_button', 'cleanup_button'
            ]
            
            for button_key in buttons_to_enable:
                if button_key in self.ui_components:
                    button = self.ui_components[button_key]
                    if hasattr(button, 'disabled'):
                        button.disabled = False
            
            # Reset progress bars
            self._reset_progress_displays()
            
            # Clear confirmation area
            self._clear_confirmation_area()
            
            # Reset flags
            self.ui_components['preprocessing_running'] = False
            self.ui_components['cleanup_running'] = False
            
            # Update status
            self._update_status_panel("warning", "Preprocessing dihentikan oleh pengguna")
            
            self.logger.info("🔄 UI berhasil direset setelah stop")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error reset UI after stop: {str(e)}")
    
    def _reset_progress_displays(self) -> None:
        """Reset semua progress displays."""
        try:
            # Reset progress bars
            progress_components = [
                'progress_bar', 'current_progress', 'overall_progress'
            ]
            
            for progress_key in progress_components:
                if progress_key in self.ui_components:
                    progress = self.ui_components[progress_key]
                    if hasattr(progress, 'value'):
                        progress.value = 0
                    if hasattr(progress, 'layout'):
                        progress.layout.visibility = 'hidden'
            
            # Reset progress labels
            label_components = [
                'overall_label', 'step_label', 'progress_message'
            ]
            
            for label_key in label_components:
                if label_key in self.ui_components:
                    label = self.ui_components[label_key]
                    if hasattr(label, 'value'):
                        label.value = ""
                    if hasattr(label, 'layout'):
                        label.layout.visibility = 'hidden'
                        
        except Exception as e:
            self.logger.debug(f"⚠️ Error reset progress displays: {str(e)}")
    
    def _clear_confirmation_area(self) -> None:
        """Bersihkan area konfirmasi."""
        try:
            if 'confirmation_area' in self.ui_components:
                confirmation_area = self.ui_components['confirmation_area']
                if hasattr(confirmation_area, 'clear_output'):
                    confirmation_area.clear_output(wait=True)
                if hasattr(confirmation_area, 'layout'):
                    confirmation_area.layout.display = 'none'
                    
        except Exception as e:
            self.logger.debug(f"⚠️ Error clear confirmation area: {str(e)}")
    
    def _log_to_ui(self, message: str, level: str = 'info', icon: str = '') -> None:
        """Log pesan ke UI."""
        try:
            # Gunakan logger handler jika tersedia
            if 'logger_handler' in self.ui_components:
                logger_handler = self.ui_components['logger_handler']
                logger_handler.log(message, level, icon)
            # Fallback ke log_message function
            elif 'log_message' in self.ui_components:
                log_func = self.ui_components['log_message']
                log_func(message, level, icon)
            # Fallback ke UI logger langsung
            else:
                try:
                    from smartcash.ui.utils.ui_logger import log_to_ui
                    log_to_ui(self.ui_components, message, level, icon)
                except ImportError:
                    pass  # Skip UI logging jika tidak tersedia
                    
        except Exception as e:
            self.logger.debug(f"⚠️ Error log to UI: {str(e)}")
    
    def _update_status_panel(self, status: str, message: str) -> None:
        """Update status panel UI."""
        try:
            if 'status_panel' in self.ui_components:
                try:
                    from smartcash.ui.utils.alert_utils import update_status_panel
                    update_status_panel(self.ui_components['status_panel'], message, status)
                except ImportError:
                    # Fallback manual update
                    self.ui_components['status_panel'].value = f"<div class='alert alert-{status}'>{message}</div>"
                    
        except Exception as e:
            self.logger.debug(f"⚠️ Error update status panel: {str(e)}")
    
    def _get_current_timestamp(self) -> str:
        """Dapatkan timestamp saat ini untuk logging."""
        try:
            from datetime import datetime
            return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return "unknown"
    
    def is_stop_requested(self) -> bool:
        """
        Cek apakah stop sudah direquest.
        
        Returns:
            True jika stop direquest, False jika tidak
        """
        return self.ui_components.get('stop_requested', False)
    
    def reset_stop_flag(self) -> None:
        """Reset flag stop_requested."""
        self.ui_components['stop_requested'] = False
        self.logger.debug("🔄 Stop flag berhasil direset")
    
    def cleanup_resources(self) -> None:
        """Cleanup resources saat stop handler tidak digunakan lagi."""
        try:
            # Cancel semua futures yang masih aktif
            self._cancel_running_futures()
            
            # Reset semua flags
            self.ui_components['stop_requested'] = False
            self.ui_components['preprocessing_running'] = False
            self.ui_components['cleanup_running'] = False
            
            self.logger.debug("🧹 Stop handler resources berhasil dibersihkan")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error cleanup stop handler: {str(e)}")

# Factory function untuk membuat stop handler
def create_stop_handler(ui_components: Dict[str, Any]) -> StopHandler:
    """
    Factory function untuk membuat stop handler.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance StopHandler yang siap digunakan
    """
    return StopHandler(ui_components)

# Helper function untuk setup stop handler pada button
def setup_stop_button(ui_components: Dict[str, Any]) -> None:
    """
    Setup stop button dengan handler.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    if 'stop_button' not in ui_components:
        return
    
    # Buat stop handler
    stop_handler = create_stop_handler(ui_components)
    
    # Simpan reference untuk cleanup nanti
    ui_components['stop_handler'] = stop_handler
    
    # Setup button click handler
    ui_components['stop_button'].on_click(
        lambda b: stop_handler.handle_stop_click(b)
    )
    
    # Pastikan tombol stop tersembunyi di awal
    if hasattr(ui_components['stop_button'], 'layout'):
        ui_components['stop_button'].layout.display = 'none'

# Helper function untuk cek status stop
def is_preprocessing_stopped(ui_components: Dict[str, Any]) -> bool:
    """
    Helper function untuk cek apakah preprocessing dihentikan.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        True jika dihentikan, False jika masih berjalan
    """
    if 'stop_handler' in ui_components:
        return ui_components['stop_handler'].is_stop_requested()
    
    return ui_components.get('stop_requested', False)