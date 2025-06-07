"""
File: smartcash/ui/dataset/downloader/handlers/button_handler.py
Deskripsi: Handler untuk tombol-tombol UI dataset downloader dengan proper state management
"""

import asyncio
from typing import Dict, Any, List, Tuple, Callable

from smartcash.ui.utils.fallback_utils import show_status_safe
from smartcash.ui.dataset.downloader.handlers.base_handler import BaseDatasetDownloadHandler


class ButtonHandler(BaseDatasetDownloadHandler):
    """
    Handler untuk tombol-tombol UI dataset downloader
    Menyediakan fungsionalitas untuk mengelola state tombol dan event handling
    """
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        # Panggil konstruktor parent class
        super().__init__(ui_components, logger)
        
        # Inisialisasi properti spesifik ButtonHandler
        self.button_configs = []
    
    def register_button_configs(self, button_configs: List[Tuple[str, Callable]]) -> None:
        """
        Mendaftarkan konfigurasi tombol
        
        Args:
            button_configs: List tuple (nama_tombol, handler_function)
        """
        self.button_configs = button_configs
    
    def _get_all_buttons(self):
        """O(n) but no tuple unpacking overhead"""
        return [self.ui_components.get(key) for key, _ in self.button_configs]
    
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup handlers dengan proper binding dan state management"""
        """O(n) single iteration"""
        for key, handler in self.button_configs:
            if button := self.ui_components.get(key):
                self._clear_button_handlers(button)
                button.on_click(handler)
        
        self.logger.success("✅ Handlers berhasil disetup dengan state management")
        return self.ui_components
    
    def _clear_button_handlers(self, button) -> None:
        """Clear existing handlers"""
        try:
            if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'callbacks'):
                button._click_handlers.callbacks.clear()
        except Exception:
            pass

    def _change_all_buttons_stat(self, enabled=True):
        """Mengubah status semua tombol"""
        for btn in self._get_all_buttons():
            if btn and hasattr(btn, 'disabled'):
                btn.disabled = not enabled

    def _prepare_button_state(self, active_button) -> None:
        """Prepare button state dan clear log"""
        # Clear log output
        log_output = self.ui_components.get('log_output')
        if log_output and hasattr(log_output, 'clear_output'):
            with log_output:
                log_output.clear_output(wait=True)
        # Panggil metode parent untuk mengubah state button
        super()._prepare_button_state(active_button, "Processing...")
        self._change_all_buttons_stat(False)
    
    def _restore_button_state(self) -> None:
        """Restore button state setelah operation"""
        super()._restore_button_state()
        self._change_all_buttons_stat(True)
    
    def handle_save_click(self, button) -> None:
        """Handle save button click dengan proper logging"""
        try:
            self._prepare_button_state(button)
            
            # Get config handler
            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self._handle_error("Config handler tidak tersedia", button)
                return
            
            # Save config
            success = config_handler.save_config(self.ui_components)
            
            if success:
                success_msg = "✅ Konfigurasi berhasil disimpan"
                self.logger.success(success_msg)
                show_status_safe(success_msg, "success", self.ui_components)
            else:
                self._handle_error("Gagal menyimpan konfigurasi", button)
                
        except Exception as e:
            self._handle_error(f"Error saat menyimpan: {str(e)}", button)
        finally:
            self._restore_button_state()
    
    def handle_reset_click(self, button) -> None:
        """Handle reset button click dengan proper logging"""
        try:
            self._prepare_button_state(button)
            
            # Get config handler
            config_handler = self.ui_components.get('config_handler')
            if not config_handler:
                self._handle_error("Config handler tidak tersedia", button)
                return
            
            # Reset config
            success = config_handler.reset_config(self.ui_components)
            
            if success:
                success_msg = "🔄 Konfigurasi berhasil direset ke default"
                self.logger.success(success_msg)
                show_status_safe(success_msg, "success", self.ui_components)
            else:
                self._handle_error("Gagal mereset konfigurasi", button)
                
        except Exception as e:
            self._handle_error(f"Error saat reset: {str(e)}", button)
        finally:
            self._restore_button_state()
    
    async def run_async_handler(self, handler_func, button, *args, **kwargs):
        """
        Menjalankan handler async dengan penanganan event loop yang tepat
        
        Args:
            handler_func: Fungsi handler async yang akan dijalankan
            button: Button yang memicu handler
            *args, **kwargs: Parameter tambahan untuk handler
        """
        try:
            # Coba dapatkan event loop yang sedang berjalan
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Jika loop sudah berjalan (di Jupyter), gunakan asyncio.ensure_future
                return asyncio.ensure_future(handler_func(button, *args, **kwargs))
            else:
                # Jika tidak ada loop yang berjalan, jalankan hingga selesai
                return loop.run_until_complete(handler_func(button, *args, **kwargs))
        except RuntimeError:
            # Jika tidak ada event loop, buat baru
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(handler_func(button, *args, **kwargs))


# Export
__all__ = ['ButtonHandler']
