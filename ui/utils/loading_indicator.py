"""
File: smartcash/ui/utils/loading_indicator.py
Deskripsi: Komponen indikator loading animasi untuk UI
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import threading
import time
from typing import Optional, Dict, Any, Callable

from smartcash.ui.utils.constants import ICONS
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class LoadingIndicator:
    """Kelas untuk menampilkan indikator loading animasi."""
    
    def __init__(
        self, 
        message: str = "Memuat...", 
        is_indeterminate: bool = False,
        max_value: int = 100,
        auto_hide: bool = True,
        animation_chars: str = "◐◓◑◒"
    ):
        """
        Inisialisasi indikator loading.
        
        Args:
            message: Pesan yang ditampilkan
            is_indeterminate: Jika True, menampilkan animasi loading tanpa progress
            max_value: Nilai maksimum untuk progress bar
            auto_hide: Otomatis sembunyikan saat selesai
            animation_chars: Karakter untuk animasi loading
        """
        self.message = message
        self.is_indeterminate = is_indeterminate
        self.max_value = max_value
        self.auto_hide = auto_hide
        self.animation_chars = animation_chars
        self.current_value = 0
        self.is_running = False
        self.animation_thread = None
        self.animation_index = 0
        
        # Buat container utama
        self.container = widgets.VBox()
        
        # Buat komponen UI
        self._create_ui_components()
        
        # Mulai animasi jika indeterminate
        if self.is_indeterminate:
            self.start_animation()
    
    def _create_ui_components(self):
        """Buat komponen UI untuk indikator loading."""
        # Buat header dengan pesan
        self.message_widget = widgets.HTML(
            value=f"<div style='display: flex; align-items: center; padding: 5px; background-color: #f8f9fa; border-radius: 4px;'>"
                  f"<span style='margin-right: 10px; font-size: 14px; color: #495057;'>{self.message}</span>"
                  f"<span id='loading-anim' style='font-size: 16px; color: #0d6efd;'>{self.animation_chars[0]}</span>"
                  f"</div>"
        )
        
        # Buat progress bar
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=self.max_value,
            description='',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(width='100%', height='4px', margin='2px 0 0 0', visibility='visible' if not self.is_indeterminate else 'hidden')
        )
        
        # Tambahkan komponen ke container
        self.container.children = [self.message_widget, self.progress_bar]
    
    def start_animation(self):
        """Mulai animasi loading."""
        if self.is_running:
            return
            
        self.is_running = True
        self.animation_thread = threading.Thread(target=self._animate, daemon=True)
        self.animation_thread.start()
    
    def stop_animation(self):
        """Hentikan animasi loading."""
        self.is_running = False
        if self.animation_thread:
            self.animation_thread.join(timeout=0.5)
    
    def _animate(self):
        """Jalankan animasi loading."""
        try:
            while self.is_running:
                # Update animasi
                self.animation_index = (self.animation_index + 1) % len(self.animation_chars)
                anim_char = self.animation_chars[self.animation_index]
                
                # Update pesan dengan karakter animasi
                self.message_widget.value = (
                    f"<div style='display: flex; align-items: center; padding: 5px; background-color: #f8f9fa; border-radius: 4px;'>"
                    f"<span style='margin-right: 10px; font-size: 14px; color: #495057;'>{self.message}</span>"
                    f"<span id='loading-anim' style='font-size: 16px; color: #0d6efd;'>{anim_char}</span>"
                    f"</div>"
                )
                
                # Tunggu sebentar
                time.sleep(0.2)
        except Exception as e:
            logger.error(f"Error pada animasi loading: {str(e)}")
    
    def update(self, value: int, message: Optional[str] = None):
        """
        Update progress bar dan pesan.
        
        Args:
            value: Nilai progress (0-100)
            message: Pesan baru (opsional)
        """
        # Update nilai progress
        self.current_value = min(value, self.max_value)
        self.progress_bar.value = self.current_value
        
        # Update pesan jika disediakan
        if message:
            self.message = message
            if not self.is_indeterminate:
                anim_char = self.animation_chars[self.animation_index]
                self.message_widget.value = (
                    f"<div style='display: flex; align-items: center; padding: 5px; background-color: #f8f9fa; border-radius: 4px;'>"
                    f"<span style='margin-right: 10px; font-size: 14px; color: #495057;'>{self.message}</span>"
                    f"<span id='loading-anim' style='font-size: 16px; color: #0d6efd;'>{anim_char}</span>"
                    f"</div>"
                )
        
        # Sembunyikan jika sudah selesai dan auto_hide diaktifkan
        if self.current_value >= self.max_value and self.auto_hide:
            self.hide()
    
    def show(self):
        """Tampilkan indikator loading."""
        self.container.layout.display = 'flex'
        if self.is_indeterminate:
            self.start_animation()
    
    def hide(self):
        """Sembunyikan indikator loading."""
        self.container.layout.display = 'none'
        if self.is_indeterminate:
            self.stop_animation()
    
    def complete(self, success_message: Optional[str] = None):
        """
        Tandai loading sebagai selesai.
        
        Args:
            success_message: Pesan sukses yang ditampilkan
        """
        # Hentikan animasi
        self.stop_animation()
        
        # Update progress ke 100%
        self.progress_bar.value = self.max_value
        
        # Update pesan sukses
        if success_message:
            self.message_widget.value = (
                f"<div style='display: flex; align-items: center; padding: 5px; background-color: #d4edda; border-radius: 4px;'>"
                f"<span style='margin-right: 10px; font-size: 14px; color: #155724;'>{success_message}</span>"
                f"<span style='font-size: 16px; color: #155724;'>{ICONS.get('success', '✅')}</span>"
                f"</div>"
            )
        
        # Sembunyikan jika auto_hide diaktifkan
        if self.auto_hide:
            threading.Timer(1.0, self.hide).start()
    
    def error(self, error_message: str):
        """
        Tampilkan pesan error.
        
        Args:
            error_message: Pesan error yang ditampilkan
        """
        # Hentikan animasi
        self.stop_animation()
        
        # Update pesan error
        self.message_widget.value = (
            f"<div style='display: flex; align-items: center; padding: 5px; background-color: #f8d7da; border-radius: 4px;'>"
            f"<span style='margin-right: 10px; font-size: 14px; color: #721c24;'>{error_message}</span>"
            f"<span style='font-size: 16px; color: #721c24;'>{ICONS.get('error', '❌')}</span>"
            f"</div>"
        )
        
        # Ubah warna progress bar
        self.progress_bar.bar_style = 'danger'


def create_loading_indicator(
    message: str = "Memuat...", 
    is_indeterminate: bool = False,
    max_value: int = 100,
    auto_hide: bool = True
) -> LoadingIndicator:
    """
    Buat dan kembalikan indikator loading.
    
    Args:
        message: Pesan yang ditampilkan
        is_indeterminate: Jika True, menampilkan animasi loading tanpa progress
        max_value: Nilai maksimum untuk progress bar
        auto_hide: Otomatis sembunyikan saat selesai
        
    Returns:
        Instance LoadingIndicator
    """
    return LoadingIndicator(
        message=message,
        is_indeterminate=is_indeterminate,
        max_value=max_value,
        auto_hide=auto_hide
    )
