# File: smartcash/interface/display/display_manager.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi lanjutan dari DisplayManager dengan struktur komposisi yang modular

import curses
import torch
import psutil
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from pathlib import Path
from datetime import datetime

from smartcash.interface.display.base_display import BaseDisplay
from smartcash.interface.display.menu_display import MenuDisplay
from smartcash.interface.display.log_display import LogDisplay
from smartcash.interface.display.status_display import StatusDisplay
from smartcash.interface.display.dialog_display import DialogDisplay

class DisplayManager:
    """
    Manager utama untuk mengelola dan mengkoordinasikan komponen display.
    Menerapkan pattern komposisi dengan berbagai komponen display.
    """
    
    def __init__(self, stdscr: curses.window):
        """
        Inisialisasi display manager.
        
        Args:
            stdscr: Curses window utama
        """
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        
        # Inisialisasi komponen display
        self._init_displays()
        
        # Setup color pairs
        self._setup_colors()
        
        # Error history untuk referensi
        self.error_log = []
        self.max_error_logs = 20
        
        # Log buffer untuk pesan status
        self.log_buffer = []
        self.max_log_lines = 5
    
    def _init_displays(self) -> None:
        """
        Inisialisasi dan atur layout komponen display.
        Menggunakan pendekatan modular untuk memisahkan tanggung jawab.
        """
        # Calculate layout
        menu_width = int(self.width * 0.6)
        status_width = self.width - menu_width - 1  # -1 untuk border
        
        log_height = self.max_log_lines + 2  # +2 untuk border
        main_height = self.height - log_height - 1  # -1 untuk status bar
        
        # Inisialisasi komponen dasar
        self.menu_display = MenuDisplay(
            self.stdscr, 
            parent=self,
            description_callback=self._show_item_description
        )
        self.menu_display.set_area(0, 2, menu_width, main_height)
        
        self.status_display = StatusDisplay(self.stdscr, parent=self)
        self.status_display.set_area(menu_width + 1, 2, status_width, main_height)
        
        self.log_display = LogDisplay(
            self.stdscr, 
            parent=self,
            max_visible_lines=self.max_log_lines
        )
        self.log_display.set_area(0, main_height + 2, self.width, log_height)
        
        # Dialog display sebagai overlay
        self.dialog_display = DialogDisplay(self.stdscr, parent=self)
    
    def _setup_colors(self) -> None:
        """Setup pasangan warna untuk interface."""
        if not curses.has_colors():
            return
            
        curses.start_color()
        curses.use_default_colors()
        
        # Initialize color pairs
        curses.init_pair(1, curses.COLOR_RED, -1)     # Error/Not Set
        curses.init_pair(2, curses.COLOR_GREEN, -1)   # Success/Set
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Warning/Highlight
        curses.init_pair(4, curses.COLOR_CYAN, -1)    # Info/Title
        curses.init_pair(5, curses.COLOR_MAGENTA, -1) # Special
        curses.init_pair(6, curses.COLOR_BLUE, -1)    # Navigation
    
    def _show_item_description(self, item) -> None:
        """
        Callback untuk menampilkan deskripsi item menu.
        
        Args:
            item: Item menu yang dipilih
        """
        if hasattr(item, 'description') and item.description:
            # Tampilkan deskripsi di status display
            pass
    
    def resize(self) -> None:
        """Tangani perubahan ukuran terminal."""
        self.height, self.width = self.stdscr.getmaxyx()
        
        # Re-layout semua komponen
        self._init_displays()
        
    def draw(self, config: Dict[str, Any] = None) -> None:
        """
        Gambar semua komponen display.
        
        Args:
            config: Konfigurasi aplikasi saat ini
        """
        # Bersihkan layar
        self.stdscr.clear()
        
        # Gambar komponen
        if hasattr(self, 'menu_display'):
            self.menu_display.draw()
            
        if hasattr(self, 'status_display') and config:
            self.status_display.draw(config)
            
        if hasattr(self, 'log_display'):
            self.log_display.draw()
            
        # Refresh layar
        self.stdscr.refresh()
    
    def show_error(
        self, 
        message: str, 
        timeout_ms: int = 0,
        show_help: bool = True
    ) -> Optional[int]:
        """
        Tampilkan pesan error dengan warna merah dan logging error.
        
        Args:
            message: Pesan error untuk ditampilkan
            timeout_ms: Timeout dalam milidetik (0 untuk tunggu key press)
            show_help: Tampilkan saran bantuan
            
        Returns:
            Key yang ditekan atau None jika timeout
        """
        # Simpan error ke history
        self._add_to_error_history(message)
        
        # Tambahkan ke log display
        if hasattr(self, 'log_display'):
            self.log_display.error(message)
            
            # Draw ulang untuk menampilkan error
            self.log_display.draw()
            self.stdscr.refresh()
        
        # Tampilkan dialog error jika diinginkan
        if timeout_ms > 0 or show_help:
            # Set timeout jika diperlukan
            if timeout_ms > 0:
                self.stdscr.timeout(timeout_ms)
                
            result = self.dialog_display.show_dialog(
                "Error", 
                message, 
                self.dialog_display.TYPE_ERROR
            )
            
            # Reset timeout
            if timeout_ms > 0:
                self.stdscr.timeout(-1)
                
            return result
            
        return None
    
    def _add_to_error_history(self, message: str) -> None:
        """
        Tambahkan error ke history dengan timestamp.
        
        Args:
            message: Pesan error
        """
        # Tambahkan timestamp ke error
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.error_log.append({
            'timestamp': timestamp,
            'message': message
        })
        
        # Batasi jumlah error dalam history
        while len(self.error_log) > self.max_error_logs:
            self.error_log.pop(0)
    
    def show_success(
        self, 
        message: str, 
        timeout_ms: int = 0
    ) -> None:
        """
        Tampilkan pesan sukses.
        
        Args:
            message: Pesan sukses
            timeout_ms: Timeout dalam milidetik (0 untuk tanpa timeout)
        """
        # Tambahkan ke log display
        if hasattr(self, 'log_display'):
            self.log_display.success(message)
            
            # Draw ulang untuk menampilkan pesan
            self.log_display.draw()
            self.stdscr.refresh()
        
        # Tampilkan dialog jika timeout > 0
        if timeout_ms > 0:
            self.dialog_display.show_dialog(
                "Sukses", 
                message, 
                self.dialog_display.TYPE_INFO,
                timeout_ms=timeout_ms
            )
    
    def show_warning(
        self, 
        message: str, 
        timeout_ms: int = 0
    ) -> None:
        """
        Tampilkan pesan warning.
        
        Args:
            message: Pesan warning
            timeout_ms: Timeout dalam milidetik
        """
        # Tambahkan ke log display
        if hasattr(self, 'log_display'):
            self.log_display.warning(message)
            
            # Draw ulang
            self.log_display.draw()
            self.stdscr.refresh()
        
        # Tampilkan dialog jika timeout > 0
        if timeout_ms > 0:
            self.dialog_display.show_dialog(
                "Peringatan", 
                message, 
                self.dialog_display.TYPE_WARNING,
                timeout_ms=timeout_ms
            )
    
    def show_config_status(self, config: Dict[str, Any]) -> None:
        """
        Tampilkan status konfigurasi saat ini.
        
        Args:
            config: Konfigurasi aplikasi saat ini
        """
        if hasattr(self, 'status_display') and config:
            # Draw ulang status display dengan konfigurasi baru
            self.status_display.draw(config)
            self.stdscr.refresh()
    
    def show_system_status(self) -> None:
        """
        Tampilkan status sistem di header.
        Info ini juga ditampilkan oleh StatusDisplay.
        """
        # Refresh status display
        if hasattr(self, 'status_display'):
            self.status_display.draw(None)
            self.stdscr.refresh()
    
    def get_user_input(
        self, 
        prompt: str, 
        validator: Optional[Callable] = None,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Dapatkan input dari user dengan validasi.
        
        Args:
            prompt: Prompt untuk ditampilkan
            validator: Fungsi validasi optional
            default: Nilai default jika input kosong
            
        Returns:
            String input atau None jika dibatalkan
        """
        if hasattr(self, 'dialog_display'):
            return self.dialog_display.get_input(
                prompt=prompt,
                validator=validator,
                default=default
            )
        return None
    
    def show_dialog(
        self, 
        title: str, 
        message: str, 
        options: Dict[str, str] = None
    ) -> Optional[str]:
        """
        Tampilkan dialog konfirmasi.
        
        Args:
            title: Judul dialog
            message: Pesan dialog
            options: Dict mapping key ke label opsi
            
        Returns:
            Key dari opsi yang dipilih atau None jika dibatalkan
        """
        if hasattr(self, 'dialog_display'):
            return self.dialog_display.show_dialog(
                title=title,
                message=message,
                dialog_type=self.dialog_display.TYPE_CONFIRM,
                options=options
            )
        return None
    
    def show_progress(
        self, 
        message: str, 
        current: int, 
        total: int
    ) -> None:
        """
        Tampilkan progress bar.
        
        Args:
            message: Pesan progress
            current: Nilai progress saat ini
            total: Nilai total progress
        """
        if hasattr(self, 'dialog_display'):
            # Hitung progress percentage
            progress = min(1.0, max(0.0, current / total))
            
            # Tampilkan progress dialog
            self.dialog_display.show_progress(
                title="Progress",
                message=message,
                progress=progress,
                can_cancel=False
            )
        else:
            # Fallback ke log display
            if hasattr(self, 'log_display'):
                progress_msg = f"{message} [{current}/{total}]"
                self.log_display.info(progress_msg)
                self.log_display.draw()
                self.stdscr.refresh()
    
    def show_help(
        self, 
        title: str, 
        content: Dict[str, str]
    ) -> None:
        """
        Tampilkan layar bantuan.
        
        Args:
            title: Judul layar bantuan
            content: Dict berisi kategori dan konten bantuan
        """
        # Format content for terminal display
        lines = []
        for category, text in content.items():
            lines.append(f"=== {category} ===")
            lines.extend(text.split('\n'))
            lines.append("")  # Add blank line between categories
        
        if hasattr(self, 'dialog_display'):
            self.dialog_display.show_terminal(
                title,
                lines,
                wait_for_key=True
            )
    
    def show_form(
        self, 
        title: str, 
        fields: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tampilkan form input dengan banyak field.
        
        Args:
            title: Judul form
            fields: List field dengan format:
                [
                    {
                        'name': 'field_name',
                        'label': 'Label untuk ditampilkan',
                        'type': 'text/number/select',
                        'default': 'Nilai default',
                        'validator': callable(value) -> bool,
                        'options': [Untuk type 'select']
                    }
                ]
                
        Returns:
            Dict hasil input form atau dict kosong jika dibatalkan
        """
        if hasattr(self, 'dialog_display'):
            return self.dialog_display.show_form(title, fields)
        return {}
    
    def show_error_history(self) -> None:
        """Tampilkan riwayat error sebagai referensi."""
        if not self.error_log:
            self.show_dialog("Riwayat Error", "Tidak ada error yang tercatat")
            return
        
        # Format error history untuk ditampilkan
        lines = []
        for i, err in enumerate(self.error_log):
            lines.append(f"[{i+1}] {err['timestamp']} - {err['message']}")
        
        if hasattr(self, 'dialog_display'):
            self.dialog_display.show_terminal(
                "Riwayat Error",
                lines,
                wait_for_key=True
            )
    
    def show_info(
        self, 
        message: str, 
        title: str = "Informasi"
    ) -> None:
        """
        Tampilkan pesan informasi.
        
        Args:
            message: Pesan informasi
            title: Judul dialog
        """
        if hasattr(self, 'dialog_display'):
            self.dialog_display.show_dialog(
                title=title,
                message=message,
                dialog_type=self.dialog_display.TYPE_INFO
            )
        else:
            # Fallback ke log display
            if hasattr(self, 'log_display'):
                self.log_display.info(message)
                self.log_display.draw()
                self.stdscr.refresh()
    
    def clear_message_area(self) -> None:
        """Bersihkan area pesan di bagian bawah layar."""
        if hasattr(self, 'log_display'):
            self.log_display.clear_all()
            self.log_display.draw()
            self.stdscr.refresh()