# File: smartcash/interface/display/log_display.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk menangani tampilan log, error, dan pesan sukses di interface

import curses
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import deque

from smartcash.interface.display.base_display import BaseDisplay

class LogDisplay(BaseDisplay):
    """
    Komponen untuk menampilkan dan mengelola log aplikasi.
    Menangani pesan error, success, warning, dan info dengan penampilan yang sesuai.
    """
    
    # Ukuran maksimal buffer log
    DEFAULT_BUFFER_SIZE = 100
    
    # Tipe-tipe log yang didukung
    LOG_TYPE_ERROR = 'error'
    LOG_TYPE_SUCCESS = 'success'
    LOG_TYPE_WARNING = 'warning'
    LOG_TYPE_INFO = 'info'
    LOG_TYPE_METRIC = 'metric'
    LOG_TYPE_DEBUG = 'debug'
    
    # Mapping tipe log ke emoji dan warna
    LOG_FORMATS = {
        LOG_TYPE_ERROR: {'emoji': '❌', 'color': BaseDisplay.COLOR_ERROR},
        LOG_TYPE_SUCCESS: {'emoji': '✅', 'color': BaseDisplay.COLOR_SUCCESS},
        LOG_TYPE_WARNING: {'emoji': '⚠️', 'color': BaseDisplay.COLOR_WARNING},
        LOG_TYPE_INFO: {'emoji': 'ℹ️', 'color': BaseDisplay.COLOR_INFO},
        LOG_TYPE_METRIC: {'emoji': '📊', 'color': BaseDisplay.COLOR_NAVIGATION},
        LOG_TYPE_DEBUG: {'emoji': '🔍', 'color': BaseDisplay.COLOR_SPECIAL}
    }
    
    def __init__(
        self, 
        stdscr: curses.window, 
        parent=None,
        max_visible_lines: int = 5,
        buffer_size: int = DEFAULT_BUFFER_SIZE
    ):
        """
        Inisialisasi komponen log display.
        
        Args:
            stdscr: Curses window
            parent: Parent display (opsional)
            max_visible_lines: Jumlah maksimal baris log yang ditampilkan
            buffer_size: Ukuran maksimal buffer log
        """
        super().__init__(stdscr, parent)
        
        # Setup log buffer
        self.buffer_size = buffer_size
        self.log_buffer = deque(maxlen=buffer_size)
        
        # Jumlah baris yang ditampilkan
        self.max_visible_lines = max_visible_lines
        
        # Tracking untuk error log history
        self.error_history = deque(maxlen=buffer_size)
        
        # Posisi scroll untuk melihat log yang lebih lama
        self.scroll_position = 0
        
        # Waktu timeout untuk pesan yang harus menghilang
        self.auto_clear_messages = []  # [(timestamp, index),...]
    
    def add_log(
        self, 
        message: str, 
        log_type: str = LOG_TYPE_INFO,
        auto_clear_seconds: float = 0.0,
        highlight_values: bool = False
    ) -> None:
        """
        Tambahkan pesan ke buffer log.
        
        Args:
            message: Pesan yang akan ditampilkan
            log_type: Tipe log (error, success, warning, info, metric, debug)
            auto_clear_seconds: Waktu dalam detik sebelum pesan dihapus (0 = tidak auto-clear)
            highlight_values: Jika True, highlight angka dalam pesan
        """
        # Validasi tipe log
        if log_type not in self.LOG_FORMATS:
            log_type = self.LOG_TYPE_INFO
        
        # Format pesan dengan emoji
        emoji = self.LOG_FORMATS[log_type]['emoji']
        color = self.LOG_FORMATS[log_type]['color']
        
        # Timestamp untuk pesan
        timestamp = time.time()
        
        # Highlight nilai numerik jika diminta
        if highlight_values:
            message = self._highlight_numeric_values(message)
        
        # Simpan pesan ke buffer
        log_entry = {
            'message': message,
            'type': log_type,
            'emoji': emoji,
            'color': color,
            'timestamp': timestamp,
            'formatted': f"{emoji} {message}"
        }
        
        self.log_buffer.appendleft(log_entry)
        
        # Reset posisi scroll ke paling atas
        self.scroll_position = 0
        
        # Tambahkan ke history error jika tipe error
        if log_type == self.LOG_TYPE_ERROR:
            self.error_history.appendleft(log_entry)
        
        # Setup auto-clear jika diperlukan
        if auto_clear_seconds > 0:
            self.auto_clear_messages.append((timestamp + auto_clear_seconds, 0))
    
    def _highlight_numeric_values(self, message: str) -> str:
        """
        Highlight nilai numerik dalam pesan.
        
        Args:
            message: Pesan original
            
        Returns:
            Pesan dengan nilai numerik yang di-highlight
        """
        # Implementasi sederhana hanya untuk terminal standar
        # Dalam curses yang sebenarnya, highlight akan ditangani saat menampilkan
        return message
    
    def error(self, message: str, auto_clear_seconds: float = 0.0) -> None:
        """
        Tambahkan pesan error ke log.
        
        Args:
            message: Pesan error
            auto_clear_seconds: Waktu auto-clear dalam detik
        """
        self.add_log(message, self.LOG_TYPE_ERROR, auto_clear_seconds)
    
    def success(self, message: str, auto_clear_seconds: float = 0.0) -> None:
        """
        Tambahkan pesan sukses ke log.
        
        Args:
            message: Pesan sukses
            auto_clear_seconds: Waktu auto-clear dalam detik
        """
        self.add_log(message, self.LOG_TYPE_SUCCESS, auto_clear_seconds)
    
    def warning(self, message: str, auto_clear_seconds: float = 0.0) -> None:
        """
        Tambahkan pesan warning ke log.
        
        Args:
            message: Pesan warning
            auto_clear_seconds: Waktu auto-clear dalam detik
        """
        self.add_log(message, self.LOG_TYPE_WARNING, auto_clear_seconds)
    
    def info(self, message: str, auto_clear_seconds: float = 0.0) -> None:
        """
        Tambahkan pesan informasi ke log.
        
        Args:
            message: Pesan informasi
            auto_clear_seconds: Waktu auto-clear dalam detik
        """
        self.add_log(message, self.LOG_TYPE_INFO, auto_clear_seconds)
    
    def metric(self, message: str, auto_clear_seconds: float = 0.0) -> None:
        """
        Tambahkan pesan metrik ke log.
        
        Args:
            message: Pesan metrik
            auto_clear_seconds: Waktu auto-clear dalam detik
        """
        self.add_log(message, self.LOG_TYPE_METRIC, auto_clear_seconds, highlight_values=True)
    
    def debug(self, message: str, auto_clear_seconds: float = 0.0) -> None:
        """
        Tambahkan pesan debug ke log.
        
        Args:
            message: Pesan debug
            auto_clear_seconds: Waktu auto-clear dalam detik
        """
        self.add_log(message, self.LOG_TYPE_DEBUG, auto_clear_seconds)
    
    def clear_all(self) -> None:
        """Bersihkan seluruh buffer log."""
        self.log_buffer.clear()
        self.auto_clear_messages.clear()
        self.scroll_position = 0
    
    def get_error_history(self) -> List[Dict]:
        """
        Dapatkan riwayat error.
        
        Returns:
            List error yang tersimpan di history
        """
        return list(self.error_history)
    
    def _process_auto_clear(self) -> None:
        """Proses auto-clear messages berdasarkan timestamp."""
        if not self.auto_clear_messages:
            return
            
        current_time = time.time()
        messages_to_remove = []
        
        # Identifikasi pesan yang harus dihapus
        for clear_time, index in self.auto_clear_messages:
            if current_time >= clear_time:
                messages_to_remove.append((clear_time, index))
        
        # Hapus dari auto_clear_messages
        for item in messages_to_remove:
            self.auto_clear_messages.remove(item)
                
        # Tidak perlu menghapus dari buffer karena indeks relatif berubah
        # Hanya perlu me-refresh tampilan
    
    def draw(self) -> None:
        """Gambar komponen log display."""
        # Proses auto-clear messages
        self._process_auto_clear()
        
        # Bersihkan area
        self.clear_area()
        
        # Gambar border dengan judul
        self.draw_border("Log Area")
        
        # Area content dimulai setelah border
        content_y = self.y + 1
        content_height = self.display_height - 2
        content_width = self.display_width - 2
        
        # Hitung berapa banyak pesan yang bisa ditampilkan
        visible_lines = min(self.max_visible_lines, content_height, len(self.log_buffer))
        
        # Tampilkan pesan dari buffer
        log_entries = list(self.log_buffer)[self.scroll_position:self.scroll_position + visible_lines]
        
        for i, entry in enumerate(log_entries):
            # Pastikan tidak melebihi area display
            if i >= content_height:
                break
                
            # Potong pesan jika terlalu panjang
            formatted_msg = entry['formatted']
            if len(formatted_msg) > content_width - 2:
                formatted_msg = formatted_msg[:content_width-5] + "..."
                
            # Tampilkan pesan dengan warna yang sesuai
            self.safe_addstr(
                content_y + i,
                self.x + 1,
                formatted_msg,
                entry['color']
            )
        
        # Tampilkan scrollbar jika buffer lebih besar dari area yang ditampilkan
        if len(self.log_buffer) > visible_lines:
            self._draw_scrollbar(content_y, content_height, visible_lines)
    
    def _draw_scrollbar(
        self, 
        content_y: int, 
        content_height: int, 
        visible_lines: int
    ) -> None:
        """
        Gambar scrollbar untuk navigasi log.
        
        Args:
            content_y: Posisi y area konten
            content_height: Tinggi area konten
            visible_lines: Jumlah baris yang terlihat
        """
        try:
            # Hitung ukuran dan posisi scrollbar
            total_entries = len(self.log_buffer)
            scrollbar_height = max(1, int(content_height * visible_lines / total_entries))
            scrollbar_pos = min(
                content_height - scrollbar_height,
                int(self.scroll_position * content_height / total_entries)
            )
            
            # Posisi x scrollbar
            scrollbar_x = self.x + self.display_width - 1
            
            # Gambar scrollbar track
            for i in range(content_height):
                if scrollbar_pos <= i < scrollbar_pos + scrollbar_height:
                    # Bagian aktif scrollbar
                    self.safe_addstr(content_y + i, scrollbar_x, "█")
                else:
                    # Track scrollbar
                    self.safe_addstr(content_y + i, scrollbar_x, "│")
        
        except curses.error:
            # Abaikan error jika layar terlalu kecil
            pass
    
    def handle_input(self, key: int) -> Optional[bool]:
        """
        Tangani input keyboard untuk navigasi log.
        
        Args:
            key: Kode tombol
            
        Returns:
            True jika input ditangani, None jika tidak
        """
        # Navigasi scroll
        if key == curses.KEY_UP and self.scroll_position > 0:
            # Scroll ke atas
            self.scroll_position -= 1
            return True
        elif key == curses.KEY_DOWN and self.scroll_position < len(self.log_buffer) - self.max_visible_lines:
            # Scroll ke bawah
            self.scroll_position += 1
            return True
        elif key == curses.KEY_PPAGE:  # Page Up
            # Scroll satu halaman ke atas
            self.scroll_position = max(0, self.scroll_position - self.max_visible_lines)
            return True
        elif key == curses.KEY_NPAGE:  # Page Down
            # Scroll satu halaman ke bawah
            max_scroll = max(0, len(self.log_buffer) - self.max_visible_lines)
            self.scroll_position = min(max_scroll, self.scroll_position + self.max_visible_lines)
            return True
        elif key == curses.KEY_HOME:
            # Scroll ke awal
            self.scroll_position = 0
            return True
        elif key == curses.KEY_END:
            # Scroll ke akhir
            self.scroll_position = max(0, len(self.log_buffer) - self.max_visible_lines)
            return True
        elif key == ord('c'):
            # Clear log
            self.clear_all()
            return True
            
        return None
    
    def show_error_history_dialog(self, dialog_display) -> None:
        """
        Tampilkan riwayat error dalam dialog.
        
        Args:
            dialog_display: Komponen dialog untuk menampilkan history
        """
        if not dialog_display:
            return
            
        # Format error history
        errors = self.get_error_history()
        if not errors:
            dialog_display.show_info("Tidak ada error yang tercatat", "Riwayat Error")
            return
            
        # Format error message untuk dialog
        error_lines = []
        for i, err in enumerate(errors):
            error_lines.append(f"[{i+1}] {err['message']}")
            
        # Tampilkan dalam dialog
        dialog_display.show_terminal("Riwayat Error", error_lines, wait_for_key=True)