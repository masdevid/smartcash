# File: smartcash/interface/display/terminal_display.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk menampilkan jendela terminal dengan dukungan scrolling

import curses
from typing import List, Optional, Tuple, Dict, Any
import re

from smartcash.interface.display.base_display import BaseDisplay

class TerminalDisplay(BaseDisplay):
    """
    Komponen untuk menampilkan jendela terminal dengan scrolling.
    Berguna untuk menampilkan log, output proses, dan informasi detail.
    """
    
    def __init__(self, stdscr: curses.window, parent=None):
        """
        Inisialisasi terminal display.
        
        Args:
            stdscr: Curses window
            parent: Parent display (opsional)
        """
        super().__init__(stdscr, parent)
        
        # Atribut untuk konten dan scrolling
        self.lines = []              # Baris-baris yang ditampilkan
        self.scroll_pos = 0          # Posisi scroll saat ini
        self.max_lines = 1000        # Batas maksimum jumlah baris
        self.footer_text = "↑↓: Scroll | Q: Kembali"  # Default footer
        
        # Pola untuk deteksi emoji dan prefiks
        self.emoji_pattern = re.compile(r'([\U00010000-\U0010ffff]|[\u2600-\u2B55])')
        self.prefix_colors = {
            '✅': self.COLOR_SUCCESS,   # Sukses - hijau
            '❌': self.COLOR_ERROR,     # Error - merah
            '⚠️': self.COLOR_WARNING,   # Peringatan - kuning
            'ℹ️': self.COLOR_INFO,      # Info - cyan
            '🔍': self.COLOR_INFO,      # Info detail - cyan
            '🚀': self.COLOR_SPECIAL,   # Khusus - magenta
            '📊': self.COLOR_NAVIGATION  # Data - biru
        }
    
    def set_content(self, lines: List[str]) -> None:
        """
        Set konten terminal.
        
        Args:
            lines: List baris teks yang akan ditampilkan
        """
        # Batasi jumlah baris
        if len(lines) > self.max_lines:
            lines = lines[-self.max_lines:]
            
        self.lines = lines
        self.scroll_pos = 0  # Reset posisi scroll
        
    def append_line(self, line: str) -> None:
        """
        Tambahkan satu baris ke konten terminal.
        
        Args:
            line: Baris teks untuk ditambahkan
        """
        self.lines.append(line)
        
        # Batasi jumlah baris
        if len(self.lines) > self.max_lines:
            self.lines.pop(0)
            
        # Auto-scroll jika sudah di bawah
        if self.scroll_pos >= len(self.lines) - self.display_height + 3:
            self.scroll_to_bottom()
            
    def append_lines(self, lines: List[str]) -> None:
        """
        Tambahkan beberapa baris ke konten terminal.
        
        Args:
            lines: List baris teks untuk ditambahkan
        """
        for line in lines:
            self.append_line(line)
            
    def clear(self) -> None:
        """Bersihkan konten terminal."""
        self.lines = []
        self.scroll_pos = 0
        
    def set_footer(self, footer_text: str) -> None:
        """
        Set teks footer.
        
        Args:
            footer_text: Teks footer baru
        """
        self.footer_text = footer_text
        
    def scroll_to_top(self) -> None:
        """Scroll ke awal konten."""
        self.scroll_pos = 0
        
    def scroll_to_bottom(self) -> None:
        """Scroll ke akhir konten."""
        max_scroll = max(0, len(self.lines) - (self.display_height - 4))
        self.scroll_pos = max_scroll
        
    def scroll_up(self, lines: int = 1) -> None:
        """
        Scroll ke atas.
        
        Args:
            lines: Jumlah baris untuk di-scroll
        """
        self.scroll_pos = max(0, self.scroll_pos - lines)
        
    def scroll_down(self, lines: int = 1) -> None:
        """
        Scroll ke bawah.
        
        Args:
            lines: Jumlah baris untuk di-scroll
        """
        max_scroll = max(0, len(self.lines) - (self.display_height - 4))
        self.scroll_pos = min(max_scroll, self.scroll_pos + lines)
        
    def page_up(self) -> None:
        """Scroll satu halaman ke atas."""
        self.scroll_up(self.display_height - 4)
        
    def page_down(self) -> None:
        """Scroll satu halaman ke bawah."""
        self.scroll_down(self.display_height - 4)
    
    def _get_line_color(self, line: str) -> int:
        """
        Deteksi warna yang sesuai untuk baris teks.
        
        Args:
            line: Baris teks
            
        Returns:
            ID pasangan warna
        """
        # Cek emoji di awal baris
        emoji_match = self.emoji_pattern.search(line)
        if emoji_match and emoji_match.start() == 0:
            emoji = emoji_match.group()
            return self.prefix_colors.get(emoji, 0)
            
        # Cek prefiks umum
        if line.startswith("ERROR") or line.startswith("FATAL"):
            return self.COLOR_ERROR
        elif line.startswith("WARNING"):
            return self.COLOR_WARNING
        elif line.startswith("INFO") or line.startswith("DEBUG"):
            return self.COLOR_INFO
        elif line.startswith("SUCCESS"):
            return self.COLOR_SUCCESS
            
        # Deteksi angka untuk highlight
        if re.search(r'\d+\.\d+|\d+%', line):
            return self.COLOR_SPECIAL
            
        return 0  # Default: tidak ada warna khusus

    def draw(self, title: Optional[str] = "Terminal") -> None:
        """
        Gambar jendela terminal dengan scrolling.
        
        Args:
            title: Judul jendela terminal (opsional)
        """
        # Bersihkan area
        self.clear_area()
        
        # Gambar border
        self.draw_border(title)
        
        # Hitung area konten yang tersedia
        content_height = self.display_height - 4  # 2 untuk border, 2 untuk header/footer
        content_width = self.display_width - 4    # 2 padding di setiap sisi
        content_y = self.y + 2
        content_x = self.x + 2
        
        # Gambar konten dengan scrolling
        visible_lines = self.lines[self.scroll_pos:self.scroll_pos + content_height]
        
        for i, line in enumerate(visible_lines):
            if i >= content_height:
                break
                
            # Potong line jika terlalu panjang
            if len(line) > content_width:
                line = line[:content_width-3] + "..."
                
            # Deteksi warna yang sesuai
            color = self._get_line_color(line)
            
            # Tampilkan line dengan warna yang sesuai
            self.safe_addstr(content_y + i, content_x, line, color)
            
        # Gambar scrollbar jika perlu
        if len(self.lines) > content_height:
            self._draw_scrollbar(content_y, content_height)
            
        # Gambar footer
        footer_y = self.y + self.display_height - 2
        footer_x = self.x + (self.display_width - len(self.footer_text)) // 2
        
        self.safe_addstr(
            footer_y, 
            footer_x, 
            self.footer_text, 
            self.COLOR_NAVIGATION
        )
        
    def _draw_scrollbar(self, content_y: int, content_height: int) -> None:
        """
        Gambar scrollbar pada posisi yang sesuai.
        
        Args:
            content_y: Posisi y dari area konten
            content_height: Tinggi area konten
        """
        scrollbar_x = self.x + self.display_width - 2
        
        # Hitung tinggi dan posisi scrollbar
        total_lines = len(self.lines)
        scrollbar_height = max(1, int(content_height * content_height / total_lines))
        scrollbar_pos = min(
            content_height - scrollbar_height,
            int(self.scroll_pos * content_height / total_lines)
        )
        
        # Gambar scrollbar track
        for i in range(content_height):
            self.safe_addstr(content_y + i, scrollbar_x, "│")
            
        # Gambar scrollbar thumb
        for i in range(scrollbar_height):
            if scrollbar_pos + i < content_height:
                self.safe_addstr(
                    content_y + scrollbar_pos + i, 
                    scrollbar_x, 
                    "█", 
                    self.COLOR_SPECIAL
                )
        
    def handle_input(self, key: int) -> Optional[bool]:
        """
        Tangani input keyboard untuk navigasi terminal.
        
        Args:
            key: Kode tombol keyboard
            
        Returns:
            True jika berhasil menangani input, False untuk kembali, None jika tidak ditangani
        """
        if key == curses.KEY_UP:
            self.scroll_up()
            return True
        elif key == curses.KEY_DOWN:
            self.scroll_down()
            return True
        elif key == curses.KEY_PPAGE:  # Page Up
            self.page_up()
            return True
        elif key == curses.KEY_NPAGE:  # Page Down
            self.page_down()
            return True
        elif key == curses.KEY_HOME:
            self.scroll_to_top()
            return True
        elif key == curses.KEY_END:
            self.scroll_to_bottom()
            return True
        elif key in [ord('q'), ord('Q')]:
            return False  # Kembali
        return None  # Tidak menangani input
        
    def show_terminal(
        self, 
        title: str, 
        lines: List[str], 
        wait_for_key: bool = True
    ) -> None:
        """
        Tampilkan terminal dengan konten dan tunggu sampai pengguna selesai.
        
        Args:
            title: Judul terminal
            lines: Konten untuk ditampilkan
            wait_for_key: True untuk menunggu tombol sebelum menutup
        """
        # Set konten
        self.set_content(lines)
        
        # Set footer sesuai mode
        if wait_for_key:
            self.set_footer("↑↓: Scroll | PgUp/PgDn: Halaman | Home/End: Awal/Akhir | Q: Kembali")
        else:
            self.set_footer("Tekan Q untuk kembali")
            
        # Gambar terminal
        self.draw(title)
        self.refresh()
        
        # Tunggu input jika perlu
        if wait_for_key:
            while True:
                key = self.stdscr.getch()
                result = self.handle_input(key)
                
                # Redraw setelah input
                self.draw(title)
                self.refresh()
                
                if result is False:  # Tombol keluar
                    break