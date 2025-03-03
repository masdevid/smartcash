# File: smartcash/interface/display/base_display.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk komponen display dengan setup warna dan utilitas dasar

import curses
from typing import Dict, Optional, Tuple, List, Any

class BaseDisplay:
    """Kelas dasar untuk semua komponen display dengan fungsionalitas umum."""
    
    # Konstanta warna untuk digunakan di seluruh aplikasi
    COLOR_ERROR = 1     # Merah - Error/Tidak Diatur
    COLOR_SUCCESS = 2   # Hijau - Sukses/Diatur
    COLOR_WARNING = 3   # Kuning - Peringatan/Highlight
    COLOR_INFO = 4      # Cyan - Informasi/Judul
    COLOR_SPECIAL = 5   # Magenta - Khusus
    COLOR_NAVIGATION = 6  # Biru - Navigasi
    
    def __init__(self, stdscr: curses.window, parent=None):
        """
        Inisialisasi komponen display.
        
        Args:
            stdscr: Curses window
            parent: Parent display (opsional)
        """
        self.stdscr = stdscr
        self.parent = parent
        self.height, self.width = stdscr.getmaxyx()
        
        # Area default (full window)
        self.x = 0
        self.y = 0
        self.display_width = self.width
        self.display_height = self.height
        
        # Setup warna
        self._setup_colors()
    
    def _setup_colors(self) -> None:
        """Setup pasangan warna untuk interface."""
        if not curses.has_colors():
            return
            
        curses.start_color()
        curses.use_default_colors()
        
        # Initialize color pairs
        curses.init_pair(self.COLOR_ERROR, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_SUCCESS, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_WARNING, curses.COLOR_YELLOW, -1)
        curses.init_pair(self.COLOR_INFO, curses.COLOR_CYAN, -1)
        curses.init_pair(self.COLOR_SPECIAL, curses.COLOR_MAGENTA, -1)
        curses.init_pair(self.COLOR_NAVIGATION, curses.COLOR_BLUE, -1)
    
    def set_area(self, x: int, y: int, width: int, height: int) -> None:
        """
        Atur area tampilan komponen.
        
        Args:
            x: Posisi x (kolom)
            y: Posisi y (baris)
            width: Lebar area
            height: Tinggi area
        """
        self.x = x
        self.y = y
        self.display_width = width
        self.display_height = height
    
    def clear_area(self) -> None:
        """Bersihkan area tampilan komponen."""
        for y in range(self.y, self.y + self.display_height):
            if y < self.height:  # Pastikan tidak melebihi batas layar
                # Gunakan spasi kosong untuk membersihkan
                self.safe_addstr(y, self.x, " " * self.display_width)
    
    def draw_border(self, title: Optional[str] = None) -> None:
        """
        Gambar border di sekeliling area komponen.
        
        Args:
            title: Judul border (opsional)
        """
        try:
            # Draw horizontal borders
            self.stdscr.addstr(self.y, self.x, "┌" + "─" * (self.display_width - 2) + "┐")
            self.stdscr.addstr(
                self.y + self.display_height - 1, 
                self.x, 
                "└" + "─" * (self.display_width - 2) + "┘"
            )
            
            # Draw vertical borders
            for i in range(1, self.display_height - 1):
                self.stdscr.addstr(self.y + i, self.x, "│")
                self.stdscr.addstr(self.y + i, self.x + self.display_width - 1, "│")
            
            # Draw title if provided
            if title:
                # Pastikan judul tidak terlalu panjang
                if len(title) > self.display_width - 4:
                    title = title[:self.display_width - 7] + "..."
                
                # Posisikan judul di tengah
                title_x = self.x + (self.display_width - len(title) - 2) // 2
                
                # Gambar judul dengan warna info
                self.stdscr.attron(curses.color_pair(self.COLOR_INFO))
                self.stdscr.addstr(self.y, title_x, f" {title} ")
                self.stdscr.attroff(curses.color_pair(self.COLOR_INFO))
                
        except curses.error:
            # Tangani kesalahan curses (biasanya dari percobaan menggambar di luar layar)
            pass
    
    def safe_addstr(self, y: int, x: int, text: str, color_pair: int = 0) -> None:
        """
        Tambahkan string dengan aman (menangani error di luar batas).
        
        Args:
            y: Posisi y (baris)
            x: Posisi x (kolom)
            text: Teks yang akan ditampilkan
            color_pair: ID pasangan warna (opsional)
        """
        try:
            # Pastikan koordinat dalam batas layar
            if y < 0 or y >= self.height or x < 0:
                return
                
            # Hitung berapa banyak karakter yang muat di layar
            max_len = self.width - x if x < self.width else 0
            if max_len <= 0:
                return
                
            # Potong teks jika terlalu panjang
            if len(text) > max_len:
                text = text[:max_len-3] + "..." if max_len > 3 else text[:max_len]
                
            # Tambahkan teks dengan warna yang sesuai
            if color_pair > 0:
                self.stdscr.attron(curses.color_pair(color_pair))
            
            self.stdscr.addstr(y, x, text)
            
            if color_pair > 0:
                self.stdscr.attroff(curses.color_pair(color_pair))
                
        except curses.error:
            # Abaikan kesalahan curses
            pass
    
    def get_input(
        self, 
        prompt: str, 
        y: Optional[int] = None, 
        x: Optional[int] = None,
        validator: Optional[callable] = None,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Dapatkan input dari pengguna dengan validasi.
        
        Args:
            prompt: Prompt untuk ditampilkan
            y: Posisi y (baris), relatif terhadap area komponen
            x: Posisi x (kolom), relatif terhadap area komponen
            validator: Fungsi validasi (opsional)
            default: Nilai default (opsional)
            
        Returns:
            String input atau None jika dibatalkan
        """
        # Gunakan posisi default jika tidak ditentukan
        if y is None:
            y = self.display_height - 2
        else:
            y += self.y  # Relatif terhadap komponen
            
        if x is None:
            x = 2
        else:
            x += self.x  # Relatif terhadap komponen
            
        # Simpan mode kursor saat ini dan aktifkan echo
        try:
            old_cursor = curses.curs_set(1)
            curses.echo()
        except:
            old_cursor = 0
            
        # Timeout untuk responsif
        self.stdscr.timeout(100)
        
        # Clear input area
        self.safe_addstr(y, x, " " * (self.display_width - x - 2))
        
        # Show prompt and default
        self.safe_addstr(y, x, prompt)
        prompt_len = len(prompt)
        
        if default:
            self.safe_addstr(y, x + prompt_len, f"[{default}] ")
            prompt_len += len(f"[{default}] ")
        
        # Setup input
        input_buffer = ""
        cursor_x = x + prompt_len
        
        while True:
            try:
                ch = self.stdscr.getch()
                
                if ch == -1:  # Timeout
                    continue
                elif ch in [ord('\n'), ord('\r')]:  # Enter
                    # Gunakan default jika tidak ada input
                    if not input_buffer and default:
                        input_buffer = default
                        
                    # Validasi jika ada validator
                    if validator and not validator(input_buffer):
                        continue
                    break
                elif ch == ord('\b') or ch == curses.KEY_BACKSPACE or ch == 127:  # Backspace
                    if input_buffer:
                        input_buffer = input_buffer[:-1]
                        cursor_x -= 1
                        self.safe_addstr(y, cursor_x, ' ')
                        self.stdscr.move(y, cursor_x)
                elif ch == 3:  # Ctrl+C
                    raise KeyboardInterrupt
                elif ch == 27:  # Escape
                    curses.curs_set(old_cursor)
                    curses.noecho()
                    self.stdscr.timeout(-1)
                    return None
                elif 32 <= ch <= 126:  # Karakter yang dapat dicetak
                    input_buffer += chr(ch)
                    self.safe_addstr(y, cursor_x, chr(ch))
                    cursor_x += 1
                    
                self.stdscr.refresh()
                
            except curses.error:
                continue
            except Exception as e:
                # Tangani kesalahan umum
                break
                
        # Restore terminal state
        try:
            curses.curs_set(old_cursor)
        except:
            pass
            
        curses.noecho()
        self.stdscr.timeout(-1)
        return input_buffer.strip()
    
    def draw(self) -> None:
        """
        Metode dasar untuk menggambar komponen.
        Harus diimplementasikan oleh subclass.
        """
        self.clear_area()
    
    def handle_input(self, key: int) -> Optional[bool]:
        """
        Metode dasar untuk menangani input keyboard.
        Harus diimplementasikan oleh subclass.
        
        Args:
            key: Kode tombol keyboard
            
        Returns:
            True jika input ditangani, False jika ingin kembali, None jika tidak ditangani
        """
        return None
        
    def refresh(self) -> None:
        """Refresh layar."""
        self.stdscr.refresh()
        
    def resize(self) -> None:
        """
        Tangani perubahan ukuran terminal.
        Harus dipanggil saat ukuran terminal berubah.
        """
        self.height, self.width = self.stdscr.getmaxyx()