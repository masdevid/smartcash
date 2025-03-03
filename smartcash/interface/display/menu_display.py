# File: smartcash/interface/display/menu_display.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk menampilkan menu dan menangani interaksi menu

import curses
from typing import Dict, List, Optional, Callable, Tuple, Any

from smartcash.interface.display.base_display import BaseDisplay
from smartcash.interface.menu.base import MenuItem

class MenuDisplay(BaseDisplay):
    """
    Komponen untuk menampilkan menu dan menangani interaksi menu.
    Menempati 60% bagian kiri dari area utama.
    """
    
    def __init__(
        self, 
        stdscr: curses.window, 
        parent=None,
        description_callback: Optional[Callable[[MenuItem], None]] = None
    ):
        """
        Inisialisasi menu display.
        
        Args:
            stdscr: Curses window
            parent: Parent display (opsional)
            description_callback: Callback untuk menampilkan deskripsi (opsional)
        """
        super().__init__(stdscr, parent)
        self.title = "Menu"
        self.items = []
        self.categories = {}  # Grup item berdasarkan kategori
        self.selected = 0
        self.description_callback = description_callback
        
        # Mapping indeks item -> posisi y pada layar
        self.item_positions = []
    
    def set_menu(self, title: str, items: List[MenuItem]) -> None:
        """
        Set menu dengan item dan judul baru.
        
        Args:
            title: Judul menu
            items: List item menu
        """
        self.title = title
        self.items = items
        self.selected = 0
        
        # Group items by category
        self.categories = {}
        for item in items:
            category = item.category or "Umum"
            if category not in self.categories:
                self.categories[category] = []
            self.categories[category].append(item)
    
    def draw(self) -> None:
        """Gambar menu dengan kategori dan item."""
        self.clear_area()
        self.item_positions = []
        
        # Gambar judul menu
        title_x = self.x + (self.display_width - len(self.title)) // 2
        self.safe_addstr(
            self.y, 
            title_x, 
            self.title, 
            self.COLOR_INFO
        )
        
        current_y = self.y + 2
        item_index = 0
        
        # Gambar setiap kategori dan item-nya
        for category, items in self.categories.items():
            # Gambar judul kategori jika tidak kosong
            if category != "Umum":
                self.safe_addstr(
                    current_y, 
                    self.x + 2, 
                    f"=== {category} ===", 
                    self.COLOR_INFO
                )
                current_y += 1
            
            # Gambar setiap item dalam kategori
            for item in items:
                # Simpan posisi y untuk item ini
                self.item_positions.append(current_y)
                
                # Pilih warna yang sesuai
                if not item.enabled:
                    color = self.COLOR_WARNING  # Item dinonaktifkan
                elif item_index == self.selected:
                    color = self.COLOR_SUCCESS  # Item dipilih
                else:
                    color = 0  # Warna default
                
                # Gambar item
                prefix = "> " if item_index == self.selected else "  "
                
                # Potong judul jika terlalu panjang
                title = item.title
                max_title_width = self.display_width - 6  # Jaga ruang untuk prefix
                if len(prefix + title) > max_title_width:
                    title = title[:max_title_width-len(prefix)-3] + "..."
                
                self.safe_addstr(
                    current_y, 
                    self.x + 2, 
                    f"{prefix}{title}", 
                    color
                )
                
                current_y += 1
                item_index += 1
                
            # Tambahkan spasi antar kategori
            current_y += 1
            
        # Tampilkan deskripsi item terpilih jika callback tersedia
        if self.description_callback and 0 <= self.selected < len(self.items):
            self.description_callback(self.items[self.selected])
            
        # Tampilkan navigasi di bagian bawah
        help_text = "↑↓: Navigasi | Enter: Pilih | Q: Kembali"
        help_y = self.y + self.display_height - 1
        help_x = self.x + (self.display_width - len(help_text)) // 2
        
        self.safe_addstr(
            help_y, 
            help_x, 
            help_text, 
            self.COLOR_NAVIGATION
        )
    
    def handle_input(self, key: int) -> Optional[bool]:
        """
        Tangani input keyboard untuk navigasi menu.
        
        Args:
            key: Kode tombol keyboard
            
        Returns:
            True jika berhasil menangani input, False untuk kembali, None jika tidak ditangani
        """
        if key == curses.KEY_UP and self.selected > 0:
            self.selected -= 1
            return True
        elif key == curses.KEY_DOWN and self.selected < len(self.items) - 1:
            self.selected += 1
            return True
        elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
            # Eksekusi aksi item yang dipilih jika diaktifkan
            if 0 <= self.selected < len(self.items) and self.items[self.selected].enabled:
                result = self.items[self.selected].action()
                return result if result is not None else True
        return None
    
    def enable_item(self, title: str) -> None:
        """
        Aktifkan item menu berdasarkan judul.
        
        Args:
            title: Judul item yang akan diaktifkan
        """
        for item in self.items:
            if item.title == title:
                item.enabled = True
                break
    
    def disable_item(self, title: str) -> None:
        """
        Nonaktifkan item menu berdasarkan judul.
        
        Args:
            title: Judul item yang akan dinonaktifkan
        """
        for item in self.items:
            if item.title == title:
                item.enabled = False
                break
    
    def get_selected_item(self) -> Optional[MenuItem]:
        """
        Dapatkan item menu yang dipilih saat ini.
        
        Returns:
            Item yang dipilih atau None jika tidak ada
        """
        if 0 <= self.selected < len(self.items):
            return self.items[self.selected]
        return None