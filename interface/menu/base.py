# File: smartcash/interface/menu/base.py
# Author: Alfrida Sabar
# Deskripsi: Perbaikan tampilan menu dasar untuk mengatasi masalah overlap

from typing import Dict, List, Callable, Optional
import curses

class MenuItem:
    """Item menu dengan fungsi aksi terkait."""
    
    def __init__(
        self, 
        title: str, 
        action: Callable,
        description: str = "",
        category: str = "",
        enabled: bool = True
    ):
        self.title = title
        self.action = action
        self.description = description
        self.category = category
        self.enabled = enabled

class BaseMenu:
    """Kelas dasar untuk semua menu dalam aplikasi."""
    
    def __init__(
        self, 
        title: str, 
        items: List[MenuItem],
        parent: Optional['BaseMenu'] = None
    ):
        self.title = title
        self.items = items
        self.selected = 0
        self.parent = parent
        self.categories = self._group_by_category()
        
    def _group_by_category(self) -> Dict[str, List[MenuItem]]:
        """Kelompokkan item menu berdasarkan kategori."""
        categories = {}
        for item in self.items:
            if item.category not in categories:
                categories[item.category] = []
            categories[item.category].append(item)
        return categories
    
    def draw(self, stdscr, start_y: int) -> None:
        """
        Gambar menu di layar dengan perbaikan untuk menghindari overlap deskripsi.
        
        Args:
            stdscr: Curses window
            start_y: Posisi y untuk mulai menggambar
        """
        height, width = stdscr.getmaxyx()
        
        # Area untuk konten utama (70% dari lebar)
        main_width = int(width * 0.65)
        
        # Area untuk deskripsi (30% dari lebar)
        desc_x = main_width + 2
        desc_width = width - desc_x - 2
        
        # Draw title
        title_x = (main_width - len(self.title)) // 2
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(start_y, title_x, self.title)
        stdscr.attroff(curses.color_pair(2))
        
        # Draw separator for description area
        for y in range(start_y + 1, height - 1):
            stdscr.addstr(y, main_width, "│")
        
        # Draw description header
        if desc_width > 10:  # Pastikan cukup ruang
            stdscr.attron(curses.color_pair(4))
            desc_title = " Deskripsi "
            desc_title_x = desc_x + (desc_width - len(desc_title)) // 2
            stdscr.addstr(start_y, desc_title_x, desc_title)
            stdscr.attroff(curses.color_pair(4))
        
        # Initialize variables for menu items
        current_y = start_y + 2
        item_index = 0
        
        # Draw categories and items
        for category, items in self.categories.items():
            if current_y >= height - 2:
                break
                
            # Draw category header if exists
            if category:
                stdscr.attron(curses.color_pair(4))
                stdscr.addstr(current_y, 2, f"=== {category} ===")
                stdscr.attroff(curses.color_pair(4))
                current_y += 1
            
            # Draw items in category
            for item in items:
                if current_y >= height - 2:
                    break
                    
                # Set appropriate colors and styles
                if not item.enabled:
                    stdscr.attron(curses.color_pair(3))  # Disabled items in yellow
                elif item_index == self.selected:
                    stdscr.attron(curses.color_pair(1))  # Selected items in red
                
                # Draw item title
                prefix = "> " if item_index == self.selected else "  "
                # Truncate title if too long
                title = item.title
                if len(prefix + title) > main_width - 4:
                    title = title[:main_width - 7 - len(prefix)] + "..."
                stdscr.addstr(current_y, 2, f"{prefix}{title}")
                
                # Reset colors
                stdscr.attroff(curses.color_pair(1))
                stdscr.attroff(curses.color_pair(3))
                
                # Increment counters
                current_y += 1
                item_index += 1
            
            # Add space between categories
            current_y += 1
        
        # Draw description of selected item in separate area
        if 0 <= self.selected < len(self.items):
            selected_item = self.items[self.selected]
            if selected_item.description and desc_width > 10:
                # Clear description area first
                for y in range(start_y + 1, height - 1):
                    stdscr.addstr(y, desc_x, " " * desc_width)
                
                # Format and draw multiline description
                desc_lines = selected_item.description.split('\n')
                desc_y = start_y + 2
                
                # Buat box untuk deskripsi
                box_height = min(len(desc_lines) + 4, height - start_y - 3)
                
                # Draw box border
                for y in range(box_height):
                    if y == 0:  # Top border
                        stdscr.addstr(desc_y - 1, desc_x, "┌" + "─" * (desc_width - 2) + "┐")
                    elif y == box_height - 1:  # Bottom border
                        stdscr.addstr(desc_y + y - 1, desc_x, "└" + "─" * (desc_width - 2) + "┘")
                    else:  # Side borders
                        stdscr.addstr(desc_y + y - 1, desc_x, "│")
                        stdscr.addstr(desc_y + y - 1, desc_x + desc_width - 1, "│")
                
                # Draw title in box
                item_title = f" {selected_item.title} "
                if len(item_title) > desc_width - 4:
                    item_title = item_title[:desc_width - 7] + "... "
                title_x = desc_x + (desc_width - len(item_title)) // 2
                stdscr.attron(curses.color_pair(2))
                stdscr.addstr(desc_y - 1, title_x, item_title)
                stdscr.attroff(curses.color_pair(2))
                
                # Draw description lines
                for i, line in enumerate(desc_lines):
                    if desc_y + i >= desc_y + box_height - 2:
                        break
                        
                    # Truncate line if too long
                    if len(line) > desc_width - 4:
                        line = line[:desc_width - 7] + "..."
                    
                    stdscr.addstr(desc_y + i, desc_x + 2, line)
    
    def handle_input(self, key: int) -> Optional[bool]:
        """
        Handle keyboard input.
        
        Returns:
            - True jika aksi dieksekusi
            - False jika kembali ke menu sebelumnya
            - None jika tidak ada aksi
        """
        if key == curses.KEY_UP and self.selected > 0:
            self.selected -= 1
            return None
        elif key == curses.KEY_DOWN and self.selected < len(self.items) - 1:
            self.selected += 1
            return None
        elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
            if 0 <= self.selected < len(self.items) and self.items[self.selected].enabled:
                result = self.items[self.selected].action()
                return result if result is not None else True
        return None
        
    def get_selected_item(self) -> MenuItem:
        """Dapatkan item yang sedang dipilih."""
        if 0 <= self.selected < len(self.items):
            return self.items[self.selected]
        return None
        
    def enable_item(self, title: str) -> None:
        """Aktifkan item menu berdasarkan judul."""
        for item in self.items:
            if item.title == title:
                item.enabled = True
                break
                
    def disable_item(self, title: str) -> None:
        """Nonaktifkan item menu berdasarkan judul."""
        for item in self.items:
            if item.title == title:
                item.enabled = False
                break