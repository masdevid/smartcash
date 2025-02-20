# File: smartcash/interface/menu/base.py
# Author: Alfrida Sabar
# Deskripsi: Komponen menu dasar untuk antarmuka SmartCash

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
        """Gambar menu di layar."""
        height, width = stdscr.getmaxyx()
        
        # Draw title
        title_x = (width - len(self.title)) // 2
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(start_y, title_x, self.title)
        stdscr.attroff(curses.color_pair(2))
        
        current_y = start_y + 2
        item_index = 0
        
        # Draw categories and items
        for category, items in self.categories.items():
            if current_y >= height:
                break
                
            # Draw category header if exists
            if category:
                stdscr.attron(curses.color_pair(4))
                stdscr.addstr(current_y, 2, f"=== {category} ===")
                stdscr.attroff(curses.color_pair(4))
                current_y += 1
            
            # Draw items in category
            for item in items:
                if current_y >= height:
                    break
                    
                # Set appropriate colors and styles
                if not item.enabled:
                    stdscr.attron(curses.color_pair(3))  # Disabled items in yellow
                elif item_index == self.selected:
                    stdscr.attron(curses.color_pair(1))  # Selected items in red
                
                # Draw item title
                prefix = "> " if item_index == self.selected else "  "
                stdscr.addstr(current_y, 2, f"{prefix}{item.title}")
                
                # Reset colors
                stdscr.attroff(curses.color_pair(1))
                stdscr.attroff(curses.color_pair(3))
                
                # Draw description for selected item
                if item_index == self.selected and item.description:
                    if current_y + 1 < height:
                        stdscr.addstr(current_y + 1, 4, item.description, 
                                    curses.color_pair(6))
                        current_y += 2
                    else:
                        current_y += 1
                else:
                    current_y += 1
                    
                item_index += 1
            
            current_y += 1  # Space between categories
    
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
            if self.items[self.selected].enabled:
                result = self.items[self.selected].action()
                return result if result is not None else True
        return None
        
    def get_selected_item(self) -> MenuItem:
        """Dapatkan item yang sedang dipilih."""
        return self.items[self.selected]
        
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