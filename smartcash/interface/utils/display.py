# File: smartcash/interface/utils/display.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk menampilkan informasi dan dialog di antarmuka

import curses
import psutil
import torch
from typing import Dict, Optional, Any, Tuple
from pathlib import Path

class DisplayManager:
    """Manager untuk menampilkan informasi di interface."""
    
    def __init__(self, stdscr: curses.window):
        self.stdscr = stdscr
        self.setup_colors()
        
    def setup_colors(self) -> None:
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
        
    def show_error(
        self, 
        message: str, 
        timeout_ms: int = 2000
    ) -> Optional[int]:
        """
        Tampilkan pesan error dengan warna merah.
        
        Args:
            message: Pesan error untuk ditampilkan
            timeout_ms: Timeout dalam milidetik
            
        Returns:
            Key yang ditekan atau None jika timeout
        """
        height, width = self.stdscr.getmaxyx()
        y = height - 2
        x = 2
        
        # Clear line
        self.stdscr.addstr(y, 0, " " * (width - 1))
        
        # Show error
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(y, x, f"❌ {message}")
        self.stdscr.attroff(curses.color_pair(1))
        self.stdscr.refresh()
        
        # Wait for timeout or key press
        self.stdscr.timeout(timeout_ms)
        key = self.stdscr.getch()
        self.stdscr.timeout(-1)
        
        # Clear message
        self.stdscr.addstr(y, 0, " " * (width - 1))
        self.stdscr.refresh()
        
        return key if key != -1 else None
        
    def show_success(
        self, 
        message: str, 
        timeout_ms: int = 1500
    ) -> None:
        """Tampilkan pesan sukses dengan warna hijau."""
        height, width = self.stdscr.getmaxyx()
        y = height - 2
        x = 2
        
        self.stdscr.addstr(y, 0, " " * (width - 1))
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(y, x, f"✅ {message}")
        self.stdscr.attroff(curses.color_pair(2))
        self.stdscr.refresh()
        
        curses.napms(timeout_ms)
        self.stdscr.addstr(y, 0, " " * (width - 1))
        self.stdscr.refresh()
        
    def show_config_status(self, config: Dict[str, Any], start_y: int = 2, start_x: int = 50) -> None:
        """Tampilkan status konfigurasi saat ini."""
        try:
            if not config:
                return
                
            self.stdscr.addstr(start_y, start_x, "Status Konfigurasi:")
            y = start_y + 2
            
            # Data source
            self.stdscr.addstr(y, start_x, "Sumber Data: ")
            self._show_config_value(y, start_x + 16, config.get('data_source'))
            y += 1
            
            # Detection mode
            self.stdscr.addstr(y, start_x, "Mode Deteksi: ")
            self._show_config_value(y, start_x + 16, config.get('detection_mode'))
            y += 1
            
            # Backbone
            self.stdscr.addstr(y, start_x, "Arsitektur: ")
            self._show_config_value(y, start_x + 16, config.get('backbone'))
            y += 2
            
            # Training parameters
            training = config.get('training', {})
            if training:
                self.stdscr.addstr(y, start_x, "Parameter Pelatihan:")
                y += 1
                
                params = [
                    ('batch_size', 'Ukuran Batch', '32'),
                    ('learning_rate', 'Learning Rate', '0.001'),
                    ('epochs', 'Epoch', '100'),
                    ('early_stopping_patience', 'Early Stopping', '10')
                ]
                
                for param, label, default in params:
                    value = training.get(param, default)
                    self.stdscr.addstr(y, start_x + 2, f"{label}: ")
                    self._show_config_value(y, start_x + 20, value)
                    y += 1
                    
        except curses.error:
            # Ignore curses errors - usually due to window resize
            pass
        except Exception as e:
            self.logger.error(f"Error showing config status: {str(e)}")
    
    def show_system_status(self) -> None:
        """Tampilkan status sistem di header."""
        height, width = self.stdscr.getmaxyx()
        
        # Get system info
        cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        gpu_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name() if gpu_available else "Tidak tersedia"
        
        # Format status items
        status = [
            f"CPU: {cpu_percent}%",
            f"RAM: {mem.percent}%",
            f"GPU: {gpu_name}"
        ]
        
        # Draw status line
        y = 1
        x = 2
        for i, item in enumerate(status):
            if i > 0:
                # Draw separator
                self.stdscr.addstr(y, x - 1, "│")
                x += 1
                
            # Set appropriate color based on value
            if i == 2:  # GPU status
                if gpu_available:
                    self.stdscr.attron(curses.color_pair(2))  # Green
                else:
                    self.stdscr.attron(curses.color_pair(1))  # Red
            else:
                value = float(item.split(":")[1].strip().rstrip("%"))
                if value > 80:
                    self.stdscr.attron(curses.color_pair(1))  # Red
                else:
                    self.stdscr.attron(curses.color_pair(2))  # Green
                    
            self.stdscr.addstr(y, x, item)
            
            # Reset colors
            for color in range(1, 7):
                self.stdscr.attroff(curses.color_pair(color))
                
            x += len(item) + 2
            
        # Draw help text
        help_text = "↑↓: Navigasi | Enter: Pilih | Ctrl+C: Keluar"
        help_x = width - len(help_text) - 2
        self.stdscr.attron(curses.color_pair(6))  # Blue
        self.stdscr.addstr(y, help_x, help_text)
        self.stdscr.attroff(curses.color_pair(6))
            
    def _show_config_value(self, y: int, x: int, value: str) -> None:
        """Tampilkan nilai konfigurasi dengan warna yang sesuai."""
        try:
            # Ensure value is string
            value_str = str(value) if value is not None else 'Belum dipilih'
            
            if value_str == 'Belum dipilih':
                self.stdscr.attron(curses.color_pair(1))
            else:
                self.stdscr.attron(curses.color_pair(2))
                
            self.stdscr.addstr(y, x, value_str)
            self.stdscr.attroff(curses.color_pair(1))
            self.stdscr.attroff(curses.color_pair(2))
            
        except Exception as e:
            self.logger.error(f"Error showing config value: {str(e)}")
            # Don't re-raise - just log and continue
        
    def get_user_input(
        self, 
        prompt: str, 
        y: Optional[int] = None, 
        x: Optional[int] = None,
        timeout: int = 100,
        validator: Optional[callable] = None
    ) -> Optional[str]:
        """
        Dapatkan input dari user dengan timeout dan validasi.
        
        Args:
            prompt: Prompt untuk ditampilkan
            y: Posisi y (opsional)
            x: Posisi x (opsional)
            timeout: Timeout dalam milidetik
            validator: Fungsi validasi input (opsional)
            
        Returns:
            String input dari user atau None jika dibatalkan
        """
        height, width = self.stdscr.getmaxyx()
        
        if y is None:
            y = height - 3
        if x is None:
            x = 2
            
        curses.echo()
        self.stdscr.timeout(timeout)
        
        # Clear input area and show prompt
        self.stdscr.addstr(y, 0, " " * (width - 1))
        self.stdscr.addstr(y, x, prompt)
        self.stdscr.refresh()
        
        # Initialize input buffer
        input_buffer = ""
        cursor_x = len(prompt) + x
        
        while True:
            try:
                ch = self.stdscr.getch()
                if ch == -1:  # No input within timeout
                    continue
                elif ch in [ord('\n'), ord('\r')]:  # Enter
                    if validator and not validator(input_buffer):
                        self.show_error("Input tidak valid")
                        continue
                    break
                elif ch == ord('\b') or ch == curses.KEY_BACKSPACE:  # Backspace
                    if input_buffer:
                        input_buffer = input_buffer[:-1]
                        cursor_x -= 1
                        self.stdscr.addch(y, cursor_x, ' ')
                        self.stdscr.move(y, cursor_x)
                elif ch == 3:  # Ctrl+C
                    raise KeyboardInterrupt
                elif ch == 27:  # Escape
                    return None
                elif 32 <= ch <= 126:  # Printable characters
                    input_buffer += chr(ch)
                    self.stdscr.addch(y, cursor_x, ch)
                    cursor_x += 1
                    
                self.stdscr.refresh()
                
            except curses.error:
                continue
                
        curses.noecho()
        self.stdscr.timeout(-1)
        return input_buffer.strip()
        
    def show_progress(
        self, 
        message: str,
        current: int,
        total: int,
        width: int = 40
    ) -> None:
        """
        Tampilkan progress bar.
        
        Args:
            message: Pesan yang ditampilkan
            current: Nilai progress saat ini
            total: Nilai total progress
            width: Lebar progress bar
        """
        height, term_width = self.stdscr.getmaxyx()
        y = height - 4
        x = 2
        
        # Calculate progress
        progress = min(1.0, current / total)
        filled = int(width * progress)
        
        # Create progress bar
        bar = "█" * filled + "░" * (width - filled)
        percent = int(progress * 100)
        
        # Clear line and show progress
        self.stdscr.addstr(y, 0, " " * (term_width - 1))
        self.stdscr.addstr(y, x, f"{message} [{bar}] {percent}%")
        self.stdscr.refresh()
        
    def clear_message_area(self) -> None:
        """Bersihkan area pesan di bagian bawah layar."""
        height, width = self.stdscr.getmaxyx()
        for y in range(height - 4, height - 1):
            self.stdscr.addstr(y, 0, " " * (width - 1))
        self.stdscr.refresh()
        
    def show_dialog(
        self,
        title: str,
        message: str,
        options: Dict[str, str] = {"y": "Ya", "n": "Tidak"}
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
        height, width = self.stdscr.getmaxyx()
        dialog_height = 6
        dialog_width = max(40, len(message) + 4)
        
        # Calculate dialog position
        dialog_y = (height - dialog_height) // 2
        dialog_x = (width - dialog_width) // 2
        
        # Save screen content
        old_screen = curses.newwin(height, width, 0, 0)
        old_screen.overlay(self.stdscr)
        
        # Draw dialog box
        self.stdscr.attron(curses.color_pair(4))
        for y in range(dialog_height):
            self.stdscr.addstr(
                dialog_y + y,
                dialog_x,
                "│" + " " * (dialog_width-2) + "│"
            )
        self.stdscr.addstr(
            dialog_y,
            dialog_x,
            "┌" + "─" * (dialog_width-2) + "┐"
        )
        self.stdscr.addstr(
            dialog_y + dialog_height - 1,
            dialog_x,
            "└" + "─" * (dialog_width-2) + "┘"
        )
        
        # Draw title
        title_x = dialog_x + (dialog_width - len(title)) // 2
        self.stdscr.addstr(dialog_y, title_x, f" {title} ")
        
        # Draw message
        msg_x = dialog_x + (dialog_width - len(message)) // 2
        self.stdscr.addstr(dialog_y + 2, msg_x, message)
        
        # Draw options
        options_text = " | ".join([f"{key}: {label}" for key, label in options.items()])
        options_x = dialog_x + (dialog_width - len(options_text)) // 2
        self.stdscr.addstr(dialog_y + 4, options_x, options_text)
        self.stdscr.attroff(curses.color_pair(4))
        
        # Get user input
        self.stdscr.refresh()
        while True:
            key = self.stdscr.getch()
            if key == 27:  # Escape
                result = None
                break
            elif chr(key).lower() in options:
                result = chr(key).lower()
                break
        
        # Restore screen content
        self.stdscr.clear()
        old_screen.overlay(self.stdscr)
        self.stdscr.refresh()
        
        return result

    def show_help(self, title: str, content: Dict[str, str]) -> None:
        """
        Tampilkan layar bantuan dengan scrolling.
        
        Args:
            title: Judul layar bantuan
            content: Dict berisi kategori dan konten bantuan
        """
        height, width = self.stdscr.getmaxyx()
        scroll_pos = 0
        
        # Format help content
        formatted_lines = []
        for category, text in content.items():
            formatted_lines.append(f"=== {category} ===")
            formatted_lines.extend(text.split('\n'))
            formatted_lines.append("")  # Add blank line between categories
        
        while True:
            self.stdscr.clear()
            
            # Draw title
            title_x = (width - len(title)) // 2
            self.stdscr.attron(curses.color_pair(4))
            self.stdscr.addstr(0, title_x, title)
            self.stdscr.attroff(curses.color_pair(4))
            
            # Draw help content with scrolling
            content_height = height - 4  # Reserve space for title and footer
            for i in range(content_height):
                line_idx = scroll_pos + i
                if line_idx >= len(formatted_lines):
                    break
                    
                line = formatted_lines[line_idx]
                if line.startswith("==="):  # Category header
                    self.stdscr.attron(curses.color_pair(2))
                    self.stdscr.addstr(i + 2, 2, line)
                    self.stdscr.attroff(curses.color_pair(2))
                else:
                    self.stdscr.addstr(i + 2, 2, line)
            
            # Draw scrollbar if needed
            if len(formatted_lines) > content_height:
                scrollbar_height = int(content_height * content_height / len(formatted_lines))
                scrollbar_pos = int(scroll_pos * content_height / len(formatted_lines))
                
                for i in range(content_height):
                    if scrollbar_pos <= i < scrollbar_pos + scrollbar_height:
                        self.stdscr.addstr(i + 2, width - 2, "█")
                    else:
                        self.stdscr.addstr(i + 2, width - 2, "│")
            
            # Draw footer
            footer = "↑↓: Scroll | Q: Kembali"
            footer_x = (width - len(footer)) // 2
            self.stdscr.attron(curses.color_pair(6))
            self.stdscr.addstr(height - 1, footer_x, footer)
            self.stdscr.attroff(curses.color_pair(6))
            
            self.stdscr.refresh()
            
            # Handle input
            key = self.stdscr.getch()
            if key in [ord('q'), ord('Q'), 27]:  # q, Q, or Escape
                break
            elif key == curses.KEY_UP and scroll_pos > 0:
                scroll_pos -= 1
            elif key == curses.KEY_DOWN and scroll_pos < len(formatted_lines) - content_height:
                scroll_pos += 1
            elif key == curses.KEY_PPAGE:  # Page Up
                scroll_pos = max(0, scroll_pos - content_height)
            elif key == curses.KEY_NPAGE:  # Page Down
                scroll_pos = min(
                    len(formatted_lines) - content_height,
                    scroll_pos + content_height
                )

    def show_version_info(self) -> None:
        """Tampilkan informasi versi aplikasi."""
        height, width = self.stdscr.getmaxyx()
        version_info = {
            "Versi": "1.0.0",
            "Commit": "main-20240220",
            "Build": "20240220.1",
            "Python": "3.9.7",
            "PyTorch": torch.__version__,
            "CUDA": torch.version.cuda if torch.cuda.is_available() else "Tidak tersedia"
        }
        
        # Calculate box dimensions
        box_width = max(len(key) + len(str(value)) + 4 for key, value in version_info.items()) + 4
        box_height = len(version_info.items()) + 4
        
        # Calculate box position
        box_y = (height - box_height) // 2
        box_x = (width - box_width) // 2
        
        # Draw box
        self.stdscr.attron(curses.color_pair(4))
        for y in range(box_height):
            self.stdscr.addstr(
                box_y + y,
                box_x,
                "│" + " " * (box_width-2) + "│"
            )
        self.stdscr.addstr(
            box_y,
            box_x,
            "┌" + "─" * (box_width-2) + "┐"
        )
        self.stdscr.addstr(
            box_y + box_height - 1,
            box_x,
            "└" + "─" * (box_width-2) + "┘"
        )
        self.stdscr.attroff(curses.color_pair(4))
        
        # Draw title
        title = "Informasi Versi"
        title_x = box_x + (box_width - len(title)) // 2
        self.stdscr.attron(curses.color_pair(2))
        self.stdscr.addstr(box_y, title_x, f" {title} ")
        self.stdscr.attroff(curses.color_pair(2))
        
        # Draw version information
        for i, (key, value) in enumerate(version_info.items()):
            y = box_y + i + 2
            self.stdscr.addstr(y, box_x + 2, key)
            self.stdscr.addstr(y, box_x + box_width - len(str(value)) - 2, str(value))
        
        # Wait for any key
        self.stdscr.refresh()
        self.stdscr.getch()