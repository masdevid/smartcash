# File: smartcash/interface/utils/display.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk menampilkan informasi dan dialog di antarmuka

import curses
import psutil
import torch
from typing import Dict, Optional, Any, Tuple, Callable, List
from pathlib import Path

class DisplayManager:
    """Manager untuk menampilkan informasi di interface."""
    
    def __init__(self, stdscr: curses.window):
        self.stdscr = stdscr
        self.error_log = []  # Simpan riwayat error untuk referensi
        self.log_buffer = []  # Buffer untuk menyimpan pesan log
        self.max_log_lines = 5  # Jumlah maksimum baris log yang ditampilkan
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
    def _add_to_log_buffer(self, message: str, color: int) -> None:
        """
        Tambahkan pesan ke buffer log.
        
        Args:
            message: Pesan untuk ditambahkan
            color: Warna pesan
        """
        self.log_buffer.append((message, color))
        # Batasi jumlah pesan dalam buffer
        if len(self.log_buffer) > self.max_log_lines:
            self.log_buffer.pop(0)
    def _draw_log_area(self) -> None:
        """Gambar area log di bagian bawah layar."""
        height, width = self.stdscr.getmaxyx()
        
        # Hapus area log terlebih dahulu
        log_start_y = height - self.max_log_lines - 2
        for y in range(log_start_y, height):
            self.stdscr.addstr(y, 0, " " * (width - 1))
        
        # Gambar border untuk area log
        self.stdscr.attron(curses.color_pair(6))  # Blue
        self.stdscr.addstr(log_start_y, 0, "─" * (width - 1))
        self.stdscr.addstr(log_start_y, 0, "┌ Log Area ┐")
        self.stdscr.attroff(curses.color_pair(6))
        
        # Tampilkan pesan log dari buffer
        for i, (message, color) in enumerate(self.log_buffer):
            y = log_start_y + i + 1
            if y < height:
                # Truncate message if too long
                if len(message) > width - 4:
                    message = message[:width-7] + "..."
                self.stdscr.attron(curses.color_pair(color))
                self.stdscr.addstr(y, 2, message)
                self.stdscr.attroff(curses.color_pair(color))
        
        self.stdscr.refresh()        
    
    def show_error(
        self, 
        message: str, 
        timeout_ms: int = 0,  # 0 berarti tunggu key press
        show_help: bool = True   # Show helpful suggestion
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
        # Log error for history
        self.error_log.append(message)
        if len(self.error_log) > 10:  # Keep only last 10 errors
            self.error_log.pop(0)
        
        # Tambahkan ke buffer log
        error_msg = f"❌ ERROR: {message}"
        self._add_to_log_buffer(error_msg, 1)  # Red color
        
        # Tampilkan area log
        self._draw_log_area()
        
        # Jika timeout_ms > 0, tunggu timeout atau key press
        if timeout_ms > 0:
            self.stdscr.timeout(timeout_ms)
            key = self.stdscr.getch()
            self.stdscr.timeout(-1)
            return key if key != -1 else None
        else:
            # Jika show_help, tampilkan pesan bantuan
            if show_help:
                height, width = self.stdscr.getmaxyx()
                help_y = height - 1
                self.stdscr.attron(curses.color_pair(3))
                self.stdscr.addstr(help_y, 0, " " * (width - 1))
                help_msg = "💡 Tekan key untuk melanjutkan..."
                self.stdscr.addstr(help_y, (width - len(help_msg)) // 2, help_msg)
                self.stdscr.attroff(curses.color_pair(3))
                self.stdscr.refresh()
                
                # Tunggu key press
                key = self.stdscr.getch()
                
                # Hapus pesan bantuan
                self.stdscr.addstr(help_y, 0, " " * (width - 1))
                self.stdscr.refresh()
                
                return key
        
        return None

    def show_success(
        self, 
        message: str, 
        timeout_ms: int = 0
    ) -> None:
        """Tampilkan pesan sukses dengan warna hijau."""
        # Tambahkan ke buffer log
        success_msg = f"✅ {message}"
        self._add_to_log_buffer(success_msg, 2)  # Green color
        
        # Tampilkan area log
        self._draw_log_area()
        
        # Jika timeout_ms > 0, tunggu timeout
        if timeout_ms > 0:
            curses.napms(timeout_ms)
            
    def show_config_status(self, config: Dict[str, Any]) -> None:
        """Tampilkan status konfigurasi saat ini dengan error handling yang lebih baik."""
        try:
            if not config:
                return
            
            height, width = self.stdscr.getmaxyx()    
            start_y = 2
            start_x = width - 30  # Posisi x untuk status konfigurasi (30 kolom dari kanan)
            
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
            # Log error but don't crash
            self.error_log.append(f"Error menampilkan status konfigurasi: {str(e)}")
    
    def _show_config_value(self, y: int, x: int, value: Any) -> None:
        """Tampilkan nilai konfigurasi dengan warna yang sesuai dan handling error yang lebih baik."""
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
            
        except curses.error:
            # Ignore curses errors (likely window resize)
            pass
        except Exception as e:
            # Silent fail - just don't display the value
            pass

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
            
    def get_user_input(
        self, 
        prompt: str, 
        y: Optional[int] = None, 
        x: Optional[int] = None,
        timeout: int = 100,
        validator: Optional[Callable] = None,
        default: Optional[str] = None
    ) -> Optional[str]:
        """
        Dapatkan input dari user dengan error handling yang ditingkatkan.
        
        Args:
            prompt: Prompt untuk ditampilkan
            y: Posisi y (opsional)
            x: Posisi x (opsional)
            timeout: Timeout dalam milidetik
            validator: Fungsi validasi input (opsional)
            default: Nilai default jika input kosong
            
        Returns:
            String input dari user atau None jika dibatalkan
        """
        height, width = self.stdscr.getmaxyx()
        
        # Pastikan y dan x valid dan di dalam batas layar
        if y is None:
            y = height - 3
        else:
            y = max(0, min(y, height - 1))
            
        if x is None:
            x = 2
        else:
            x = max(0, min(x, width - 1))
            
        # Simpan mode kursor saat ini dan aktifkan echo
        try:
            old_cursor = curses.curs_set(1)
            curses.echo()
        except:
            old_cursor = 0
            
        self.stdscr.timeout(timeout)
        
        # Clear input area and show prompt
        try:
            self.stdscr.addstr(y, 0, " " * (width - 1))
            if x + len(prompt) < width:
                self.stdscr.addstr(y, x, prompt)
            
            # Show default value if provided
            if default:
                if x + len(prompt) + len(f"[{default}] ") < width:
                    self.stdscr.addstr(y, x + len(prompt), f"[{default}] ")
                
            self.stdscr.refresh()
        except curses.error as e:
            # Handle drawing errors
            self._add_to_log_buffer(f"Error saat menampilkan prompt: {str(e)}", 1)
            curses.curs_set(old_cursor)
            curses.noecho()
            self.stdscr.timeout(-1)
            return None
        
        # Initialize input buffer
        input_buffer = ""
        cursor_x = len(prompt) + x
        if default:
            cursor_x += len(f"[{default}] ")
        
        # Pastikan cursor_x tetap dalam batas layar
        cursor_x = min(cursor_x, width - 1)
        
        while True:
            try:
                ch = self.stdscr.getch()
                if ch == -1:  # No input within timeout
                    continue
                elif ch in [ord('\n'), ord('\r')]:  # Enter
                    # If input is empty and default exists, use default
                    if not input_buffer and default:
                        input_buffer = default
                        
                    # Validate if validator provided
                    if validator and not validator(input_buffer):
                        self._add_to_log_buffer("Input tidak valid", 1)
                        continue
                    break
                elif ch == ord('\b') or ch == curses.KEY_BACKSPACE or ch == 127:  # Backspace
                    if input_buffer:
                        input_buffer = input_buffer[:-1]
                        if cursor_x > 0:
                            cursor_x -= 1
                            try:
                                self.stdscr.addch(y, cursor_x, ' ')
                                self.stdscr.move(y, cursor_x)
                            except curses.error:
                                # Handle edge case
                                pass
                elif ch == 3:  # Ctrl+C
                    raise KeyboardInterrupt
                elif ch == 27:  # Escape
                    curses.curs_set(old_cursor)
                    curses.noecho()
                    self.stdscr.timeout(-1)
                    return None
                elif 32 <= ch <= 126:  # Printable characters
                    if cursor_x < width - 1:  # Pastikan cursor tidak melewati batas
                        input_buffer += chr(ch)
                        try:
                            self.stdscr.addch(y, cursor_x, ch)
                            cursor_x += 1
                        except curses.error:
                            # Tetap tambahkan ke buffer tapi jangan tampilkan
                            pass
                    
                self.stdscr.refresh()
                
            except curses.error:
                continue
            except Exception as e:
                self._add_to_log_buffer(f"Error saat input: {str(e)}", 1)
                curses.curs_set(old_cursor)
                curses.noecho()
                self.stdscr.timeout(-1)
                return None
                
        # Restore terminal state
        try:
            curses.curs_set(old_cursor)
        except:
            pass
            
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
        try:
            height, term_width = self.stdscr.getmaxyx()
            
            # Calculate progress
            progress = min(1.0, current / total)
            filled = int(width * progress)
            
            # Create progress bar
            bar = "█" * filled + "░" * (width - filled)
            percent = int(progress * 100)
            
            # Format pesan
            progress_msg = f"{message} [{bar}] {percent}%"
            
            # Tambahkan ke buffer log
            self._add_to_log_buffer(progress_msg, 6)  # Blue
            
            # Tampilkan area log
            self._draw_log_area()
            
        except curses.error:
            # Ignore curses errors
            pass
        except Exception as e:
            # Log error tapi jangan crash
            self.error_log.append(f"Error progress bar: {str(e)}")
    
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
        
        # Calculate dialog dimensions
        message_lines = message.split('\n')
        dialog_height = len(message_lines) + 6  # Title + message + options + padding
        dialog_width = max(40, max(len(line) for line in message_lines) + 4, len(title) + 4)
        
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
        
        # Draw message (multi-line support)
        for i, line in enumerate(message_lines):
            msg_x = dialog_x + (dialog_width - len(line)) // 2
            self.stdscr.addstr(dialog_y + 2 + i, msg_x, line)
        
        # Draw options
        options_text = " | ".join([f"{key}: {label}" for key, label in options.items()])
        options_x = dialog_x + (dialog_width - len(options_text)) // 2
        self.stdscr.addstr(dialog_y + dialog_height - 2, options_x, options_text)
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

    def show_error_history(self) -> None:
        """
        Tampilkan riwayat error untuk debugging.
        """
        if not self.error_log:
            self.show_info("Tidak ada error yang tercatat")
            return
            
        # Format error history sebagai list baris
        error_lines = [f"[{i+1}] {err}" for i, err in enumerate(self.error_log)]
        
        # Gunakan terminal mode untuk menampilkan
        self.show_terminal("Riwayat Error", error_lines)

    def show_info(
        self, 
        message: str, 
        title: str = "Informasi", 
        timeout_ms: int = 0
    ) -> None:
        """
        Tampilkan pesan informasi dalam box.
        
        Args:
            message: Pesan informasi
            title: Judul box
            timeout_ms: Timeout dalam milidetik (0 untuk tunggu keypress)
        """
        try:
            height, width = self.stdscr.getmaxyx()
            
            # Calculate box dimensions
            msg_lines = message.split('\n')
            box_height = min(len(msg_lines) + 4, height - 4)
            box_width = min(max(len(line) for line in msg_lines) + 6, width - 4)
            box_width = max(box_width, len(title) + 6)  # Pastikan judul muat
            
            # Calculate position
            box_y = (height - box_height) // 2
            box_x = (width - box_width) // 2
            
            # Create window for box
            win = curses.newwin(box_height, box_width, box_y, box_x)
            win.box()
            
            # Add title
            win.attron(curses.color_pair(4))
            title_x = (box_width - len(title) - 2) // 2
            win.addstr(0, title_x, f" {title} ")
            win.attroff(curses.color_pair(4))
            
            # Add message lines
            for i, line in enumerate(msg_lines[:box_height-4]):
                # Truncate if needed
                if len(line) > box_width - 4:
                    line = line[:box_width-7] + "..."
                win.addstr(i + 1, 2, line)
                
            # Add footer
            win.attron(curses.color_pair(6))
            footer = " Tekan key untuk menutup " if timeout_ms == 0 else " Tunggu... "
            footer_x = (box_width - len(footer)) // 2
            win.addstr(box_height-1, footer_x, footer)
            win.attroff(curses.color_pair(6))
            
            win.refresh()
            
            # Wait for keypress or timeout
            if timeout_ms > 0:
                win.timeout(timeout_ms)
                win.getch()  # Ignore result
            else:
                win.timeout(-1)
                win.getch()
                
        except curses.error:
            # Handle window size errors silently
            pass
        except Exception as e:
            # Fallback to simple error display
            try:
                self._add_to_log_buffer(f"Display error: {str(e)}", 1)
                self._draw_log_area()
                curses.napms(1000)
            except:
                pass
    def show_terminal(
        self,
        title: str,
        lines: List[str],
        wait_for_key: bool = True
    ) -> None:
        """
        Tampilkan terminal-like window dengan scrolling dan border.
        
        Args:
            title: Judul terminal
            lines: List baris teks yang akan ditampilkan
            wait_for_key: Tunggu keypress sebelum menutup
        """
        try:
            height, width = self.stdscr.getmaxyx()
            
            # Calculate dimensions
            term_height = height - 6  # Leave space for header and footer
            term_width = width - 4
            
            # Create window
            term_y = 3
            term_x = 2
            
            # Draw border around terminal area
            self.stdscr.attron(curses.color_pair(4))
            for y in range(term_y-1, term_y+term_height+1):
                if y == term_y-1:  # Top border
                    self.stdscr.addstr(y, term_x-1, "┌" + "─" * term_width + "┐")
                elif y == term_y+term_height:  # Bottom border
                    self.stdscr.addstr(y, term_x-1, "└" + "─" * term_width + "┘")
                else:  # Side borders
                    self.stdscr.addstr(y, term_x-1, "│")
                    self.stdscr.addstr(y, term_x+term_width, "│")
            
            # Add title
            title_x = term_x + (term_width - len(title)) // 2
            self.stdscr.addstr(term_y-1, title_x, f" {title} ")
            self.stdscr.attroff(curses.color_pair(4))
            
            # Draw the content with scrolling
            scroll_pos = 0
            max_scroll = max(0, len(lines) - term_height)
            
            while True:
                # Clear terminal area
                for y in range(term_height):
                    self.stdscr.addstr(term_y+y, term_x, " " * term_width)
                
                # Draw visible lines
                for i in range(min(term_height, len(lines) - scroll_pos)):
                    line = lines[scroll_pos + i]
                    
                    # Detect color codes in line
                    if line.startswith("✅"):
                        self.stdscr.attron(curses.color_pair(2))
                    elif line.startswith("❌"):
                        self.stdscr.attron(curses.color_pair(1))
                    elif line.startswith("⚠️"):
                        self.stdscr.attron(curses.color_pair(3))
                    elif line.startswith("ℹ️") or line.startswith("🔍"):
                        self.stdscr.attron(curses.color_pair(4))
                    
                    # Truncate if needed
                    if len(line) > term_width:
                        line = line[:term_width-3] + "..."
                    
                    self.stdscr.addstr(term_y+i, term_x, line)
                    
                    # Reset colors
                    for color in range(1, 7):
                        self.stdscr.attroff(curses.color_pair(color))
                
                # Draw scrollbar if needed
                if len(lines) > term_height:
                    scrollbar_height = max(1, int(term_height * term_height / len(lines)))
                    scrollbar_pos = min(
                        term_height - scrollbar_height,
                        int(scroll_pos * term_height / len(lines))
                    )
                    
                    for i in range(term_height):
                        if scrollbar_pos <= i < scrollbar_pos + scrollbar_height:
                            self.stdscr.addstr(term_y+i, term_x+term_width+1, "█")
                        else:
                            self.stdscr.addstr(term_y+i, term_x+term_width+1, "│")
                
                # Draw footer
                footer_y = term_y + term_height + 1
                self.stdscr.attron(curses.color_pair(6))
                footer = "↑↓: Scroll | Q: Kembali" if wait_for_key else "Tekan Q untuk kembali"
                footer_x = term_x + (term_width - len(footer)) // 2
                self.stdscr.addstr(footer_y, footer_x, footer)
                self.stdscr.attroff(curses.color_pair(6))
                
                self.stdscr.refresh()
                
                if not wait_for_key:
                    break
                    
                # Handle input
                key = self.stdscr.getch()
                if key in [ord('q'), ord('Q')]:
                    break
                elif key == curses.KEY_UP and scroll_pos > 0:
                    scroll_pos -= 1
                elif key == curses.KEY_DOWN and scroll_pos < max_scroll:
                    scroll_pos += 1
                elif key == curses.KEY_PPAGE:  # Page Up
                    scroll_pos = max(0, scroll_pos - term_height)
                elif key == curses.KEY_NPAGE:  # Page Down
                    scroll_pos = min(max_scroll, scroll_pos + term_height)
                elif key == curses.KEY_HOME:
                    scroll_pos = 0
                elif key == curses.KEY_END:
                    scroll_pos = max_scroll
        
        except Exception as e:
            self._add_to_log_buffer(f"Error tampilan terminal: {str(e)}", 1)
            self._draw_log_area()