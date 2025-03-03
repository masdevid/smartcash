# File: smartcash/interface/display/dialog_display.py
# Author: Alfrida Sabar
# Deskripsi: Komponen untuk menampilkan dialog dan form input dengan menangani validasi

import curses
import time
from typing import Dict, List, Optional, Callable, Tuple, Any, Union

from smartcash.interface.display.base_display import BaseDisplay

class DialogDisplay(BaseDisplay):
    """
    Komponen untuk menampilkan dialog, mengambil input, dan konfirmasi.
    Mendukung berbagai tipe dialog:
    - Dialog info/error
    - Form input dengan validasi
    - Dialog konfirmasi dengan opsi
    - Dialog progress
    """
    
    # Tipe dialog
    TYPE_INFO = 0      # Informasi umum
    TYPE_ERROR = 1     # Pesan error
    TYPE_WARNING = 2   # Peringatan
    TYPE_CONFIRM = 3   # Konfirmasi dengan opsi
    TYPE_FORM = 4      # Form input
    TYPE_PROGRESS = 5  # Progress dialog
    
    def __init__(self, stdscr: curses.window, parent=None):
        """
        Inisialisasi dialog display.
        
        Args:
            stdscr: Curses window
            parent: Parent display (opsional)
        """
        super().__init__(stdscr, parent)
        
        # Karakter untuk border dialog (dapat dikonfigurasi)
        self.border_chars = {
            'top_left': '┌',
            'top_right': '┐',
            'bottom_left': '└',
            'bottom_right': '┘',
            'horizontal': '─',
            'vertical': '│'
        }
        
        # Auto-sizing
        self.auto_size = True
        
        # Prefiks untuk tipe dialog
        self.type_prefixes = {
            self.TYPE_INFO: "ℹ️",
            self.TYPE_ERROR: "❌",
            self.TYPE_WARNING: "⚠️",
            self.TYPE_CONFIRM: "❓",
            self.TYPE_FORM: "📝",
            self.TYPE_PROGRESS: "🔄"
        }
    
    def set_size(self, width: int, height: int) -> None:
        """
        Atur ukuran dialog secara manual.
        
        Args:
            width: Lebar dialog
            height: Tinggi dialog
        """
        self.display_width = min(width, self.width - 4)
        self.display_height = min(height, self.height - 4)
        self.auto_size = False
    
    def center_dialog(self) -> None:
        """Posisikan dialog di tengah layar."""
        self.x = (self.width - self.display_width) // 2
        self.y = (self.height - self.display_height) // 2
    
    def _calculate_size(
        self, 
        title: str, 
        message: List[str], 
        options: Dict[str, str] = None
    ) -> None:
        """
        Hitung ukuran dialog berdasarkan konten.
        
        Args:
            title: Judul dialog
            message: Pesan dialog dalam bentuk list baris
            options: Opsi dialog (opsional)
        """
        if not self.auto_size:
            return
            
        # Minimal width and height
        min_width = max(40, len(title) + 6)
        min_height = 6  # Judul + minimal 1 baris pesan + baris opsi + padding
        
        # Calculate width based on content
        content_width = max([len(line) for line in message])
        if options:
            options_text = " | ".join([f"{k}: {v}" for k, v in options.items()])
            content_width = max(content_width, len(options_text))
            
        # Final width and height with padding
        self.display_width = min(max(min_width, content_width + 6), self.width - 4)
        self.display_height = min(min_height + len(message), self.height - 4)
        
        # Center the dialog
        self.center_dialog()
    
    def _draw_dialog_border(self, title: str, dialog_type: int = TYPE_INFO) -> None:
        """
        Gambar border dialog dengan judul.
        
        Args:
            title: Judul dialog
            dialog_type: Tipe dialog (default: INFO)
        """
        # Draw horizontal borders
        self.safe_addstr(
            self.y, 
            self.x, 
            f"{self.border_chars['top_left']}{self.border_chars['horizontal'] * (self.display_width - 2)}{self.border_chars['top_right']}"
        )
        self.safe_addstr(
            self.y + self.display_height - 1, 
            self.x, 
            f"{self.border_chars['bottom_left']}{self.border_chars['horizontal'] * (self.display_width - 2)}{self.border_chars['bottom_right']}"
        )
        
        # Draw vertical borders
        for i in range(1, self.display_height - 1):
            self.safe_addstr(self.y + i, self.x, self.border_chars['vertical'])
            self.safe_addstr(self.y + i, self.x + self.display_width - 1, self.border_chars['vertical'])
        
        # Draw title with appropriate color and prefix
        if title:
            # Add prefix based on dialog type
            prefix = self.type_prefixes.get(dialog_type, "")
            full_title = f" {prefix} {title} " if prefix else f" {title} "
            
            # Ensure title fits
            if len(full_title) > self.display_width - 4:
                full_title = full_title[:self.display_width - 7] + "... "
                
            # Calculate title position
            title_x = self.x + (self.display_width - len(full_title)) // 2
            
            # Choose color based on dialog type
            color = self.COLOR_INFO
            if dialog_type == self.TYPE_ERROR:
                color = self.COLOR_ERROR
            elif dialog_type == self.TYPE_WARNING:
                color = self.COLOR_WARNING
            elif dialog_type == self.TYPE_CONFIRM:
                color = self.COLOR_SPECIAL
                
            # Draw title
            self.safe_addstr(self.y, title_x, full_title, color)
    
    def show_dialog(
        self, 
        title: str, 
        message: Union[str, List[str]], 
        dialog_type: int = TYPE_INFO, 
        options: Dict[str, str] = None, 
        timeout_ms: int = 0
    ) -> Optional[str]:
        """
        Tampilkan dialog dan tunggu input.
        
        Args:
            title: Judul dialog
            message: Pesan dialog
            dialog_type: Tipe dialog
            options: Opsi pilihan {key: label}
            timeout_ms: Timeout dalam milidetik (0 = tunggu sampai key press)
            
        Returns:
            Key dari opsi yang dipilih atau None jika dibatalkan/timeout
        """
        # Process message into list if it's a string
        if isinstance(message, str):
            # Split by newlines and wrap long lines
            message_lines = []
            for line in message.split("\n"):
                # Simple word wrap for long lines
                while line and len(line) > (self.width - 10):
                    # Find space to break
                    break_pos = line[:self.width-10].rfind(" ")
                    if break_pos == -1:  # No space found
                        break_pos = self.width - 10
                    
                    message_lines.append(line[:break_pos])
                    line = line[break_pos:].lstrip()
                
                if line:  # Add remaining text
                    message_lines.append(line)
        else:
            message_lines = message
        
        # Default options if none provided
        if options is None:
            if dialog_type in [self.TYPE_INFO, self.TYPE_ERROR, self.TYPE_WARNING]:
                options = {"o": "OK"}
            elif dialog_type == self.TYPE_CONFIRM:
                options = {"y": "Ya", "n": "Tidak"}
        
        # Calculate dialog size based on content
        self._calculate_size(title, message_lines, options)
        
        # Store current screen to restore later
        screen_backup = curses.newwin(self.height, self.width, 0, 0)
        screen_backup.overlay(self.stdscr)
        
        try:
            # Clear dialog area
            for y in range(self.y, self.y + self.display_height):
                self.safe_addstr(y, self.x, " " * self.display_width)
            
            # Draw dialog border and title
            self._draw_dialog_border(title, dialog_type)
            
            # Draw message content
            content_start_y = self.y + 2
            for i, line in enumerate(message_lines):
                if content_start_y + i >= self.y + self.display_height - 2:
                    # No more space for content
                    break
                
                # Center short lines, left-align longer ones
                if len(line) < self.display_width - 10:
                    line_x = self.x + (self.display_width - len(line)) // 2
                else:
                    line_x = self.x + 3
                
                self.safe_addstr(content_start_y + i, line_x, line)
            
            # Draw options if any
            if options:
                options_y = self.y + self.display_height - 2
                options_text = " | ".join([f"{k}: {v}" for k, v in options.items()])
                options_x = self.x + (self.display_width - len(options_text)) // 2
                
                # Draw with navigation color
                self.safe_addstr(options_y, options_x, options_text, self.COLOR_NAVIGATION)
            
            # Refresh to show dialog
            self.stdscr.refresh()
            
            # Handle timeout or wait for keypress
            if timeout_ms > 0:
                self.stdscr.timeout(timeout_ms)
                key = self.stdscr.getch()
                self.stdscr.timeout(-1)  # Reset timeout
                
                if key == -1:  # Timeout
                    return None
                    
                if chr(key).lower() in options:
                    return chr(key).lower()
                    
                return None
            else:
                # Wait for valid keypress
                while True:
                    key = self.stdscr.getch()
                    
                    if key == 27:  # Esc key
                        return None
                    
                    if key >= 0 and chr(key).lower() in options:
                        return chr(key).lower()
                        
        finally:
            # Restore screen
            self.stdscr.clear()
            screen_backup.overlay(self.stdscr)
            self.stdscr.refresh()
            
        return None
    
    def show_form(
        self, 
        title: str, 
        fields: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Tampilkan form dialog untuk meminta input banyak field.
        
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
            Dict berisi nilai input: {'field_name': value, ...}
            atau dict kosong jika dibatalkan
        """
        if not fields:
            return {}
        
        # Prepare message lines with field labels
        message_lines = ["Silakan isi form berikut:"]
        for field in fields:
            label = field.get('label', field['name'])
            message_lines.append(f"{label}:")
        
        # Calculate form size
        self._calculate_size(title, message_lines)
        
        # Results to return
        results = {}
        for field in fields:
            # Default value for the field
            results[field['name']] = field.get('default', '')
        
        # Save current screen
        screen_backup = curses.newwin(self.height, self.width, 0, 0)
        screen_backup.overlay(self.stdscr)
        
        try:
            # Draw form shell
            self._draw_dialog_border(title, self.TYPE_FORM)
            
            # Draw form instructions
            self.safe_addstr(
                self.y + 1, 
                self.x + 2, 
                "Tekan Tab untuk berpindah field, Esc untuk batal, Enter untuk konfirmasi", 
                self.COLOR_INFO
            )
            
            # Setup initial field positions
            field_positions = []
            content_start_y = self.y + 3
            
            for i, field in enumerate(fields):
                label = field.get('label', field['name'])
                label_y = content_start_y + i * 2
                
                if label_y >= self.y + self.display_height - 2:
                    # No more space for fields
                    break
                
                field_positions.append(label_y)
                self.safe_addstr(label_y, self.x + 3, f"{label}:")
                
                # Draw default value or placeholder
                value = str(field.get('default', ''))
                if field.get('type') == 'select' and field.get('options'):
                    # For select fields, show the options
                    options_text = " | ".join(field['options'])
                    self.safe_addstr(label_y + 1, self.x + 5, options_text, self.COLOR_WARNING)
                else:
                    # For text/number fields, show the default
                    self.safe_addstr(label_y + 1, self.x + 5, value or "[Kosong]", self.COLOR_WARNING)
            
            # Draw navigation help
            nav_text = "Tab: Next | Esc: Cancel | Enter: Submit"
            nav_y = self.y + self.display_height - 2
            nav_x = self.x + (self.display_width - len(nav_text)) // 2
            self.safe_addstr(nav_y, nav_x, nav_text, self.COLOR_NAVIGATION)
            
            # Active field index
            active_field = 0
            
            # Input loop
            while True:
                # Highlight active field
                for i, y in enumerate(field_positions):
                    prefix = "> " if i == active_field else "  "
                    label = fields[i].get('label', fields[i]['name'])
                    self.safe_addstr(y, self.x + 1, prefix)
                    
                    # Redraw field value
                    value_y = y + 1
                    field_type = fields[i].get('type', 'text')
                    
                    # Clear value line
                    self.safe_addstr(value_y, self.x + 5, " " * (self.display_width - 10))
                    
                    if field_type == 'select' and fields[i].get('options'):
                        # For select fields, show options with current highlighted
                        options = fields[i]['options']
                        current_value = results[fields[i]['name']]
                        options_text = ""
                        
                        for j, opt in enumerate(options):
                            if opt == current_value:
                                options_text += f"[{opt}] "
                            else:
                                options_text += f"{opt} "
                                
                        self.safe_addstr(value_y, self.x + 5, options_text, self.COLOR_WARNING)
                    else:
                        # For text/number fields, show the value
                        value = str(results[fields[i]['name']]) or "[Kosong]"
                        self.safe_addstr(value_y, self.x + 5, value, self.COLOR_WARNING)
                
                # Refresh screen
                self.stdscr.refresh()
                
                # Get user input
                key = self.stdscr.getch()
                
                if key == 27:  # Esc key - cancel
                    return {}
                    
                elif key == 9 or key == curses.KEY_DOWN:  # Tab or Down - next field
                    active_field = (active_field + 1) % len(field_positions)
                    
                elif key == curses.KEY_UP:  # Up - previous field
                    active_field = (active_field - 1) % len(field_positions)
                    
                elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:  # Enter - submit
                    # Validate all fields
                    valid = True
                    for field in fields:
                        validator = field.get('validator')
                        if validator and not validator(results[field['name']]):
                            valid = False
                            break
                            
                    if valid:
                        return results
                    else:
                        # Show validation error
                        self.show_dialog(
                            "Validasi Gagal",
                            "Beberapa nilai tidak valid. Silakan periksa kembali input Anda.",
                            self.TYPE_ERROR
                        )
                        
                        # Redraw form (screen was replaced by error dialog)
                        self._draw_dialog_border(title, self.TYPE_FORM)
                
                elif key == curses.KEY_LEFT and fields[active_field].get('type') == 'select':
                    # For select fields, cycle options backward
                    options = fields[active_field].get('options', [])
                    if options:
                        current = results[fields[active_field]['name']]
                        try:
                            idx = options.index(current)
                            idx = (idx - 1) % len(options)
                            results[fields[active_field]['name']] = options[idx]
                        except ValueError:
                            results[fields[active_field]['name']] = options[0]
                
                elif key == curses.KEY_RIGHT and fields[active_field].get('type') == 'select':
                    # For select fields, cycle options forward
                    options = fields[active_field].get('options', [])
                    if options:
                        current = results[fields[active_field]['name']]
                        try:
                            idx = options.index(current)
                            idx = (idx + 1) % len(options)
                            results[fields[active_field]['name']] = options[idx]
                        except ValueError:
                            results[fields[active_field]['name']] = options[0]
                
                elif fields[active_field].get('type') != 'select':
                    # Handle text input for active field
                    field = fields[active_field]
                    field_name = field['name']
                    current_value = str(results[field_name])
                    
                    if key == ord('\b') or key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                        # Delete last character
                        if current_value:
                            results[field_name] = current_value[:-1]
                    
                    elif 32 <= key <= 126:  # Printable characters
                        # Add character to field
                        char = chr(key)
                        
                        # For number fields, only allow digits and decimal point
                        if field.get('type') == 'number':
                            if char.isdigit() or (char == '.' and '.' not in current_value):
                                results[field_name] = current_value + char
                        else:
                            results[field_name] = current_value + char
        
        finally:
            # Restore screen
            self.stdscr.clear()
            screen_backup.overlay(self.stdscr)
            self.stdscr.refresh()
            
        return {}
    
    def show_progress(
        self, 
        title: str, 
        message: str, 
        progress: float, 
        can_cancel: bool = False
    ) -> bool:
        """
        Tampilkan dialog progress.
        
        Args:
            title: Judul dialog
            message: Pesan progress
            progress: Nilai progress (0.0 - 1.0)
            can_cancel: Apakah user dapat membatalkan
            
        Returns:
            False jika dibatalkan, True jika tidak
        """
        # Pastikan progress dalam range yang benar
        progress = max(0.0, min(1.0, progress))
        
        # Progress bar width (adjusted for dialog width)
        progress_width = max(20, self.display_width - 10)
        
        # Format the message with progress
        message_lines = [message]
        
        # Calculate dialog size
        self._calculate_size(title, message_lines)
        
        # Draw dialog
        self._draw_dialog_border(title, self.TYPE_PROGRESS)
        
        # Draw message
        msg_y = self.y + 2
        msg_x = self.x + (self.display_width - len(message)) // 2
        self.safe_addstr(msg_y, msg_x, message)
        
        # Draw progress bar
        bar_y = self.y + 4
        bar_x = self.x + (self.display_width - progress_width) // 2
        
        # Draw empty bar
        self.safe_addstr(bar_y, bar_x, "╞" + "═" * progress_width + "╡")
        
        # Fill bar according to progress
        filled_width = int(progress_width * progress)
        self.safe_addstr(bar_y, bar_x + 1, "█" * filled_width, self.COLOR_SUCCESS)
        
        # Draw percentage
        percent = int(progress * 100)
        percent_text = f"{percent}%"
        percent_x = self.x + (self.display_width - len(percent_text)) // 2
        self.safe_addstr(bar_y + 1, percent_x, percent_text)
        
        # Draw cancel option if applicable
        if can_cancel:
            cancel_text = "Tekan 'C' untuk batal"
            cancel_x = self.x + (self.display_width - len(cancel_text)) // 2
            self.safe_addstr(
                self.y + self.display_height - 2, 
                cancel_x, 
                cancel_text, 
                self.COLOR_NAVIGATION
            )
            
            # Check for cancel
            self.stdscr.nodelay(True)  # Non-blocking input
            key = self.stdscr.getch()
            self.stdscr.nodelay(False)  # Reset to blocking
            
            if key in [ord('c'), ord('C')]:
                return False
        
        # Refresh to show progress
        self.stdscr.refresh()
        return True
    
    def get_input(
        self, 
        prompt: str, 
        validator: Optional[Callable[[str], bool]] = None,
        default: Optional[str] = None,
        password: bool = False
    ) -> Optional[str]:
        """
        Tampilkan prompt dan dapatkan input pengguna dengan validasi.
        
        Args:
            prompt: Prompt untuk ditampilkan
            validator: Fungsi validasi (opsional)
            default: Nilai default (opsional)
            password: Jika True, karakter input akan disembunyikan
            
        Returns:
            String input atau None jika dibatalkan
        """
        # Create minimal dialog for input
        height = 7  # Minimal height for input dialog
        width = max(60, len(prompt) + 20)  # Ensure enough width for prompt
        
        # Set size manually and center
        self.set_size(width, height)
        self.center_dialog()
        
        # Save current screen
        screen_backup = curses.newwin(self.height, self.width, 0, 0)
        screen_backup.overlay(self.stdscr)
        
        # Draw dialog
        self._draw_dialog_border("Input", self.TYPE_FORM)
        
        # Draw prompt
        prompt_y = self.y + 2
        prompt_x = self.x + 2
        self.safe_addstr(prompt_y, prompt_x, prompt)
        
        # Draw default value hint if available
        if default:
            default_text = f"[Default: {default}]"
            self.safe_addstr(
                prompt_y, 
                prompt_x + len(prompt) + 1, 
                default_text, 
                self.COLOR_INFO
            )
        
        # Draw input area
        input_y = self.y + 3
        input_x = self.x + 2
        input_width = self.display_width - 4
        self.safe_addstr(input_y, input_x, "_" * input_width, self.COLOR_WARNING)
        
        # Draw help text
        help_y = self.y + 5
        help_text = "Enter: Konfirmasi | Esc: Batal"
        help_x = self.x + (self.display_width - len(help_text)) // 2
        self.safe_addstr(help_y, help_x, help_text, self.COLOR_NAVIGATION)
        
        # Get input with cursor
        try:
            curses.curs_set(1)  # Show cursor
            curses.echo()  # Echo input
            
            # Setup for input
            self.stdscr.move(input_y, input_x)  # Position cursor
            
            # Input buffer
            input_buffer = ""
            cursor_pos = 0
            
            # Clear input line
            self.safe_addstr(input_y, input_x, " " * input_width)
            self.stdscr.move(input_y, input_x)
            
            # Input loop
            while True:
                key = self.stdscr.getch()
                
                if key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:  # Enter - submit
                    # If empty and default exists, use default
                    if not input_buffer and default is not None:
                        input_buffer = default
                    
                    # Validate
                    if validator and not validator(input_buffer):
                        # Show validation error at the bottom
                        error_y = self.y + self.display_height - 2
                        self.safe_addstr(
                            error_y, 
                            self.x + 2, 
                            "⚠️ Input tidak valid                                 ", 
                            self.COLOR_ERROR
                        )
                        self.stdscr.refresh()
                        time.sleep(1)  # Show error briefly
                        
                        # Clear error and continue
                        self.safe_addstr(error_y, self.x + 2, " " * (self.display_width - 4))
                        continue
                    
                    break  # Valid input, exit loop
                    
                elif key == 27:  # Esc - cancel
                    curses.curs_set(0)  # Hide cursor
                    curses.noecho()  # Stop echo
                    
                    # Restore screen
                    self.stdscr.clear()
                    screen_backup.overlay(self.stdscr)
                    self.stdscr.refresh()
                    
                    return None  # Return None to indicate cancellation
                    
                elif key == ord('\b') or key == curses.KEY_BACKSPACE or key == 127:  # Backspace
                    if cursor_pos > 0:
                        # Remove character at cursor_pos - 1
                        input_buffer = input_buffer[:cursor_pos-1] + input_buffer[cursor_pos:]
                        cursor_pos -= 1
                        
                        # Redraw input
                        self.safe_addstr(input_y, input_x, " " * input_width)
                        display_text = "*" * len(input_buffer) if password else input_buffer
                        self.safe_addstr(input_y, input_x, display_text)
                        self.stdscr.move(input_y, input_x + cursor_pos)
                
                elif key == curses.KEY_LEFT and cursor_pos > 0:  # Left arrow
                    cursor_pos -= 1
                    self.stdscr.move(input_y, input_x + cursor_pos)
                    
                elif key == curses.KEY_RIGHT and cursor_pos < len(input_buffer):  # Right arrow
                    cursor_pos += 1
                    self.stdscr.move(input_y, input_x + cursor_pos)
                    
                elif key == curses.KEY_HOME:  # Home
                    cursor_pos = 0
                    self.stdscr.move(input_y, input_x)
                    
                elif key == curses.KEY_END:  # End
                    cursor_pos = len(input_buffer)
                    self.stdscr.move(input_y, input_x + cursor_pos)
                    
                elif 32 <= key <= 126:  # Printable characters
                    # Insert character at cursor position
                    char = chr(key)
                    input_buffer = input_buffer[:cursor_pos] + char + input_buffer[cursor_pos:]
                    cursor_pos += 1
                    
                    # Redraw input with appropriate masking
                    self.safe_addstr(input_y, input_x, " " * input_width)
                    display_text = "*" * len(input_buffer) if password else input_buffer
                    self.safe_addstr(input_y, input_x, display_text)
                    self.stdscr.move(input_y, input_x + cursor_pos)
                
                # Refresh after each keypress
                self.stdscr.refresh()
        
        finally:
            # Reset terminal state
            try:
                curses.curs_set(0)  # Hide cursor
            except:
                pass
            curses.noecho()  # Stop echo
            
            # Restore screen
            self.stdscr.clear()
            screen_backup.overlay(self.stdscr)
            self.stdscr.refresh()
        
        return input_buffer