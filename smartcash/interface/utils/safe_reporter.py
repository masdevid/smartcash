# File: smartcash/interface/utils/safe_reporter.py
# Author: Alfrida Sabar
# Deskripsi: Reporter progress yang aman untuk berbagai antarmuka dengan dukungan terminal emulation

from typing import Optional, Dict, List
from smartcash.interface.utils.display import DisplayManager
from smartcash.utils.logger import SmartCashLogger

class SafeProgressReporter:
    """Progress reporter yang aman untuk berbagai antarmuka dengan tampilan terminal-like."""
    
    def __init__(self, display_manager: Optional[DisplayManager] = None):
        """
        Inisialisasi reporter.
        
        Args:
            display_manager: Optional display manager
        """
        self.display_manager = display_manager
        self.logger = SmartCashLogger(__name__)
        self.log_buffer = []  # Buffer untuk menyimpan pesan log
        self.max_buffer_size = 1000  # Batas maksimum ukuran buffer
    
    def _truncate_message(self, message: str, max_length: int = 80) -> str:
        """
        Potong pesan jika terlalu panjang.
        
        Args:
            message: Pesan asli
            max_length: Panjang maksimal pesan
        
        Returns:
            Pesan yang dipotong
        """
        return message[:max_length] + '...' if len(message) > max_length else message
    
    def show_progress(
        self, 
        message: str, 
        current: int, 
        total: int
    ):
        """
        Tampilkan progress dengan aman.
        
        Args:
            message: Pesan progress
            current: Posisi saat ini
            total: Total langkah
        """
        try:
            # Singkat pesan
            safe_message = self._truncate_message(message)
            
            # Format progress bar
            progress = min(1.0, max(0.0, current / total))
            bar_width = 20
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
            percent = int(progress * 100)
            
            progress_msg = f"{safe_message} [{bar}] {percent}% ({current}/{total})"
            
            # Tambahkan ke buffer log
            self._add_to_log_buffer(progress_msg)
            
            # Gunakan display manager jika tersedia
            if self.display_manager:
                try:
                    # Pastikan current dan total valid
                    current = max(0, min(current, total))
                    self.display_manager.show_progress(
                        message=safe_message, 
                        current=current, 
                        total=total
                    )
                except Exception as display_err:
                    self.logger.warning(f"⚠️ Error tampilan progress: {display_err}")
            else:
                # Fallback ke print biasa
                print(progress_msg)
        
        except Exception as e:
            # Tangani error apapun dengan mencetak sederhana
            print(f"Progress error: {e}")
    
    def show_dialog(
        self, 
        title: str, 
        message: str, 
        options: Dict[str, str] = None
    ) -> Optional[str]:
        """
        Tampilkan dialog dengan aman.
        
        Args:
            title: Judul dialog
            message: Isi pesan
            options: Opsi dialog
        
        Returns:
            Pilihan pengguna atau None
        """
        try:
            # Potong judul dan pesan
            safe_title = self._truncate_message(title, 30)
            safe_message = self._truncate_message(message, 500)
            
            # Default options jika tidak disediakan
            if options is None:
                options = {"o": "OK"}
                
            # Format pesan dialog dan tambahkan ke log buffer
            dialog_msg = f"[{safe_title}] {safe_message}"
            self._add_to_log_buffer(dialog_msg)
            
            # Tambahkan opsi ke log buffer
            options_str = " | ".join([f"{key}: {val}" for key, val in options.items()])
            self._add_to_log_buffer(f"Opsi: {options_str}")
            
            # Gunakan display manager jika tersedia
            if self.display_manager:
                try:
                    return self.display_manager.show_dialog(
                        title=safe_title, 
                        message=safe_message,
                        options=options
                    )
                except Exception as e:
                    self.logger.warning(f"⚠️ Error tampilan dialog: {e}")
                    # Fallback ke print
                    print(f"{safe_title}: {safe_message}")
                    print(f"Opsi: {options_str}")
                    return 'o'
            else:
                # Fallback ke print
                print(f"{safe_title}: {safe_message}")
                print(f"Opsi: {options_str}")
                return 'o'
        
        except Exception as e:
            print(f"Dialog error: {e}")
            return None
            
    def _add_to_log_buffer(self, message: str):
        """
        Tambahkan pesan ke buffer log.
        
        Args:
            message: Pesan untuk ditambahkan
        """
        self.log_buffer.append(message)
        
        # Batasi ukuran buffer
        if len(self.log_buffer) > self.max_buffer_size:
            self.log_buffer = self.log_buffer[-self.max_buffer_size:]
    
    def start_terminal_output(
        self, 
        title: str = "Proses Training", 
        wait_for_key: bool = True
    ) -> None:
        """
        Mulai menampilkan output terminal-like.
        
        Args:
            title: Judul terminal
            wait_for_key: Tunggu keypress di akhir
        """
        # Tampilkan buffer log di terminal jika display manager tersedia
        if self.display_manager:
            try:
                self.display_manager.show_terminal(
                    title=title,
                    lines=self.log_buffer,
                    wait_for_key=wait_for_key
                )
            except Exception as e:
                self.logger.warning(f"⚠️ Error menampilkan terminal: {e}")
                # Fallback - tampilkan beberapa pesan terakhir
                print(f"\n--- {title} ---")
                for msg in self.log_buffer[-20:]:  # Tampilkan 20 pesan terakhir
                    print(msg)
                
    def log_message(self, message: str, type_prefix: str = "ℹ️"):
        """
        Tambahkan pesan log ke buffer dan tampilkan jika memungkinkan.
        
        Args:
            message: Pesan yang akan ditampilkan
            type_prefix: Prefix yang menunjukkan tipe pesan
        """
        formatted_message = f"{type_prefix} {message}"
        self._add_to_log_buffer(formatted_message)
        
        # Jika display manager tersedia, tampilkan pesan
        if self.display_manager:
            prefix_to_color = {
                "✅": 2,  # Success - Green
                "❌": 1,  # Error - Red
                "⚠️": 3,  # Warning - Yellow
                "ℹ️": 4,  # Info - Cyan
                "🔍": 4,  # Info/detail - Cyan
                "🚀": 5,  # Special/launch - Magenta
                "📊": 6   # Data/metric - Blue
            }
            
            color = prefix_to_color.get(type_prefix, 0)
            if type_prefix == "✅":
                self.display_manager.show_success(message)
            elif type_prefix == "❌":
                self.display_manager.show_error(message, timeout_ms=0)
            else:
                # Tambahkan ke buffer log pada display manager
                try:
                    # Gunakan metode _add_to_log_buffer jika tersedia
                    if hasattr(self.display_manager, '_add_to_log_buffer'):
                        self.display_manager._add_to_log_buffer(formatted_message, color)
                        self.display_manager._draw_log_area()
                except Exception:
                    # Ignore errors
                    pass
        else:
            # Fallback ke print
            print(formatted_message)
            
    def log_success(self, message: str):
        """Log pesan sukses."""
        self.log_message(message, "✅")
        
    def log_error(self, message: str):
        """Log pesan error."""
        self.log_message(message, "❌")
        
    def log_warning(self, message: str):
        """Log pesan warning."""
        self.log_message(message, "⚠️")
        
    def log_info(self, message: str):
        """Log pesan informasi."""
        self.log_message(message, "ℹ️")
        
    def log_metric(self, message: str):
        """Log metrik."""
        self.log_message(message, "📊")