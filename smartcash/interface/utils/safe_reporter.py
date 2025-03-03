
from typing import Optional, Dict
from smartcash.interface.utils.display import DisplayManager
from smartcash.utils.logger import SmartCashLogger

class SafeProgressReporter:
    """Progress reporter yang aman untuk berbagai antarmuka."""
    logger = SmartCashLogger(__name__)
    
    def __init__(self, display_manager: Optional[DisplayManager] = None):
        """
        Inisialisasi reporter.
        
        
        Args:
            display_manager: Optional display manager
        """
        self.display_manager = display_manager
    
    def _truncate_message(self, message: str, max_length: int = 50) -> str:
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
                    # Fallback ke print jika display manager gagal
                    print(f"{safe_message}: {current}/{total}")
            else:
                # Fallback ke print biasa
                print(f"{safe_message}: {current}/{total}")
        
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
            safe_message = self._truncate_message(message, 100)
            
            # Gunakan display manager jika tersedia
            if self.display_manager:
                try:
                    return self.display_manager.show_dialog(
                        title=safe_title, 
                        message=safe_message,
                        options=options or {"o": "OK"}
                    )
                except Exception:
                    # Fallback ke print
                    print(f"{safe_title}: {safe_message}")
                    return 'o'
            else:
                # Fallback ke print
                print(f"{safe_title}: {safe_message}")
                return 'o'
        
        except Exception as e:
            print(f"Dialog error: {e}")
            return None
