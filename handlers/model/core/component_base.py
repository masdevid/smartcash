# File: smartcash/handlers/model/core/component_base.py
# Deskripsi: Kelas dasar untuk semua komponen model dengan dependency injection

from typing import Dict, Optional, Any, Type, Callable
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger

class ComponentBase:
    """Kelas dasar untuk semua komponen model dengan dependency injection bawaan."""
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "component"
    ):
        """
        Inisialisasi component base.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
            name: Nama komponen untuk logging
        """
        self.config = config
        self.logger = logger or get_logger(f"model.{name}")
        self.name = name
        
        # Deteksi environment
        self.in_colab = self._is_colab()
        
        # Inisialisasi komponen
        self._initialize()
    
    def _initialize(self) -> None:
        """Inisialisasi internal komponen, di-override oleh subclass."""
        pass
    
    def _is_colab(self) -> bool:
        """Deteksi apakah kode berjalan di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def create_output_dir(self, subdir: Optional[str] = None) -> Path:
        """
        Buat dan dapatkan direktori output.
        
        Args:
            subdir: Subdirektori opsional
            
        Returns:
            Path ke direktori output
        """
        base_dir = Path(self.config.get('output_dir', f'runs/{self.name}'))
        output_dir = base_dir / subdir if subdir else base_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def safe_execute(self, func: Callable, error_msg: str, *args, **kwargs) -> Any:
        """
        Eksekusi fungsi dengan error handling.
        
        Args:
            func: Fungsi yang akan dieksekusi
            error_msg: Pesan error jika gagal
            *args, **kwargs: Argumen untuk fungsi
            
        Returns:
            Hasil eksekusi fungsi
            
        Raises:
            Exception: Jika fungsi gagal dieksekusi
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"❌ {error_msg}: {str(e)}")
            raise