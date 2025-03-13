# File: smartcash/handlers/model/integration/base_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk semua adapter integrasi, direfaktor untuk konsistensi dan DRY

from typing import Dict, Optional, Any, Callable
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.model.core.model_component import ModelComponent

class BaseAdapter(ModelComponent):
    """
    Kelas dasar untuk semua adapter integrasi.
    Meng-extends ModelComponent dengan fungsionalitas adapter-specific.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        adapter_name: str = "base_adapter"
    ):
        """
        Inisialisasi base adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
            adapter_name: Nama adapter untuk logging
        """
        # Normalisasi adapter_name
        name = adapter_name.replace('_adapter', '')
        super().__init__(config, logger, f"adapter.{name}")
        
        # Simpan nama untuk kebutuhan konfigurasi
        self.adapter_type = name
    
    def _initialize(self) -> None:
        """Setup output directory berdasarkan tipe adapter"""
        # Baca dari config berdasarkan adapter_type
        default_path = f"runs/{self.adapter_type}"
        
        # Coba dapatkan dari config dengan fallback ke default
        output_path = self.config.get(self.adapter_type, {}).get('output_dir', 
                      self.config.get('output_dir', default_path))
        
        self.output_dir = Path(output_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging inisialisasi
        self.logger.info(f"🔄 {self.__class__.__name__} diinisialisasi (output: {self.output_dir})")
    
    def process(self, *args, **kwargs) -> Any:
        """
        Implementasi default process dari ModelComponent.
        Subclass harus mengimplementasikan method ini atau override-nya.
        
        Args:
            *args, **kwargs: Argumen untuk proses
            
        Returns:
            Hasil proses
        """
        self.logger.warning(f"⚠️ Method process belum diimplementasi di {self.__class__.__name__}")
        return None
    
    def with_error_handling(self, func: Callable, error_msg: str = None) -> Callable:
        """
        Decorator untuk menambahkan error handling ke method.
        
        Args:
            func: Fungsi yang akan ditambahkan error handling
            error_msg: Pesan error kustom (opsional)
            
        Returns:
            Fungsi dengan error handling
        """
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = error_msg or f"Gagal menjalankan {func.__name__}"
                self.logger.error(f"❌ {msg}: {str(e)}")
                raise
        return wrapper