# File: smartcash/handlers/base_handler.py
# Author: Alfrida Sabar
# Deskripsi: Base handler dengan fungsionalitas umum untuk semua handler

import time
from typing import Dict, Optional, Any, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.logging_factory import LoggingFactory
from smartcash.utils.config_manager import get_config_manager
from smartcash.exceptions.handler import ErrorHandler
from smartcash.exceptions.factory import ErrorFactory

class BaseHandler:
    """
    Base class untuk semua handler SmartCash.
    
    Fitur:
    - Integrasi dengan sistem logging
    - Integrasi dengan konfigurasi
    - Error handling konsisten
    - Metode utilitas umum
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[SmartCashLogger] = None,
        handler_name: Optional[str] = None
    ):
        """
        Inisialisasi base handler.
        
        Args:
            config_path: Path ke file konfigurasi (opsional)
            config: Dictionary konfigurasi (opsional)
            logger: Custom logger (opsional)
            handler_name: Nama handler untuk logging (opsional)
        """
        # Setup nama handler
        self.handler_name = handler_name or self.__class__.__name__
        
        # Setup logging
        self.logger = logger or LoggingFactory.get_logger(self.handler_name)
        
        # Setup error handler
        self.error_handler = ErrorHandler(self.handler_name)
        
        # Setup konfigurasi
        self.config_manager = get_config_manager(config_path)
        
        # Update konfigurasi jika disediakan
        if config:
            for key, value in config.items():
                self.config_manager.set(key, value)
        
        # Konfigurasi aktif
        self.config = self.config_manager.get_config()
        
        # Track waktu eksekusi
        self.execution_times = {}
        
    def run(self, operation: str, **kwargs):
        """
        Metode wrapper untuk menjalankan operasi dengan error handling.
        
        Args:
            operation: Nama operasi yang akan dijalankan
            **kwargs: Parameter operasi
            
        Returns:
            Hasil operasi
            
        Raises:
            SmartCashError: Jika operasi gagal
        """
        self.logger.info(f"🚀 Menjalankan operasi: {operation}")
        start_time = time.time()
        
        try:
            # Cek apakah operasi tersedia
            if not hasattr(self, operation) or not callable(getattr(self, operation)):
                raise AttributeError(f"Operasi tidak tersedia: {operation}")
                
            # Jalankan operasi
            result = getattr(self, operation)(**kwargs)
            
            # Simpan waktu eksekusi
            execution_time = time.time() - start_time
            self.execution_times[operation] = execution_time
            
            self.logger.info(f"✅ Operasi {operation} selesai dalam {execution_time:.2f}s")
            return result
            
        except Exception as e:
            # Tangani error
            execution_time = time.time() - start_time
            self.execution_times[operation] = execution_time
            
            self.logger.error(f"❌ Operasi {operation} gagal: {str(e)}")
            
            # Konversi ke SmartCashError
            if not hasattr(e, 'message'):  # Jika bukan SmartCashError
                e = ErrorFactory.from_exception(
                    e, 
                    additional_message=f"Operasi {operation} gagal"
                )
            
            # Re-raise untuk dihandle di level lebih tinggi
            raise e
    
    def get_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi saat ini.
        
        Returns:
            Dictionary konfigurasi
        """
        return self.config_manager.get_config()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """
        Update konfigurasi.
        
        Args:
            config: Dictionary konfigurasi untuk update
        """
        for key, value in config.items():
            self.config_manager.set(key, value)
            
        # Update referensi konfigurasi
        self.config = self.config_manager.get_config()
        
    def validate_config(self) -> bool:
        """
        Validasi konfigurasi.
        
        Returns:
            True jika valid
            
        Raises:
            ConfigError: Jika konfigurasi tidak valid
        """
        return self.config_manager.validate()
    
    def get_execution_time(self, operation: str) -> float:
        """
        Dapatkan waktu eksekusi operasi.
        
        Args:
            operation: Nama operasi
            
        Returns:
            Waktu eksekusi dalam detik
        """
        return self.execution_times.get(operation, 0)
    
    def get_all_execution_times(self) -> Dict[str, float]:
        """
        Dapatkan semua waktu eksekusi.
        
        Returns:
            Dictionary waktu eksekusi
        """
        return self.execution_times.copy()