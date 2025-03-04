# File: smartcash/exceptions/handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk manajemen error di SmartCash

import sys
import logging
import logging.handlers
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Type, Callable
from smartcash.utils.logger import get_logger

from .base import (ConfigError, DataError, ModelError,
    TrainingError, EvaluationError, PreprocessingError,
    ValidationError, ResourceError
)

class ErrorHandler:
    """Handler untuk manajemen error global SmartCash."""
    
    # Mapping tipe error ke kategori untuk pesan user
    ERROR_CATEGORIES: Dict[Type[Exception], str] = {
        ConfigError: "Kesalahan Konfigurasi",
        DataError: "Kesalahan Data",
        ModelError: "Kesalahan Model",
        TrainingError: "Kesalahan Training",
        EvaluationError: "Kesalahan Evaluasi",
        PreprocessingError: "Kesalahan Preprocessing",
        ValidationError: "Kesalahan Validasi",
        ResourceError: "Kesalahan Resource",
        PermissionError: "Kesalahan Izin",
        FileNotFoundError: "File Tidak Ditemukan",
        KeyboardInterrupt: "Aplikasi Dihentikan",
        Exception: "Kesalahan Sistem"  # Fallback
    }

    def __init__(self, name: str = "smartcash.error"):
        """
        Initialize error handler.
        
        Args:
            name: Nama untuk logger error handler
        """
        self.logger = get_logger(name)
        self.setup_file_logging()
    
    def setup_file_logging(self) -> None:
        """Setup logging ke file dengan rotasi harian."""
        # Buat direktori logs jika belum ada
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        # Setup rotating file handler
        log_file = logs_dir / "smartcash.log"
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )

        # Format log dengan timestamp dan level
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Tambahkan handler ke logger
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.INFO)
    
    def handle(
        self, 
        error: Exception,
        cleanup_func: Optional[Callable] = None,
        exit_on_error: bool = True
    ) -> None:
        """
        Handle error dengan logging dan feedback.
        
        Args:
            error: Exception yang terjadi
            cleanup_func: Optional function untuk cleanup
            exit_on_error: Apakah exit program setelah error
        """
        try:
            # Tentukan kategori error
            error_type = type(error)
            error_category = self.ERROR_CATEGORIES.get(
                error_type,
                self.ERROR_CATEGORIES[Exception]
            )

            # Log error dengan stack trace
            self.logger.error(
                f"❌ {error_category}: {str(error)}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )

            # Cleanup jika ada
            if cleanup_func:
                try:
                    cleanup_func()
                except Exception as cleanup_error:
                    self.logger.error(
                        f"❌ Error saat cleanup: {str(cleanup_error)}"
                    )

            # Format pesan error untuk user
            error_msg = (
                f"\n❌ {error_category}\n"
                f"📝 Detail: {str(error)}\n"
                f"⏰ Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📁 Log tersimpan di: logs/smartcash.log\n"
            )

            # Handle special case untuk KeyboardInterrupt
            if isinstance(error, KeyboardInterrupt):
                print("\n👋 Program dihentikan oleh user.")
                sys.exit(0)
            
            # Tampilkan error ke user
            print(error_msg)
            
            # Exit jika diminta
            if exit_on_error:
                sys.exit(1)

        except Exception as handler_error:
            # Fatal error dalam error handler
            self.logger.critical(
                f"❌ Fatal error dalam error handler: {str(handler_error)}\n"
                f"Original error: {str(error)}\n"
                f"Stack trace:\n{traceback.format_exc()}"
            )
            sys.exit(1)
    
    def format_error(self, error: Exception) -> str:
        """
        Format error message untuk display.
        
        Args:
            error: Exception untuk diformat
            
        Returns:
            String pesan error yang terformat
        """
        error_type = type(error)
        error_category = self.ERROR_CATEGORIES.get(
            error_type,
            self.ERROR_CATEGORIES[Exception]
        )
        
        return (
            f"{error_category}\n"
            f"Detail: {str(error)}"
        )