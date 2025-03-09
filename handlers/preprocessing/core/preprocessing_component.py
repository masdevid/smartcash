"""
File: smartcash/handlers/preprocessing/core/preprocessing_component.py
Author: Alfrida Sabar
Deskripsi: Komponen dasar untuk semua komponen preprocessing yang digunakan dalam pipeline preprocessing
           SmartCash. Menyediakan struktur dasar dan fungsionalitas umum.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger


class PreprocessingComponent(ABC):
    """
    Kelas abstrak dasar untuk semua komponen preprocessing.
    Menyediakan interface standar untuk semua komponen preprocessing.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        logger: Optional[SmartCashLogger] = None,
        **kwargs
    ):
        """
        Inisialisasi komponen preprocessing.
        
        Args:
            config: Konfigurasi untuk komponen
            logger: Logger kustom (opsional)
            **kwargs: Parameter tambahan
        """
        self.config = config
        self.logger = logger or get_logger(self.__class__.__name__)
        self.result = {}
        self.start_time = None
        
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Mengeksekusi komponen preprocessing dengan timing dan logging.
        
        Args:
            **kwargs: Parameter untuk proses
            
        Returns:
            Dict[str, Any]: Hasil dari proses
        """
        self.start_time = time.time()
        
        # Log dimulainya proses
        self.logger.start(f"🚀 Memulai proses {self.__class__.__name__}")
        
        try:
            # Jalankan implementasi proses yang sebenarnya
            self.result = self.process(**kwargs)
            
            # Tambahkan metadata timing
            elapsed = time.time() - self.start_time
            self.result['metadata'] = {
                'elapsed_time': elapsed,
                'component': self.__class__.__name__,
                'parameters': kwargs
            }
            
            # Log selesainya proses
            self.logger.success(
                f"✅ Proses {self.__class__.__name__} selesai dalam {elapsed:.2f} detik"
            )
            
            return self.result
            
        except Exception as e:
            # Log error
            self.logger.error(f"❌ Error dalam {self.__class__.__name__}: {str(e)}")
            # Re-raise exception
            raise
    
    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Any]:
        """
        Proses utama yang harus diimplementasikan oleh subclass.
        
        Args:
            **kwargs: Parameter untuk proses
            
        Returns:
            Dict[str, Any]: Hasil dari proses
        """
        pass
    
    def get_path(self, path_key: str, default: Optional[str] = None) -> Path:
        """
        Helper untuk mendapatkan path dari konfigurasi dan memastikan path tersebut ada.
        
        Args:
            path_key: Kunci path dalam konfigurasi
            default: Path default jika tidak ditemukan dalam konfigurasi
            
        Returns:
            Path: Path yang sudah divalidasi
        """
        # Coba ambil dari config
        if path_key in self.config:
            path = Path(self.config[path_key])
        elif default:
            path = Path(default)
        else:
            raise ValueError(f"Path untuk '{path_key}' tidak ditemukan dalam konfigurasi dan tidak ada default")
        
        # Buat direktori jika belum ada
        if not path.exists() and not path.is_file():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"📁 Membuat direktori baru: {path}")
            
        return path
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Helper untuk mendapatkan nilai dari hierarki konfigurasi.
        Mendukung dot notation (misalnya 'data.preprocessing.cache_dir').
        
        Args:
            key: Kunci konfigurasi, dapat menggunakan dot notation
            default: Nilai default jika kunci tidak ditemukan
            
        Returns:
            Any: Nilai dari konfigurasi
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default