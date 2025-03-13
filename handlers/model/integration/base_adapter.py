# File: smartcash/handlers/model/integration/base_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk semua adapter integrasi

from typing import Dict, Optional, Any
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger

class BaseAdapter:
    """
    Kelas dasar untuk semua adapter integrasi.
    Menerapkan pola DRY untuk fungsionalitas umum adapter.
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
        self.config = config
        self.logger = logger or get_logger(f"model.{adapter_name}")
        self.adapter_name = adapter_name
        
        # Default output directory
        self.output_dir = self._get_default_output_dir()
        
        # Deteksi environment
        self.in_colab = self._is_running_in_colab()
        
        # Inisialisasi custom
        self._initialize()
    
    def _initialize(self) -> None:
        """
        Hook untuk inisialisasi custom di subclass.
        Override metode ini di subclass untuk inisialisasi khusus.
        """
        pass
    
    def _get_default_output_dir(self) -> Path:
        """
        Dapatkan direktori output default dari config.
        
        Returns:
            Path direktori output
        """
        # Baca dari config berdasarkan adapter_name
        adapter_type = self.adapter_name.replace('_adapter', '')
        default_path = f"runs/{adapter_type}"
        
        # Coba dapatkan dari config
        output_path = self.config.get(adapter_type, {}).get('output_dir', default_path)
        output_path = self.config.get('output_dir', output_path)
        
        return Path(output_path)
    
    def _is_running_in_colab(self) -> bool:
        """
        Deteksi apakah kode berjalan di Google Colab.
        
        Returns:
            True jika di Colab, False jika tidak
        """
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _create_output_dir(self, subdir: Optional[str] = None) -> Path:
        """
        Buat direktori output dan pastikan ada.
        
        Args:
            subdir: Subdirektori opsional untuk dibuat
            
        Returns:
            Path ke direktori output
        """
        if subdir:
            output_dir = self.output_dir / subdir
        else:
            output_dir = self.output_dir
            
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.adapter_name} (output_dir={self.output_dir})"