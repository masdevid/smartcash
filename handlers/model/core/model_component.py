# File: smartcash/handlers/model/core/model_component.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar yang disederhanakan untuk semua komponen model dengan lazy-loading bawaan

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Type, Callable
import inspect
import importlib
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger

class ModelComponent(ABC):
    """
    Kelas dasar untuk semua komponen model.
    Menyediakan fungsionalitas umum seperti logging, lazy-loading, dan error handling.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None,
        name: str = "model_component"
    ):
        """
        Inisialisasi model component.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom (opsional)
            name: Nama komponen untuk logging
        """
        self.config = config
        self.logger = logger or get_logger(f"model.{name}")
        self.name = name
        
        # Penyimpanan internal untuk komponen lazy-loaded
        self._lazy_components = {}
        
        # Deteksi environment
        self.in_colab = self._is_running_in_colab()
        
        # Inisialisasi internal komponen
        self._initialize()
    
    def _initialize(self) -> None:
        """
        Inisialisasi internal komponen.
        Override oleh subclass untuk inisialisasi kustom.
        """
        pass
    
    def _is_running_in_colab(self) -> bool:
        """Deteksi apakah kode berjalan di Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def get_component(self, component_id: str, factory_func: Callable) -> Any:
        """
        Dapatkan komponen dengan lazy initialization.
        
        Args:
            component_id: ID unik untuk komponen
            factory_func: Fungsi untuk membuat komponen jika belum ada
            
        Returns:
            Komponen yang diminta
        """
        if component_id not in self._lazy_components:
            self._lazy_components[component_id] = factory_func()
        return self._lazy_components[component_id]
    
    def lazy_import(self, module_path: str, class_name: str) -> Type:
        """
        Import kelas secara lazy.
        
        Args:
            module_path: Path ke modul (format: 'package.module')
            class_name: Nama kelas yang akan diimpor
            
        Returns:
            Kelas yang diimpor
        """
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            self.logger.error(f"❌ Gagal mengimpor {class_name} dari {module_path}: {str(e)}")
            raise
    
    def safe_execute(self, func: Callable, error_msg: str, *args, **kwargs) -> Any:
        """
        Eksekusi fungsi dengan error handling yang konsisten.
        
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
    
    def create_output_dir(self, subdir: Optional[str] = None) -> Path:
        """
        Buat dan dapatkan direktori output.
        
        Args:
            subdir: Subdirektori opsional (akan dibuat jika belum ada)
            
        Returns:
            Path ke direktori output
        """
        base_dir = Path(self.config.get('output_dir', f'runs/{self.name}'))
        output_dir = base_dir / subdir if subdir else base_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Proses utama komponen.
        
        Args:
            *args: Argumen untuk proses
            **kwargs: Keyword argumen untuk proses
            
        Returns:
            Hasil proses
        """
        pass