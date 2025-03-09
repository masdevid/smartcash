# File: smartcash/handlers/dataset/facades/dataset_base_facade.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk facade dataset yang mengelola komponen-komponen utama

from pathlib import Path
from typing import Dict, Optional, Any, Callable

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config


class DatasetBaseFacade:
    """
    Kelas dasar untuk facade dataset yang mengelola inisialisasi komponen dan konfigurasi.
    Menyediakan fitur registrasi dan akses komponen melalui lazy initialization.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetBaseFacade.
        
        Args:
            config: Konfigurasi dataset
            data_dir: Direktori dataset (opsional)
            cache_dir: Direktori cache (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or SmartCashLogger(__name__)
        
        # Setup paths
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.cache_dir = Path(cache_dir or config.get('data', {}).get('preprocessing', {}).get('cache_dir', '.cache/smartcash'))
        self.img_size = tuple(config.get('model', {}).get('img_size', [640, 640]))
        self.batch_size = config.get('model', {}).get('batch_size', 16)
        self.num_workers = config.get('model', {}).get('workers', 4)
        
        # Aktifkan layer sesuai konfigurasi
        self.layer_config = get_layer_config()
        self.active_layers = config.get('layers', ['banknote'])
        
        # Dictionary untuk menyimpan komponen yang di-lazy load
        self._components = {}
        
        self.logger.info(
            f"🔧 DatasetBaseFacade diinisialisasi:\n"
            f"   • Data dir: {self.data_dir}\n"
            f"   • Img size: {self.img_size}\n"
            f"   • Batch size: {self.batch_size}\n"
            f"   • Layers aktif: {self.active_layers}"
        )
    
    def _get_component(self, component_id: str, factory_func: Callable) -> Any:
        """
        Dapatkan komponen dengan lazy initialization.
        
        Args:
            component_id: ID unik untuk komponen
            factory_func: Fungsi factory untuk membuat komponen jika belum ada
            
        Returns:
            Komponen yang diminta
        """
        if component_id not in self._components:
            self._components[component_id] = factory_func()
        return self._components[component_id]
    
    def _get_split_path(self, split: str) -> Path:
        """
        Dapatkan path untuk split dataset.
        
        Args:
            split: Split dataset ('train', 'val', 'test')
            
        Returns:
            Path ke direktori split dataset
        """
        # Normalisasi nama split
        if split in ('val', 'validation'):
            split = 'valid'
            
        # Cek konfigurasi khusus untuk path split
        split_paths = self.config.get('data', {}).get('local', {})
        if split in split_paths:
            return Path(split_paths[split])
            
        # Fallback ke path default
        return self.data_dir / split