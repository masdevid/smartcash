# File: smartcash/handlers/dataset/explorers/base_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk explorer dataset SmartCash

from pathlib import Path
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.layer_config_manager import get_layer_config

class BaseExplorer(ABC):
    """
    Kelas dasar untuk eksplorasi dataset SmartCash.
    Mendefinisikan interface dan fungsionalitas dasar yang digunakan
    oleh explorer spesifik.
    """
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger: Optional[SmartCashLogger] = None):
        """Inisialisasi BaseExplorer."""
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or SmartCashLogger(__name__)
        
        # Dapatkan konfigurasi layer dari layer config manager
        self.layer_config_manager = get_layer_config()
        self.active_layers = config.get('layers', self.layer_config_manager.get_layer_names())
        
        # Class ID to layer mapping untuk lookup cepat
        self.class_to_layer = {}
        self.class_to_name = {}
        
        for layer_name in self.layer_config_manager.get_layer_names():
            layer_config = self.layer_config_manager.get_layer_config(layer_name)
            for i, cls_id in enumerate(layer_config['class_ids']):
                self.class_to_layer[cls_id] = layer_name
                if i < len(layer_config['classes']):
                    self.class_to_name[cls_id] = layer_config['classes'][i]
        
        self.logger.info(f"🔍 {self.__class__.__name__} diinisialisasi untuk: {self.data_dir}")
    
    def _get_split_path(self, split: str) -> Path:
        """Dapatkan path untuk split dataset tertentu."""
        # Handling untuk nama split yang berbeda
        if split in ('val', 'validation'):
            split = 'valid'
        return self.data_dir / split
    
    def _get_class_name(self, cls_id: int) -> str:
        """Dapatkan nama kelas berdasarkan ID kelas."""
        if cls_id in self.class_to_name:
            return self.class_to_name[cls_id]
            
        # Jika tidak ditemukan, coba cari di layer config
        for layer_name in self.layer_config_manager.get_layer_names():
            layer_config = self.layer_config_manager.get_layer_config(layer_name)
            if cls_id in layer_config['class_ids']:
                idx = layer_config['class_ids'].index(cls_id)
                if idx < len(layer_config['classes']):
                    return layer_config['classes'][idx]
        
        # Fallback jika tidak ditemukan
        return f"Class-{cls_id}"
    
    @abstractmethod
    def explore(self, split: str, **kwargs) -> Dict[str, Any]:
        """
        Metode utama untuk eksplorasi dataset.
        Harus diimplementasikan oleh subclass.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            **kwargs: Parameter tambahan
            
        Returns:
            Dict berisi hasil eksplorasi
        """
        pass