# smartcash/common/interfaces/layer_config_interface.py
"""
Interface untuk konfigurasi layer yang digunakan oleh domain Dataset
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

class ILayerConfigManager(ABC):
    @abstractmethod
    def get_layer_config(self, layer_name: str) -> Dict[str, Any]:
        """Mendapatkan konfigurasi untuk layer tertentu"""
        pass
    
    @abstractmethod
    def get_class_map(self) -> Dict[int, str]:
        """Mendapatkan mapping class_id ke class_name"""
        pass
    
    @abstractmethod
    def get_layer_for_class_id(self, class_id: int) -> str:
        """Mendapatkan nama layer untuk class_id tertentu"""
        pass