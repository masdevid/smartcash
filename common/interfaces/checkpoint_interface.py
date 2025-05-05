# smartcash/common/interfaces/checkpoint_interface.py
"""
Interface untuk checkpoint yang digunakan oleh domain Model
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class ICheckpointService(ABC):
    @abstractmethod
    def save_checkpoint(self, model: Any, path: str, optimizer: Optional[Any] = None, 
                       epoch: int = 0, metadata: Optional[Dict[str, Any]] = None, 
                       is_best: bool = False) -> str:
        """Simpan checkpoint model"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: str, model: Any, 
                      optimizer: Optional[Any] = None, 
                      map_location: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint model"""
        pass