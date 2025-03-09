# File: smartcash/handlers/model/observers/base_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer dasar yang sederhana untuk monitoring model

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from smartcash.utils.logger import get_logger

class BaseObserver(ABC):
    """Observer dasar untuk monitoring model dan training."""
    
    def __init__(self, logger=None, name="observer"):
        """Inisialisasi observer."""
        self.logger = logger or get_logger(f"model.{name}")
        self.name = name
        self._callbacks = {}
    
    @abstractmethod
    def update(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Update dari subject (model/trainer)."""
        pass
    
    def on_training_start(self, **kwargs):
        """Callback untuk awal training."""
        self.update('training_start', kwargs)
        self._trigger_callbacks('training_start', kwargs)
    
    def on_training_end(self, **kwargs):
        """Callback untuk akhir training."""
        self.update('training_end', kwargs)
        self._trigger_callbacks('training_end', kwargs)
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """Callback untuk awal epoch."""
        self.update('epoch_start', {'epoch': epoch, **kwargs})
        self._trigger_callbacks('epoch_start', {'epoch': epoch, **kwargs})
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """Callback untuk akhir epoch."""
        self.update('epoch_end', {'epoch': epoch, 'metrics': metrics, **kwargs})
        self._trigger_callbacks('epoch_end', {'epoch': epoch, 'metrics': metrics, **kwargs})
    
    def _trigger_callbacks(self, event: str, data: Dict[str, Any]):
        """Trigger semua callback yang terdaftar untuk event."""
        if event in self._callbacks:
            for callback in self._callbacks[event]:
                callback(**data)
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """Daftarkan callback untuk event tertentu."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)