# File: smartcash/handlers/model/observers/base_observer.py
# Author: Alfrida Sabar
# Deskripsi: Kelas dasar untuk semua observer model

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable

from smartcash.utils.logger import get_logger, SmartCashLogger

class BaseObserver(ABC):
    """
    Observer dasar untuk monitoring model dan training.
    Implementasi observer pattern untuk mendapatkan update.
    """
    
    def __init__(
        self, 
        logger: Optional[SmartCashLogger] = None,
        name: str = "observer"
    ):
        """
        Inisialisasi observer.
        
        Args:
            logger: Custom logger (opsional)
            name: Nama observer untuk logger
        """
        self.logger = logger or get_logger(f"model.{name}")
        self.name = name
        
        # Callback internal
        self._callbacks = {}
    
    @abstractmethod
    def update(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update dari subject (model/trainer/dll).
        
        Args:
            event: Nama event
            data: Data tambahan (opsional)
        """
        pass
    
    def on_training_start(self, **kwargs) -> None:
        """
        Callback untuk awal training.
        
        Args:
            **kwargs: Parameter tambahan
        """
        self.update('training_start', kwargs)
        
        # Panggil callback jika ada
        if 'training_start' in self._callbacks:
            for callback in self._callbacks['training_start']:
                callback(**kwargs)
    
    def on_training_end(self, **kwargs) -> None:
        """
        Callback untuk akhir training.
        
        Args:
            **kwargs: Parameter tambahan
        """
        self.update('training_end', kwargs)
        
        # Panggil callback jika ada
        if 'training_end' in self._callbacks:
            for callback in self._callbacks['training_end']:
                callback(**kwargs)
    
    def on_epoch_start(self, epoch: int, **kwargs) -> None:
        """
        Callback untuk awal epoch.
        
        Args:
            epoch: Nomor epoch
            **kwargs: Parameter tambahan
        """
        self.update('epoch_start', {'epoch': epoch, **kwargs})
        
        # Panggil callback jika ada
        if 'epoch_start' in self._callbacks:
            for callback in self._callbacks['epoch_start']:
                callback(epoch=epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs) -> None:
        """
        Callback untuk akhir epoch.
        
        Args:
            epoch: Nomor epoch
            metrics: Metrik training/validasi
            **kwargs: Parameter tambahan
        """
        self.update('epoch_end', {'epoch': epoch, 'metrics': metrics, **kwargs})
        
        # Panggil callback jika ada
        if 'epoch_end' in self._callbacks:
            for callback in self._callbacks['epoch_end']:
                callback(epoch=epoch, metrics=metrics, **kwargs)
    
    def on_batch_start(self, batch_idx: int, **kwargs) -> None:
        """
        Callback untuk awal batch.
        
        Args:
            batch_idx: Indeks batch
            **kwargs: Parameter tambahan
        """
        self.update('batch_start', {'batch_idx': batch_idx, **kwargs})
        
        # Panggil callback jika ada
        if 'batch_start' in self._callbacks:
            for callback in self._callbacks['batch_start']:
                callback(batch_idx=batch_idx, **kwargs)
    
    def on_batch_end(self, batch_idx: int, loss: float, **kwargs) -> None:
        """
        Callback untuk akhir batch.
        
        Args:
            batch_idx: Indeks batch
            loss: Loss pada batch
            **kwargs: Parameter tambahan
        """
        self.update('batch_end', {'batch_idx': batch_idx, 'loss': loss, **kwargs})
        
        # Panggil callback jika ada
        if 'batch_end' in self._callbacks:
            for callback in self._callbacks['batch_end']:
                callback(batch_idx=batch_idx, loss=loss, **kwargs)
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Daftarkan callback untuk event tertentu.
        
        Args:
            event: Nama event
            callback: Fungsi callback
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        
        self._callbacks[event].append(callback)
    
    def unregister_callback(self, event: str, callback: Callable) -> None:
        """
        Hapus callback dari event tertentu.
        
        Args:
            event: Nama event
            callback: Fungsi callback yang akan dihapus
        """
        if event in self._callbacks and callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)