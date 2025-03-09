# File: smartcash/handlers/model/observers/model_observer_interface.py
# Author: Alfrida Sabar
# Deskripsi: Interface observer untuk model yang terintegrasi dengan framework observer

from typing import Dict, Any, Optional
from smartcash.utils.observer import Observer
from smartcash.utils.logger import SmartCashLogger, get_logger

class ModelObserverInterface(Observer):
    """
    Interface observer untuk model yang terintegrasi dengan framework observer.
    Memperluas observer umum untuk event-event spesifik training model.
    """
    
    def __init__(self, name: str = "model_observer", logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi model observer.
        
        Args:
            name: Nama observer
            logger: Custom logger (opsional)
        """
        super().__init__(name=name)
        self.logger = logger or get_logger(f"model.{name}")
        
    def on_training_start(self, **kwargs):
        """
        Event ketika training dimulai.
        
        Args:
            **kwargs: Parameter training seperti model, epochs, dll.
        """
        self.notify('training_start', kwargs)
    
    def on_training_end(self, **kwargs):
        """
        Event ketika training selesai.
        
        Args:
            **kwargs: Hasil training seperti best_val_loss, epochs_completed, dll.
        """
        self.notify('training_end', kwargs)
    
    def on_epoch_start(self, epoch: int, **kwargs):
        """
        Event ketika epoch dimulai.
        
        Args:
            epoch: Nomor epoch
            **kwargs: Parameter tambahan
        """
        self.notify('epoch_start', {'epoch': epoch, **kwargs})
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """
        Event ketika epoch selesai.
        
        Args:
            epoch: Nomor epoch
            metrics: Metrik-metrik epoch seperti loss, accuracy, dll.
            **kwargs: Parameter tambahan
        """
        self.notify('epoch_end', {'epoch': epoch, 'metrics': metrics, **kwargs})
    
    def on_batch_end(self, batch_idx: int, epoch: int, loss: float, **kwargs):
        """
        Event ketika batch selesai.
        
        Args:
            batch_idx: Indeks batch
            epoch: Nomor epoch
            loss: Nilai loss batch
            **kwargs: Parameter tambahan
        """
        self.notify('batch_end', {'batch_idx': batch_idx, 'epoch': epoch, 'loss': loss, **kwargs})
    
    def on_validation_start(self, epoch: int, **kwargs):
        """
        Event ketika validasi dimulai.
        
        Args:
            epoch: Nomor epoch
            **kwargs: Parameter tambahan
        """
        self.notify('validation_start', {'epoch': epoch, **kwargs})
    
    def on_validation_end(self, epoch: int, metrics: Dict[str, Any], **kwargs):
        """
        Event ketika validasi selesai.
        
        Args:
            epoch: Nomor epoch
            metrics: Metrik-metrik validasi seperti val_loss, val_accuracy, dll.
            **kwargs: Parameter tambahan
        """
        self.notify('validation_end', {'epoch': epoch, 'metrics': metrics, **kwargs})
    
    def on_checkpoint_save(self, checkpoint_path: str, is_best: bool = False, **kwargs):
        """
        Event ketika checkpoint disimpan.
        
        Args:
            checkpoint_path: Path ke checkpoint yang disimpan
            is_best: Flag apakah ini checkpoint terbaik
            **kwargs: Parameter tambahan
        """
        self.notify('checkpoint_save', {'checkpoint_path': checkpoint_path, 'is_best': is_best, **kwargs})