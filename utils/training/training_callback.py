"""
File: smartcash/utils/training/training_callbacks.py
Author: Alfrida Sabar
Deskripsi: Sistem callback untuk proses training dengan pendekatan observer pattern
           yang terintegrasi dengan implementasi observer terkonsolidasi
"""

from typing import Dict, List, Callable, Any, Optional, Union
import functools

from smartcash.utils.observer import (
    EventDispatcher, EventTopics, ObserverManager, BaseObserver
)
from smartcash.utils.logger import get_logger, SmartCashLogger


class TrainingCallbacks:
    """
    Sistem callback untuk proses training dengan pendekatan event-handler.
    Menggunakan observer pattern terkonsolidasi dari utils/observer.
    Mendukung berbagai tipe callback pada berbagai titik dalam training pipeline.
    """
    
    # Mapping event lama ke EventTopics
    EVENT_MAPPING = {
        'batch_end': EventTopics.BATCH_END,
        'epoch_end': EventTopics.EPOCH_END,
        'training_end': EventTopics.TRAINING_END,
        'validation_end': EventTopics.VALIDATION_END,
        'checkpoint_saved': EventTopics.CHECKPOINT_SAVE,
    }
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi sistem callback.
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger("TrainingCallbacks")
        
        # Daftar event yang didukung (menggunakan format lama untuk kompatibilitas)
        self.supported_events = list(self.EVENT_MAPPING.keys())
        
        # Observer Manager untuk mengelola observer
        self.observer_manager = ObserverManager(auto_register=True)
        
        # Simpan referensi ke observer yang dibuat untuk pengelolaan
        self._observer_refs = {}  # {(event, callback_id): observer}
    
    def _get_event_topic(self, event: str) -> str:
        """
        Konversi event lama ke format EventTopics.
        
        Args:
            event: Nama event (format lama)
            
        Returns:
            EventTopic yang sesuai
        """
        if event in self.EVENT_MAPPING:
            return self.EVENT_MAPPING[event]
        
        # Fallback ke event original jika tidak ada dalam mapping
        self.logger.warning(f"⚠️ Event tidak dalam mapping: {event}, menggunakan as-is")
        return event
    
    def register(self, event: str, callback: Callable) -> bool:
        """
        Register callback function untuk event tertentu.
        
        Args:
            event: Nama event yang valid
            callback: Fungsi callback yang akan dipanggil
            
        Returns:
            Bool yang menunjukkan keberhasilan registrasi
        """
        if event not in self.supported_events:
            self.logger.warning(f"⚠️ Event tidak didukung: {event}")
            return False
        
        # Konversi ke EventTopics
        event_topic = self._get_event_topic(event)
        
        # Buat adapter untuk callback
        # Format lama menggunakan **kwargs, format baru menggunakan (event_type, sender, **kwargs)
        adapter_callback = lambda event_type, sender, **kwargs: callback(**kwargs)
        
        # Buat observer
        callback_id = id(callback)
        observer = self.observer_manager.create_simple_observer(
            event_type=event_topic,
            callback=adapter_callback,
            name=f"CallbackObserver_{event}_{callback_id}",
            priority=0
        )
        
        # Simpan referensi ke observer
        self._observer_refs[(event, callback_id)] = observer
        
        self.logger.debug(f"🔌 Registered callback untuk '{event}' (Topic: {event_topic})")
        return True
    
    def unregister(self, event: str, callback: Callable) -> bool:
        """
        Unregister callback function dari event tertentu.
        
        Args:
            event: Nama event
            callback: Fungsi callback yang akan dihapus
            
        Returns:
            Bool yang menunjukkan keberhasilan penghapusan
        """
        callback_id = id(callback)
        key = (event, callback_id)
        
        if key in self._observer_refs:
            observer = self._observer_refs[key]
            
            # Unregister dari EventDispatcher
            event_topic = self._get_event_topic(event)
            EventDispatcher.unregister(event_topic, observer)
            
            # Hapus dari observer refs
            del self._observer_refs[key]
            return True
        
        return False
    
    def clear(self, event: str = None) -> None:
        """
        Hapus semua callback untuk event tertentu atau semua event.
        
        Args:
            event: Nama event (optional, jika None akan menghapus semua)
        """
        if event is None:
            # Unregister semua observer
            for key, observer in list(self._observer_refs.items()):
                event_name, _ = key
                event_topic = self._get_event_topic(event_name)
                EventDispatcher.unregister(event_topic, observer)
            
            # Reset observer refs
            self._observer_refs = {}
        else:
            # Unregister observer untuk event tertentu
            keys_to_remove = []
            for key, observer in self._observer_refs.items():
                event_name, _ = key
                if event_name == event:
                    event_topic = self._get_event_topic(event)
                    EventDispatcher.unregister(event_topic, observer)
                    keys_to_remove.append(key)
            
            # Hapus referensi
            for key in keys_to_remove:
                del self._observer_refs[key]
    
    def trigger(self, event: str, **kwargs) -> None:
        """
        Jalankan semua callback untuk event tertentu.
        
        Args:
            event: Nama event
            **kwargs: Parameter yang akan diteruskan ke callback
        """
        if event not in self.supported_events:
            return
        
        # Konversi ke EventTopics
        event_topic = self._get_event_topic(event)
        
        # Trigger event via EventDispatcher
        EventDispatcher.notify(event_topic, self, **kwargs)
    
    def get_callback_count(self, event: str = None) -> Dict[str, int]:
        """
        Dapatkan jumlah callback yang terdaftar.
        
        Args:
            event: Nama event (optional)
            
        Returns:
            Dict berisi jumlah callback per event atau untuk event tertentu
        """
        if event is not None:
            count = sum(1 for key in self._observer_refs if key[0] == event)
            return {event: count}
        
        # Hitung untuk semua event
        counts = {}
        for supported_event in self.supported_events:
            counts[supported_event] = sum(1 for key in self._observer_refs if key[0] == supported_event)
        
        return counts
    
    def add_default_callbacks(self) -> None:
        """
        Tambahkan callback default untuk monitoring.
        """
        # Progress tracking untuk batch
        def batch_progress(batch, loss, batch_size, **kwargs):
            if self.logger and batch % 20 == 0:
                self.logger.info(f"🔄 Batch {batch}: loss={loss:.4f}")
        
        # Callback untuk mencatat metrik validasi
        def validation_metrics(val_loss, metrics, **kwargs):
            if self.logger:
                metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                self.logger.info(f"📊 Validasi: loss={val_loss:.4f}, {metrics_str}")
        
        # Tambahkan callback default
        self.register('batch_end', batch_progress)
        self.register('validation_end', validation_metrics)