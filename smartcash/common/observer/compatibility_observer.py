# File: smartcash/common/observer/compatibility_observer.py
# Deskripsi: Observer adapter untuk kompatibilitas dengan sistem observer lama

from typing import Any, Callable, Dict, Optional

from smartcash.common.observer.base_observer import BaseObserver
from smartcash.common.logger import get_logger


class CompatibilityObserver(BaseObserver):
    """
    Observer adapter untuk kompatibilitas dengan sistem observer lama.
    
    Observer ini menerima observer lama (yang tidak inherit BaseObserver)
    dan membungkusnya dalam BaseObserver yang valid.
    """
    
    def __init__(
        self,
        observer: Any,
        name: Optional[str] = None,
        priority: int = 0
    ):
        """
        Inisialisasi compatibility observer.
        
        Args:
            observer: Observer lama yang akan dibungkus
            name: Nama observer (opsional)
            priority: Prioritas observer
        """
        # Gunakan nama dari observer jika tersedia
        if name is None and hasattr(observer, '__class__'):
            name = f"Compat_{observer.__class__.__name__}"
        elif name is None:
            name = f"Compat_{id(observer)}"
            
        super().__init__(name=name, priority=priority)
        self.observer = observer
        self.logger = get_logger(f"compat_observer.{name}")
        
        # Periksa metode yang tersedia pada observer
        self.has_update = hasattr(observer, 'update') and callable(getattr(observer, 'update'))
        self.has_on_events = self._check_on_event_methods()
        
        if not (self.has_update or self.has_on_events):
            self.logger.warning(f"⚠️ Observer {name} tidak memiliki metode update atau on_event_* yang diperlukan")
    
    def _check_on_event_methods(self) -> bool:
        """Periksa apakah observer memiliki setidaknya satu metode on_*."""
        on_methods = [
            'on_preprocessing_start', 'on_preprocessing_end', 'on_preprocessing_progress',
            'on_validation_start', 'on_validation_end', 
            'on_augmentation_start', 'on_augmentation_end',
            'on_batch_start', 'on_batch_end',
            'on_epoch_start', 'on_epoch_end',
            'on_train_start', 'on_train_end',
            'on_evaluation_start', 'on_evaluation_end'
        ]
        
        return any(hasattr(self.observer, method) and callable(getattr(self.observer, method)) 
                for method in on_methods)
    
    def _map_event_to_method(self, event_type: str) -> Optional[str]:
        """
        Memetakan tipe event ke nama metode observer.
        
        Args:
            event_type: Tipe event
            
        Returns:
            Nama metode yang sesuai dengan event, atau None jika tidak ada
        """
        # Mapping event ke metode
        event_mapping = {
            'preprocessing.start': 'on_preprocessing_start',
            'preprocessing.end': 'on_preprocessing_end',
            'preprocessing.progress': 'on_preprocessing_progress',
            'preprocessing.validation': 'on_validation',
            'preprocessing.augmentation': 'on_augmentation',
            'training.epoch.start': 'on_epoch_start',
            'training.epoch.end': 'on_epoch_end',
            'training.batch.start': 'on_batch_start',
            'training.batch.end': 'on_batch_end',
            'training.start': 'on_train_start',
            'training.end': 'on_train_end',
            'evaluation.start': 'on_evaluation_start',
            'evaluation.end': 'on_evaluation_end'
        }
        
        # Check direct mapping
        if event_type in event_mapping:
            method_name = event_mapping[event_type]
            if hasattr(self.observer, method_name) and callable(getattr(self.observer, method_name)):
                return method_name
        
        # Untuk event dengan format action
        if '.' in event_type:
            base_event = event_type.split('.')[0]
            if 'action' in event_type:
                action = event_type.split('action=')[-1].split(',')[0]
                method_name = f"on_{base_event}_{action}"
                if hasattr(self.observer, method_name) and callable(getattr(self.observer, method_name)):
                    return method_name
        
        return None
    
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """
        Implementasi metode update.
        
        Args:
            event_type: Tipe event
            sender: Objek yang mengirimkan event
            **kwargs: Parameter tambahan
        """
        try:
            # Jika observer memiliki metode update, panggil itu
            if self.has_update:
                self.observer.update(event_type, sender, **kwargs)
                return
                
            # Jika observer memiliki metode on_* yang sesuai, panggil itu
            method_name = self._map_event_to_method(event_type)
            if method_name and hasattr(self.observer, method_name):
                method = getattr(self.observer, method_name)
                method(**kwargs)
                return
                
            # Jika tidak ada metode yang cocok, log warning
            self.logger.debug(f"⚠️ Tidak ada handler untuk event {event_type} pada observer {self.name}")
            
        except Exception as e:
            self.logger.warning(f"❌ Error pada observer {self.name} saat memproses event {event_type}: {str(e)}")