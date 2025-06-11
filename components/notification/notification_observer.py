"""
File: smartcash/components/notification/notification_observer.py
Deskripsi: Adapter observer untuk notification manager dengan implementasi BaseObserver
"""

from typing import Dict, Any, Optional, Callable, List, Union
from smartcash.components.observer.base_observer import BaseObserver
from smartcash.common.logger import get_logger

class NotificationObserver(BaseObserver):
    """Adapter observer untuk notification manager yang mengimplementasikan BaseObserver"""
    
    def __init__(self, 
                event_type: str,
                callback: Callable,
                name: Optional[str] = None,
                priority: int = 0):
        """
        Inisialisasi NotificationObserver dengan callback.
        
        Args:
            event_type: Tipe event yang akan diobservasi
            callback: Fungsi callback yang akan dipanggil saat event terjadi
            name: Nama observer (opsional)
            priority: Prioritas observer
        """
        # Gunakan nama event sebagai filter
        super().__init__(
            name=name or f"NotificationObserver_{event_type}",
            priority=priority,
            event_filter=event_type,
            enabled=True
        )
        self.event_type = event_type
        self.callback = callback
        self.logger = get_logger("smartcash.notification")
    
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """
        Metode yang dipanggil saat event terjadi.
        
        Args:
            event_type: Tipe event yang terjadi
            sender: Pengirim event
            **kwargs: Parameter tambahan untuk event
        """
        try:
            # Panggil callback dengan parameter yang sesuai
            self.callback(event_type=event_type, sender=sender, **kwargs)
        except Exception as e:
            # Silent fail untuk mencegah error callback mengganggu proses utama
            if self.logger:
                self.logger.error(f"🔥 Error pada NotificationObserver.update: {str(e)}")

def create_notification_observer(event_type: Union[str, List[str]], 
                               callback: Callable,
                               name: Optional[str] = None,
                               priority: int = 0) -> Union[NotificationObserver, List[NotificationObserver]]:
    """
    Factory function untuk membuat NotificationObserver.
    
    Args:
        event_type: Tipe event atau list tipe event yang akan diobservasi
        callback: Fungsi callback yang akan dipanggil saat event terjadi
        name: Nama observer (opsional)
        priority: Prioritas observer
        
    Returns:
        NotificationObserver atau list NotificationObserver
    """
    # Jika event_type adalah list, buat observer untuk setiap event
    if isinstance(event_type, list):
        return [NotificationObserver(et, callback, f"{name}_{i}" if name else None, priority) 
                for i, et in enumerate(event_type)]
    
    # Jika event_type adalah string, buat satu observer
    return NotificationObserver(event_type, callback, name, priority)
