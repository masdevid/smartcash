"""
File: smartcash/components/observer/decorators.py
Deskripsi: Decorator untuk observer pattern di SmartCash
"""

import functools
import inspect
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union, get_type_hints

from smartcash.components.observer.event_dispatcher import EventDispatcher
from smartcash.components.observer.base_observer import BaseObserver
from smartcash.common.logger import get_logger


# Logger
_logger = get_logger("observer_decorators")

# Lock untuk thread-safety
_lock = threading.RLock()

# Set untuk menyimpan metode yang ditandai sebagai observable
_observable_methods = set()

# Dict untuk menyimpan cache class observer yang dibuat melalui decorator
_observer_class_cache: Dict[str, Type[BaseObserver]] = {}


def observable(
    event_type: Optional[str] = None,
    include_args: bool = True,
    include_result: bool = False,
    include_self: bool = False,
    include_self_attributes: Optional[List[str]] = None,
) -> Callable:
    """
    Decorator untuk menandai metode sebagai observable.
    
    Ketika metode yang ditandai dengan decorator ini dipanggil, akan mengirimkan
    notifikasi ke semua observer yang terdaftar untuk event yang sesuai.
    
    Args:
        event_type: Tipe event yang akan dikirim (default: nama kelas + "." + nama metode)
        include_args: Sertakan argumen metode dalam notifikasi
        include_result: Sertakan hasil metode dalam notifikasi
        include_self: Sertakan instance kelas (self) dalam notifikasi sebagai 'instance'
        include_self_attributes: List atribut dari instance yang akan disertakan
        
    Returns:
        Fungsi yang dibungkus
    """
    def decorator(func):
        # Dapatkan informasi fungsi
        sig = inspect.signature(func)
        func_name = func.__name__
        
        # Tambahkan ke set observable methods
        with _lock:
            _observable_methods.add(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Dapatkan instance (self) jika metode kelas
            instance = args[0] if args and not isinstance(func, staticmethod) else None
            
            # Tentukan event_type
            actual_event_type = event_type
            if actual_event_type is None and instance is not None:
                # Gunakan nama kelas + nama metode sebagai default event_type
                actual_event_type = f"{instance.__class__.__name__.lower()}.{func_name}"
            elif actual_event_type is None:
                # Jika tidak ada instance, gunakan nama fungsi saja
                actual_event_type = func_name
            
            # Gabungkan args ke kwargs untuk notifikasi
            notification_kwargs = {}
            
            # Tambahkan argumen jika diperlukan
            if include_args:
                # Dapatkan nama parameter dari signature
                parameters = list(sig.parameters.keys())
                
                # Jika instance bukan None, skip parameter pertama (self)
                start_idx = 1 if instance is not None and not isinstance(func, staticmethod) else 0
                
                # Tambahkan positional args
                for i, arg in enumerate(args[start_idx:], start=start_idx):
                    if i < len(parameters):
                        notification_kwargs[parameters[i]] = arg
                
                # Tambahkan keyword args
                notification_kwargs.update(kwargs)
            
            # Tambahkan instance jika diperlukan
            if include_self and instance is not None:
                notification_kwargs['instance'] = instance
                
                # Tambahkan atribut tertentu dari instance
                if include_self_attributes:
                    for attr in include_self_attributes:
                        if hasattr(instance, attr):
                            notification_kwargs[f"instance_{attr}"] = getattr(instance, attr)
            
            # Panggil fungsi asli
            result = func(*args, **kwargs)
            
            # Tambahkan hasil jika diperlukan
            if include_result:
                notification_kwargs['result'] = result
            
            # Kirim notifikasi
            sender = instance if instance is not None else func
            EventDispatcher.notify(actual_event_type, sender, **notification_kwargs)
            
            return result
        
        # Tambahkan atribut ke fungsi yang dibungkus
        wrapper.__observable__ = True
        wrapper.__event_type__ = event_type
        
        return wrapper
    
    return decorator


def observe(
    event_types: Union[str, List[str]],
    priority: int = 0,
    event_filter: Optional[Any] = None,
    method_name: Optional[str] = None
) -> Callable:
    """
    Decorator untuk membuat metode kelas menjadi observer.
    
    Decorator ini akan membuat kelas dengan metode yang ditandai menjadi observer
    untuk event yang ditentukan. Kelas yang menggunakan decorator ini harus memiliki
    metode dengan nama `method_name` (default: "update").
    
    Args:
        event_types: Tipe event atau list tipe event yang akan diobservasi
        priority: Prioritas observer
        event_filter: Filter untuk event yang akan diproses
        method_name: Nama metode yang akan dipanggil (default: "update")
        
    Returns:
        Decorator untuk kelas
    """
    # Konversi event_types ke list jika perlu
    if isinstance(event_types, str):
        event_types = [event_types]
    
    # Gunakan nama metode default jika tidak disebutkan
    if method_name is None:
        method_name = "update"
    
    def decorator(cls):
        # Buat nama kelas untuk observer
        observer_class_name = f"{cls.__name__}Observer"
        
        # Periksa apakah kelas sudah memiliki metode yang ditentukan
        if not hasattr(cls, method_name) or not callable(getattr(cls, method_name)):
            raise AttributeError(
                f"Kelas {cls.__name__} tidak memiliki metode '{method_name}' yang dapat dijadikan observer")
        
        # Periksa apakah kelas observer sudah di-cache
        cache_key = f"{cls.__module__}.{cls.__name__}_{method_name}"
        if cache_key in _observer_class_cache:
            # Gunakan kelas yang sudah di-cache
            observer_class = _observer_class_cache[cache_key]
        else:
            # Buat kelas observer baru yang mewarisi BaseObserver
            class ClassMethodObserver(BaseObserver):
                def __init__(self, instance, **kwargs):
                    super().__init__(**kwargs)
                    self.instance = instance
                
                def update(self, event_type, sender, **kwargs):
                    # Panggil metode pada instance
                    getattr(self.instance, method_name)(event_type, sender, **kwargs)
            
            # Ganti nama kelas untuk debugging yang lebih mudah
            ClassMethodObserver.__name__ = observer_class_name
            
            # Simpan ke cache
            with _lock:
                _observer_class_cache[cache_key] = ClassMethodObserver
                
            observer_class = ClassMethodObserver
        
        # Override metode __new__ untuk mendaftarkan observer saat instance dibuat
        original_new = cls.__new__
        
        @functools.wraps(original_new)
        def new_new(cls, *args, **kwargs):
            # Panggil __new__ asli
            instance = original_new(cls) if original_new is object.__new__ else original_new(cls, *args, **kwargs)
            
            # Buat dan daftarkan observer untuk instance ini
            observer = observer_class(
                instance=instance,
                name=f"{instance.__class__.__name__}_{id(instance)}",
                priority=priority,
                event_filter=event_filter
            )
            
            # Daftarkan observer ke dispatcher untuk semua event
            for event_type in event_types:
                EventDispatcher.register(event_type, observer, priority)
            
            # Simpan observer ke instance untuk reference
            if not hasattr(instance, '_observers'):
                instance._observers = []
            instance._observers.append(observer)
            return instance
        
        # Ganti metode __new__
        cls.__new__ = new_new
        
        # Override metode __del__ untuk membatalkan registrasi observer
        original_del = cls.__del__ if hasattr(cls, '__del__') else None
        
        def new_del(self):
            # Batalkan registrasi semua observer
            if hasattr(self, '_observers'):
                for observer in self._observers:
                    EventDispatcher.unregister_from_all(observer)
                self._observers.clear()
            
            # Panggil __del__ asli jika ada
            if original_del is not None:
                original_del(self)
        
        # Ganti metode __del__
        cls.__del__ = new_del
        
        return cls
    
    return decorator