# File: smartcash/utils/observer/observer_manager.py
# Deskripsi: Manager untuk observer pattern di SmartCash dengan perbaikan dictionary storage

import inspect
from typing import Dict, List, Optional, Type, Any, Set, Callable, Union
import threading

from smartcash.utils.observer.base_observer import BaseObserver
from smartcash.utils.observer.event_dispatcher import EventDispatcher
from smartcash.utils.logger import get_logger


class ObserverManager:
    """Manager untuk observer pattern di SmartCash."""
    
    # Logger
    _logger = get_logger("observer_manager")
    
    # Lock untuk thread-safety
    _lock = threading.RLock()
    
    # Set untuk menyimpan semua observer yang dibuat (menggunakan set karena BaseObserver sudah bisa di-hash)
    _observers: Set[BaseObserver] = set()
    
    # Dictionary untuk menyimpan observer berdasarkan grupnya
    _observer_groups: Dict[str, Set[BaseObserver]] = {}
    
    def __init__(self, auto_register: bool = True):
        """Inisialisasi ObserverManager."""
        self.auto_register = auto_register
    
    def create_simple_observer(
        self, 
        event_type: Union[str, List[str]],
        callback: Callable,
        name: Optional[str] = None,
        priority: int = 0,
        event_filter: Optional[Any] = None,
        group: Optional[str] = None
    ) -> BaseObserver:
        """Membuat simple observer dengan callback function."""
        # Validasi callback
        if not callable(callback):
            raise TypeError("callback harus merupakan callable")
        
        # Buat simple observer
        class SimpleObserver(BaseObserver):
            def update(self, event_type, sender, **kwargs):
                return callback(event_type, sender, **kwargs)
        
        # Gunakan nama function sebagai nama observer jika tidak disebutkan
        if name is None and hasattr(callback, '__name__'):
            name = f"SimpleObserver_{callback.__name__}"
        elif name is None:
            name = f"SimpleObserver_{id(callback)}"
        
        # Buat instance observer
        observer = SimpleObserver(name=name, priority=priority, event_filter=event_filter)
        
        # Tambahkan ke set dan grup jika perlu
        with self._lock:
            self._observers.add(observer)
            if group is not None:
                if group not in self._observer_groups:
                    self._observer_groups[group] = set()
                self._observer_groups[group].add(observer)
        
        # Daftarkan ke event dispatcher jika auto_register
        if self.auto_register:
            if isinstance(event_type, list):
                EventDispatcher.register_many(event_type, observer, priority)
            else:
                EventDispatcher.register(event_type, observer, priority)
        
        return observer
    
    def create_observer(
        self,
        observer_class: Type[BaseObserver],
        event_type: Union[str, List[str]],
        name: Optional[str] = None,
        priority: Optional[int] = None,
        event_filter: Optional[Any] = None,
        group: Optional[str] = None,
        **kwargs
    ) -> BaseObserver:
        """Membuat dan mendaftarkan observer dari kelas yang diberikan."""
        # Validasi kelas
        if not inspect.isclass(observer_class) or not issubclass(observer_class, BaseObserver):
            raise TypeError(f"observer_class harus merupakan subclass dari BaseObserver")
        
        # Buat instance observer
        init_params = {}
        if name is not None:
            init_params['name'] = name
        if priority is not None:
            init_params['priority'] = priority
        if event_filter is not None:
            init_params['event_filter'] = event_filter
        
        # Tambahkan parameter tambahan
        init_params.update(kwargs)
        
        # Instantiate observer
        observer = observer_class(**init_params)
        
        # Tambahkan ke set dan grup jika perlu
        with self._lock:
            self._observers.add(observer)
            if group is not None:
                if group not in self._observer_groups:
                    self._observer_groups[group] = set()
                self._observer_groups[group].add(observer)
        
        # Daftarkan ke event dispatcher jika auto_register
        if self.auto_register:
            if isinstance(event_type, list):
                EventDispatcher.register_many(event_type, observer, observer.priority)
            else:
                EventDispatcher.register(event_type, observer, observer.priority)
        
        return observer
    
    def create_progress_observer(
        self,
        event_types: List[str],
        total: int,
        desc: str = "Processing",
        name: Optional[str] = None,
        use_tqdm: bool = True,
        callback: Optional[Callable] = None,
        group: Optional[str] = None
    ) -> BaseObserver:
        """Membuat observer untuk monitoring progress dengan tqdm."""
        # Import tqdm jika perlu
        if use_tqdm:
            try:
                from tqdm import tqdm
            except ImportError:
                self._logger.warning("⚠️ Package tqdm tidak tersedia, menggunakan mode tanpa tqdm")
                use_tqdm = False
        
        # Buat progress observer
        class ProgressObserver(BaseObserver):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.progress = 0
                self.total = total
                self.pbar = tqdm(total=total, desc=desc) if use_tqdm else None
            
            def update(self, event_type, sender, **kwargs):
                # Update progres
                increment = kwargs.get('increment', 1)
                self.progress += increment
                
                # Update progress bar jika ada
                if self.pbar is not None:
                    self.pbar.update(increment)
                
                # Call callback jika ada
                if callback is not None:
                    callback(event_type, sender, progress=self.progress, total=self.total, **kwargs)
                
                # Tutup progress bar jika selesai
                if self.progress >= self.total and self.pbar is not None:
                    self.pbar.close()
                    self.pbar = None
            
            def __del__(self):
                # Pastikan progress bar ditutup
                if hasattr(self, 'pbar') and self.pbar is not None:
                    self.pbar.close()
        
        # Gunakan nama default jika tidak disebutkan
        if name is None:
            name = f"ProgressObserver_{desc}"
        
        # Buat observer
        return self.create_observer(ProgressObserver, event_types, name=name, group=group)
    
    def create_logging_observer(
        self,
        event_types: List[str],
        log_level: str = "info",
        name: Optional[str] = None,
        format_string: Optional[str] = None,
        include_timestamp: bool = True,
        include_sender: bool = False,
        logger_name: Optional[str] = None,
        group: Optional[str] = None
    ) -> BaseObserver:
        """Membuat observer untuk logging event."""
        # Buat custom logger jika diperlukan
        logger = get_logger(logger_name or "event_logger")
        
        # Dapatkan metode log berdasarkan level
        log_method = getattr(logger, log_level.lower())
        
        # Buat logging observer
        class LoggingObserver(BaseObserver):
            def update(self, event_type, sender, **kwargs):
                # Format pesan
                if format_string is not None:
                    # Gunakan string template jika disediakan
                    try:
                        message = format_string.format(event_type=event_type, sender=sender, **kwargs)
                    except KeyError as e:
                        message = f"Event '{event_type}' terjadi (error format: {e})"
                else:
                    # Buat pesan default
                    message = f"Event '{event_type}' terjadi"
                    
                    # Tambahkan timestamp jika diperlukan
                    if include_timestamp and 'timestamp' in kwargs:
                        message += f" pada {kwargs['timestamp']}"
                    
                    # Tambahkan sender jika diperlukan
                    if include_sender:
                        message += f" dari {sender}"
                    
                    # Tambahkan data lain jika ada
                    if kwargs:
                        # Filter data yang akan ditampilkan
                        filtered_data = {k: v for k, v in kwargs.items()
                                        if k not in ('timestamp', 'notification_id') 
                                        and not k.startswith('_')}
                        if filtered_data:
                            message += f" dengan data: {filtered_data}"
                
                # Log pesan
                log_method(message)
        
        # Gunakan nama default jika tidak disebutkan
        if name is None:
            name = f"LoggingObserver_{log_level}"
        
        # Buat observer
        return self.create_observer(LoggingObserver, event_types, name=name, group=group)
    
    def get_observers_by_group(self, group: str) -> List[BaseObserver]:
        """Mendapatkan semua observer dalam grup tertentu."""
        with self._lock:
            if group not in self._observer_groups:
                return []
            return list(self._observer_groups[group])
    
    def unregister_group(self, group: str) -> int:
        """Membatalkan registrasi semua observer dalam grup tertentu."""
        observers = self.get_observers_by_group(group)
        
        for observer in observers:
            EventDispatcher.unregister_from_all(observer)
        
        # Hapus grup
        with self._lock:
            if group in self._observer_groups:
                del self._observer_groups[group]
        
        count = len(observers)
        self._logger.debug(f"🔌 Membatalkan registrasi {count} observer dari grup '{group}'")
        return count
    
    def unregister_all(self) -> int:
        """Membatalkan registrasi semua observer yang dikelola."""
        with self._lock:
            count = len(self._observers)
            
            for observer in self._observers:
                EventDispatcher.unregister_from_all(observer)
            
            # Bersihkan set dan grup
            self._observers.clear()
            self._observer_groups.clear()
        
        self._logger.debug(f"🔌 Membatalkan registrasi {count} observer yang dikelola")
        return count
    
    def get_all_observers(self) -> List[BaseObserver]:
        """Mendapatkan semua observer yang dikelola."""
        with self._lock:
            return list(self._observers)
    
    def get_observer_count(self) -> int:
        """Mendapatkan jumlah observer yang dikelola."""
        with self._lock:
            return len(self._observers)
    
    def get_group_count(self) -> int:
        """Mendapatkan jumlah grup observer."""
        with self._lock:
            return len(self._observer_groups)
    
    def get_stats(self) -> Dict[str, Any]:
        """Mendapatkan statistik manager."""
        with self._lock:
            stats = {
                'observer_count': len(self._observers),
                'group_count': len(self._observer_groups),
                'groups': {group: len(observers) for group, observers in self._observer_groups.items()}
            }
        
        # Tambahkan statistik dari dispatcher
        stats.update(EventDispatcher.get_stats())
        return stats