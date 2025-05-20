"""
File: smartcash/components/observer/manager_observer.py
Deskripsi: Manager untuk observer pattern di SmartCash dengan integrasi progress_tracker dan optimasi resource
"""
import inspect
import weakref
import concurrent.futures
from typing import Dict, List, Optional, Type, Any, Set, Callable, Union
from threading import RLock

from smartcash.common.logger import get_logger
from smartcash.components.observer.base_observer import BaseObserver
from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
from smartcash.components.observer.cleanup_observer import register_observer_manager

# Import progress_tracker tanpa circular dependency
from smartcash.common.progress import get_progress_tracker


class ObserverManager:
    """
    Manager untuk observer pattern di SmartCash dengan integrasi common.
    
    Mengintegrasikan progress_tracker dan progress_observer dari common
    untuk mengurangi duplikasi kode dan meningkatkan konsistensi.
    """
    
    def __init__(self, 
                auto_register: bool = True, 
                logger: Optional[Any] = None,
                auto_cleanup: bool = True,
                max_workers: Optional[int] = None):
        """
        Inisialisasi ObserverManager dengan dukungan auto-cleanup.
        
        Args:
            auto_register: Otomatis daftarkan observer ke dispatcher
            logger: Logger custom (optional)
            auto_cleanup: Otomatis daftarkan manager untuk cleanup saat aplikasi selesai
            max_workers: Jumlah maksimum worker thread (None untuk auto)
        """
        self.auto_register = auto_register
        self.logger = logger or get_logger()
        self._observer_groups = {}  # Group -> set(observer_weakrefs)
        self._observers = weakref.WeakSet()  # Semua observer
        self._lock = RLock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        
        # Daftarkan manager untuk auto cleanup jika diminta
        if auto_cleanup:
            register_observer_manager(self)
            
        self.logger.debug("🔄 ObserverManager diinisialisasi")
    
    def create_simple_observer(
        self, 
        event_type: Union[str, List[str]],
        callback: Callable,
        name: Optional[str] = None,
        priority: int = 0,
        event_filter: Optional[Any] = None,
        group: Optional[str] = None
    ) -> BaseObserver:
        """
        Membuat simple observer dengan callback function.
        
        Args:
            event_type: Tipe event atau list tipe event
            callback: Fungsi callback untuk observer
            name: Nama observer (opsional)
            priority: Prioritas observer
            event_filter: Filter untuk event
            group: Grup observer untuk manajemen
            
        Returns:
            BaseObserver instance
        """
        # Validasi callback dengan one-liner
        if not callable(callback): raise TypeError("callback harus callable")
        
        # Buat anonymous class dengan inline lambda untuk update method
        SimpleObserver = type('SimpleObserver', (BaseObserver,), {
            'update': lambda self, event_type, sender, **kwargs: callback(event_type, sender, **kwargs)
        })
        
        # Gunakan nama function sebagai nama observer dengan one-liner
        observer_name = name or (callback.__name__ if hasattr(callback, '__name__') else f"SimpleObserver_{id(callback)}")
        
        # Buat instance dan register dalam satu langkah
        observer = SimpleObserver(name=observer_name, priority=priority, event_filter=event_filter)
        return self._register_observer(observer, event_type, group)
    
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
        """
        Membuat dan mendaftarkan observer dari kelas yang diberikan.
        
        Args:
            observer_class: Kelas observer (subclass dari BaseObserver)
            event_type: Tipe event atau list tipe event
            name: Nama observer (opsional)
            priority: Prioritas observer (opsional)
            event_filter: Filter untuk event (opsional)
            group: Grup observer untuk manajemen
            **kwargs: Parameter tambahan untuk constructor observer
            
        Returns:
            BaseObserver instance
        """
        # Validasi kelas observer
        if not inspect.isclass(observer_class) or not issubclass(observer_class, BaseObserver):
            raise TypeError(f"observer_class harus subclass dari BaseObserver")
        
        # Buat parameter init dengan one-liner
        init_params = {k: v for k, v in {'name': name, 'priority': priority, 'event_filter': event_filter}.items() if v is not None}
        init_params.update(kwargs)  # Tambahkan parameter tambahan
        
        # Instantiate observer dan register dalam satu langkah
        observer = observer_class(**init_params)
        return self._register_observer(observer, event_type, group)
    
    def _register_observer(self, observer: BaseObserver, event_type: Union[str, List[str]], group: Optional[str]) -> BaseObserver:
        """
        Helper internal untuk mendaftarkan observer ke registry dan grup.
        
        Args:
            observer: Observer yang akan didaftarkan
            event_type: Tipe event atau list tipe event
            group: Grup observer untuk manajemen
            
        Returns:
            Observer yang didaftarkan
        """
        with self._lock:
            # Tambahkan ke set utama
            self._observers.add(observer)
            
            # Tambahkan ke grup jika perlu
            if group is not None:
                if group not in self._observer_groups:
                    self._observer_groups[group] = weakref.WeakSet()
                self._observer_groups[group].add(observer)
        
        # Daftarkan ke event dispatcher jika auto_register
        if self.auto_register:
            if isinstance(event_type, list):
                EventDispatcher.register_many(event_type, observer)
            else:
                EventDispatcher.register(event_type, observer)
        
        return observer
    
    def create_progress_observer(
        self,
        event_types: List[str],
        total: int,
        desc: str = "Processing",
        name: Optional[str] = None,
        group: Optional[str] = None,
        show_progress: bool = True
    ) -> Any:  # Return type Any untuk menghindari circular import
        """
        Membuat observer untuk monitoring progress.
        
        Args:
            event_types: List tipe event yang akan diobservasi
            total: Total unit yang akan diproses
            desc: Deskripsi progress
            name: Nama observer (opsional)
            group: Grup observer untuk manajemen
            show_progress: Tampilkan progress bar
            
        Returns:
            ProgressObserver instance
        """
        # Import di sini untuk menghindari circular import
        from smartcash.common.progress import create_progress_tracker_observer
        
        # Gunakan fungsi helper untuk membuat observer
        tracker, observer = create_progress_tracker_observer(
            name=name or desc,
            total=total,
            desc=desc,
            display=show_progress
        )
        
        # Simpan ke grup jika perlu
        if group is not None:
            with self._lock:
                if group not in self._observer_groups:
                    self._observer_groups[group] = weakref.WeakSet()
                self._observer_groups[group].add(observer)
                
            # Tambahkan atribut grup untuk referensi
            observer.group = group
        
        # Tambahkan ke set utama
        self._observers.add(observer)
        
        # Daftarkan ke event dispatcher jika auto_register
        if self.auto_register:
            EventDispatcher.register_many(event_types, observer)
            
        return observer
    
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
        """
        Membuat observer untuk logging event.
        
        Args:
            event_types: List tipe event yang akan diobservasi
            log_level: Level log yang akan digunakan
            name: Nama observer (opsional)
            format_string: Format string untuk pesan log (opsional)
            include_timestamp: Sertakan timestamp dalam pesan
            include_sender: Sertakan sender dalam pesan
            logger_name: Nama logger kustom (opsional)
            group: Grup observer untuk manajemen
            
        Returns:
            BaseObserver instance
        """
        # Buat custom logger dengan one-liner
        logger = get_logger(logger_name or "event_logger")
        
        # Dapatkan metode log dengan getattr
        log_method = getattr(logger, log_level.lower(), logger.info)
        
        # Buat logging observer dengan class factory dan lambda expression
        LoggingObserver = type('LoggingObserver', (BaseObserver,), {
            'update': lambda self, event_type, sender, **kwargs: log_method(
                format_string.format(event_type=event_type, sender=sender, **kwargs) if format_string else
                self._format_message(event_type, sender, include_timestamp, include_sender, kwargs)
            ),
            '_format_message': lambda self, event_type, sender, include_timestamp, include_sender, kwargs: ''.join([
                f"Event '{event_type}' terjadi",
                f" pada {kwargs['timestamp']}" if include_timestamp and 'timestamp' in kwargs else "",
                f" dari {sender}" if include_sender else "",
                f" dengan data: {self._filter_data(kwargs)}" if kwargs else ""
            ]),
            '_filter_data': lambda self, kwargs: {k: v for k, v in kwargs.items() if k not in ('timestamp', 'notification_id') and not k.startswith('_')}
        })
        
        # Buat instance observer
        return self.create_observer(
            LoggingObserver, 
            event_types, 
            name=name or f"LoggingObserver_{log_level}", 
            group=group
        )
    
    def execute_async(self, func: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """
        Eksekusi fungsi secara asinkron menggunakan ThreadPoolExecutor internal.
        
        Args:
            func: Fungsi yang akan dieksekusi
            *args: Argumen positional untuk fungsi
            **kwargs: Argumen keyword untuk fungsi
            
        Returns:
            Future yang mewakili hasil eksekusi
        """
        return self._executor.submit(func, *args, **kwargs)
    
    def execute_for_all_observers(
        self, 
        func: Callable[[BaseObserver], Any],
        group: Optional[str] = None,
        parallel: bool = True
    ) -> List[Any]:
        """
        Eksekusi fungsi untuk semua observer atau observer dalam grup tertentu.
        
        Args:
            func: Fungsi yang menerima observer sebagai argumen
            group: Terbatas ke grup tertentu (opsional)
            parallel: Eksekusi paralel menggunakan ThreadPoolExecutor
            
        Returns:
            List hasil eksekusi
        """
        # Pilih observer dari grup atau semua
        observers = self.get_observers_by_group(group) if group else self.get_all_observers()
        
        if not observers:
            return []
        
        if parallel:
            # Eksekusi paralel dengan ThreadPoolExecutor
            futures = [self._executor.submit(func, obs) for obs in observers]
            return [future.result() for future in concurrent.futures.as_completed(futures)]
        else:
            # Eksekusi sekuensial dengan list comprehension
            return [func(obs) for obs in observers]
    
    def get_observers_by_group(self, group: str) -> List[BaseObserver]:
        """Mendapatkan semua observer dalam grup tertentu dengan one-liner."""
        with self._lock:
            return list(self._observer_groups.get(group, set()))
    
    def unregister_group(self, group: str) -> int:
        """
        Membatalkan registrasi semua observer dalam grup tertentu.
        
        Args:
            group: Nama grup
            
        Returns:
            Jumlah observer yang dibatalkan
        """
        observers = self.get_observers_by_group(group)
        count = 0
        
        if not observers:
            return 0
        
        # Unregister semua observer dalam grup dengan paralelisme
        def unregister_observer(obs):
            EventDispatcher.unregister_from_all(obs)
            return 1
            
        results = self.execute_for_all_observers(unregister_observer, group=group)
        count = sum(results)
        
        # Hapus grup
        with self._lock:
            if group in self._observer_groups:
                self._observer_groups[group].clear()
                del self._observer_groups[group]
        
        return count
    
    def unregister_all(self) -> int:
        """
        Membatalkan registrasi semua observer yang dikelola.
        
        Returns:
            Jumlah observer yang dibatalkan
        """
        with self._lock:
            # Konversi ke list untuk mencegah perubahan saat iterasi
            observers = list(self._observers)
            count = len(observers)
            
            # Unregister semua observer
            for observer in observers:
                EventDispatcher.unregister_from_all(observer)
            
            # Bersihkan set dan grup
            self._observers.clear()
            self._observer_groups.clear()
            
            # Shutdown executor
            self._executor.shutdown(wait=False)
            
            return count
    
    def get_all_observers(self) -> List[BaseObserver]:
        """Mendapatkan semua observer yang dikelola dengan one-liner."""
        with self._lock:
            return list(self._observers)
    
    def get_observer_count(self) -> int:
        """Mendapatkan jumlah observer yang dikelola dengan one-liner."""
        with self._lock:
            return len(self._observers)
    
    def get_group_count(self) -> int:
        """Mendapatkan jumlah grup observer dengan one-liner."""
        with self._lock:
            return len(self._observer_groups)
    
    def get_stats(self) -> Dict[str, Any]:
        """Mendapatkan statistik manager dengan one-liner."""
        with self._lock:
            stats = {
                'observer_count': len(self._observers),
                'group_count': len(self._observer_groups),
                'groups': {group: len(observers) for group, observers in self._observer_groups.items()}
            }
        
        # Tambahkan statistik dari dispatcher
        stats.update(EventDispatcher.get_stats())
        return stats
    
    def shutdown(self):
        """Shutdown manager dan lepaskan resources."""
        self.unregister_all()
        if hasattr(self._executor, '_shutdown') and not self._executor._shutdown:
            self._executor.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup saat instance dihapus."""
        try:
            self.shutdown()
        except:
            pass

    def register(self, observer, event_types=None):
        """
        Register an observer for the given event types (or observer.event_types if not provided).
        """
        if event_types is None and hasattr(observer, 'event_types'):
            event_types = getattr(observer, 'event_types')
        if event_types is None:
            raise ValueError('event_types must be provided or observer must have event_types attribute')
        if isinstance(event_types, (list, tuple, set)):
            for event_type in event_types:
                EventDispatcher.register(event_type, observer)
        else:
            EventDispatcher.register(event_types, observer)

# Singleton instance for ObserverManager
_observer_manager_instance = None

def get_observer_manager():
    global _observer_manager_instance
    if _observer_manager_instance is None:
        _observer_manager_instance = ObserverManager()
    return _observer_manager_instance