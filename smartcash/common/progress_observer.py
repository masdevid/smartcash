"""
File: smartcash/common/progress_observer.py
Deskripsi: Implementasi observer pattern untuk progress tracking dengan integrasi event system
"""

from typing import Dict, Any, List, Optional, Callable, Tuple

from smartcash.common.logger import get_logger
from smartcash.common.progress_tracker import ProgressTracker, get_progress_tracker
# Hapus import BaseObserver untuk menghindari circular import
# Gunakan string untuk type hinting

# Import EventTopics dengan cara yang menghindari circular import
try:
    from smartcash.components.observer.event_topics_observer import EventTopics
except ImportError:
    # Fallback jika EventTopics tidak bisa diimpor
    class EventTopics:
        """Fallback EventTopics."""
        PROGRESS_UPDATE = "progress.update"
        PROGRESS_START = "progress.start"
        PROGRESS_COMPLETE = "progress.complete"

class ProgressObserver:
    """Observer untuk melacak progres dari event."""
    
    def __init__(
        self, 
        tracker: Optional[ProgressTracker] = None,
        tracker_name: str = "default",
        event_types: List[str] = None, 
        total: int = 100,
        desc: str = "Progress",
        progress_key: str = 'progress',
        total_key: str = 'total',
        enabled: bool = True,
        logger = None
    ):
        """
        Inisialisasi ProgressObserver.
        
        Args:
            tracker: ProgressTracker yang akan diupdate (opsional)
            tracker_name: Nama tracker jika tidak menyediakan tracker langsung
            event_types: List tipe event yang akan diobservasi
            total: Total nilai progres (100%)
            desc: Deskripsi untuk progress tracker
            progress_key: Kunci untuk nilai progres dalam event
            total_key: Kunci untuk nilai total dalam event
            enabled: Apakah observer aktif
            logger: Logger kustom (opsional)
        """
        self.enabled = enabled
        self.tracker = tracker or get_progress_tracker(tracker_name, total, desc)
        self.event_types = event_types or [
            EventTopics.PROGRESS_UPDATE,
            EventTopics.PROGRESS_START,
            EventTopics.PROGRESS_COMPLETE
        ]
        self.progress_key = progress_key
        self.total_key = total_key
        self.logger = logger or get_logger("progress_observer")
        self.name = f"ProgressObserver_{tracker_name}"
        self.priority = 0  # Default priority
        
        # Setup tracker jika belum diset
        if self.tracker.total <= 0:
            self.tracker.set_total(total)
    
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """
        Update status saat menerima event.
        
        Args:
            event_type: Tipe event
            sender: Pengirim event
            **kwargs: Parameter tambahan dari event
        """
        if not self.enabled:
            return
            
        if event_type == EventTopics.PROGRESS_START:
            # Set atau reset total jika disediakan
            if self.total_key in kwargs:
                self.tracker.set_total(kwargs[self.total_key])
            
            # Set deskripsi jika disediakan
            if 'description' in kwargs:
                self.tracker.desc = kwargs['description']
                if self.tracker.pbar:
                    self.tracker.pbar.set_description(kwargs['description'])
                    
            # Reset tracker
            self.tracker.current = 0
            self.tracker.is_completed = False
            
            # Log start
            if 'message' in kwargs:
                self.logger.info(f"🚀 {kwargs['message']}")
            else:
                self.logger.info(f"🚀 {self.tracker.desc} dimulai")
                
        elif event_type == EventTopics.PROGRESS_UPDATE:
            # Dapatkan nilai progres
            if self.progress_key in kwargs:
                current_progress = kwargs[self.progress_key]
                
                # Jika total diberikan, update itu juga
                if self.total_key in kwargs:
                    new_total = kwargs[self.total_key]
                    if new_total != self.tracker.total:
                        self.tracker.set_total(new_total)
                
                # Jika progres adalah persentase (0-1)
                if 0 <= current_progress <= 1:
                    target = int(current_progress * self.tracker.total)
                    increment = target - self.tracker.current
                else:
                    # Progres adalah nilai absolut
                    increment = current_progress - self.tracker.current
                    
                # Update tracker jika ada progres
                if increment > 0:
                    message = kwargs.get('message')
                    self.tracker.update(increment, message)
                    
                # Update metrics jika ada
                if 'metrics' in kwargs and isinstance(kwargs['metrics'], dict):
                    self.tracker.set_metrics(kwargs['metrics'])
                    
        elif event_type == EventTopics.PROGRESS_COMPLETE:
            # Tandai progres sebagai selesai
            message = kwargs.get('message')
            self.tracker.complete(message)
    
    def should_process_event(self, event_type: str) -> bool:
        """Method untuk kompatibilitas dengan BaseObserver."""
        return self.enabled

class ProgressEventEmitter:
    """Emitter untuk mengirim event progres ke EventDispatcher."""
    
    def __init__(
        self,
        total: int = 100,
        description: str = "Progress",
        event_topic_prefix: str = "progress",
        logger = None
    ):
        """
        Inisialisasi ProgressEventEmitter.
        
        Args:
            total: Total nilai progres (100%)
            description: Deskripsi progres
            event_topic_prefix: Prefix untuk event topics
            logger: Logger kustom (opsional)
        """
        self.total = total
        self.description = description
        self.current = 0
        self.event_topic_prefix = event_topic_prefix
        self.logger = logger or get_logger("progress_emitter")
        self.metrics = {}
        
        # Event topics
        self.start_event = f"{event_topic_prefix}_start"
        if self.start_event != EventTopics.PROGRESS_START:
            self.start_event = EventTopics.PROGRESS_START
            
        self.update_event = f"{event_topic_prefix}_update"
        if self.update_event != EventTopics.PROGRESS_UPDATE:
            self.update_event = EventTopics.PROGRESS_UPDATE
            
        self.complete_event = f"{event_topic_prefix}_complete"
        if self.complete_event != EventTopics.PROGRESS_COMPLETE:
            self.complete_event = EventTopics.PROGRESS_COMPLETE
    
    def start(self, description: Optional[str] = None, total: Optional[int] = None) -> None:
        """
        Mulai progres dan kirim event start.
        
        Args:
            description: Deskripsi progres (opsional)
            total: Total nilai progres (opsional)
        """
        if description:
            self.description = description
            
        if total is not None and total > 0:
            self.total = total
            
        self.current = 0
        self.metrics = {}
        
        # Kirim event dengan safe import
        self._notify(
            event_type=self.start_event,
            sender=self,
            description=self.description,
            total=self.total
        )
        
        self.logger.info(f"🚀 {self.description} dimulai")
    
    def update(self, progress: float, message: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update progres dan kirim event update.
        
        Args:
            progress: Nilai progres baru (absolut atau relatif)
            message: Pesan status (opsional)
            metrics: Metrik tambahan (opsional)
        """
        # Tentukan apakah ini nilai relatif (0-1) atau absolut
        if 0 <= progress <= 1:
            # Nilai relatif, konversi ke absolut
            new_current = int(progress * self.total)
        else:
            # Nilai absolut
            new_current = int(progress)
            
        # Update current
        self.current = min(new_current, self.total)
        
        # Update metrics jika disediakan
        if metrics:
            self.metrics.update(metrics)
            
        # Kirim event
        self._notify(
            event_type=self.update_event,
            sender=self,
            progress=self.current,
            total=self.total,
            progress_pct=self.current / self.total if self.total > 0 else 0,
            message=message,
            metrics=self.metrics
        )
    
    def increment(self, increment: int = 1, message: Optional[str] = None, metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Tambah nilai progres dan kirim event update.
        
        Args:
            increment: Nilai tambahan
            message: Pesan status (opsional)
            metrics: Metrik tambahan (opsional)
        """
        self.current = min(self.current + increment, self.total)
        
        # Update metrics jika disediakan
        if metrics:
            self.metrics.update(metrics)
            
        # Kirim event
        self._notify(
            event_type=self.update_event,
            sender=self,
            progress=self.current,
            total=self.total,
            progress_pct=self.current / self.total if self.total > 0 else 0,
            message=message,
            metrics=self.metrics
        )
    
    def complete(self, message: Optional[str] = None, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Tandai progres sebagai selesai dan kirim event complete.
        
        Args:
            message: Pesan selesai (opsional)
            final_metrics: Metrik final (opsional)
        """
        self.current = self.total
        
        # Update metrics jika disediakan
        if final_metrics:
            self.metrics.update(final_metrics)
            
        # Kirim event
        self._notify(
            event_type=self.complete_event,
            sender=self,
            progress=self.current,
            total=self.total,
            progress_pct=1.0,
            message=message,
            metrics=self.metrics
        )
        
        self.logger.info(f"✅ {self.description} selesai")
    
    def _notify(self, event_type: str, sender: Any, **kwargs):
        """Safe notify yang menghindari circular import."""
        try:
            # Import di sini untuk menghindari circular import
            from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
            EventDispatcher.notify(event_type, sender, **kwargs)
        except ImportError:
            # Fallback jika EventDispatcher tidak bisa diimpor
            pass

def create_progress_tracker_observer(
    name: str,
    total: int = 100,
    desc: str = "Progress",
    display: bool = True
) -> Tuple[ProgressTracker, "ProgressObserver"]:
    """
    Buat ProgressTracker dan ProgressObserver yang terintegrasi.
    
    Args:
        name: Nama untuk tracker
        total: Total nilai progres
        desc: Deskripsi progres
        display: Apakah menampilkan progress bar
        
    Returns:
        Tuple (ProgressTracker, ProgressObserver)
    """
    tracker = get_progress_tracker(name, total, desc, display=display)
    observer = ProgressObserver(tracker=tracker, total=total, desc=desc)
    
    # Daftarkan ke EventDispatcher dengan lazy import
    try:
        from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
        for event_type in observer.event_types:
            EventDispatcher.register(event_type, observer)
    except ImportError:
        # Jika EventDispatcher tidak bisa diimpor, log warning
        observer.logger.warning("⚠️ EventDispatcher tidak bisa diimpor, observer tidak didaftarkan")
        
    return tracker, observer

def update_progress(
    callback: Callable, 
    current: int, 
    total: int, 
    message: Optional[str] = None, 
    status: str = 'info', 
    suppress_notify: bool = False,
    **kwargs
) -> None:
    """
    Update progress dengan callback dan notifikasi observer.
    
    Args:
        callback: Callback function untuk progress reporting
        current: Nilai progress saat ini
        total: Nilai total progress
        message: Pesan progress
        status: Status progress ('info', 'success', 'warning', 'error')
        suppress_notify: Jangan kirim notifikasi ke observer
        **kwargs: Parameter tambahan untuk callback
    """
    # Ensure total_files_all is set
    if 'total_files_all' not in kwargs: 
        kwargs['total_files_all'] = total
    
    # Call progress callback jika ada
    if callback:
        callback(
            progress=current, 
            total=total, 
            message=message or f"Progres: {int(current/total*100) if total > 0 else 0}%", 
            status=status, 
            **kwargs
        )
    
    # Notifikasi observer jika tidak disertakan flag suppress_notify
    if not suppress_notify:
        try:
            from smartcash.components.observer import notify
            from smartcash.components.observer.event_topics_observer import EventTopics
            
            # Bersihkan kwargs duplikasi
            notify_kwargs = {k: v for k, v in kwargs.items() if k not in ('current_progress', 'current_total')}
            
            notify(
                event_type=EventTopics.PROGRESS_UPDATE, 
                sender="progress_util", 
                message=message or f"Progres: {int(current/total*100) if total > 0 else 0}%", 
                progress=current, 
                total=total, 
                **notify_kwargs
            )
        except Exception:
            pass