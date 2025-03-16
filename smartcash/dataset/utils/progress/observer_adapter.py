"""
File: smartcash/dataset/utils/progress/observer_adapter.py
Deskripsi: Adapter untuk integrasi sistem progress dengan observer pattern
"""

from typing import Dict, Any, List, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.components.observer.base_observer import BaseObserver
from smartcash.components.observer.event_dispatcher_observer import EventDispatcher
from smartcash.components.observer.event_topics_observer import EventTopics
from smartcash.dataset.utils.progress.progress_tracker import ProgressTracker


class ProgressObserver(BaseObserver):
    """Observer untuk melacak progres dari event."""
    
    def __init__(
        self, 
        tracker: ProgressTracker, 
        event_types: List[str] = None, 
        total: int = 100,
        progress_key: str = 'progress',
        total_key: str = 'total',
        enabled: bool = True,
        logger=None
    ):
        """
        Inisialisasi ProgressObserver.
        
        Args:
            tracker: ProgressTracker yang akan diupdate
            event_types: List tipe event yang akan diobservasi
            total: Total nilai progres (100%)
            progress_key: Kunci untuk nilai progres dalam event
            total_key: Kunci untuk nilai total dalam event
            enabled: Apakah observer aktif
            logger: Logger kustom (opsional)
        """
        super().__init__(enabled=enabled)
        
        self.tracker = tracker
        self.event_types = event_types or [
            EventTopics.PROGRESS_UPDATE,
            EventTopics.PROGRESS_START,
            EventTopics.PROGRESS_COMPLETE
        ]
        self.total = total
        self.progress_key = progress_key
        self.total_key = total_key
        self.logger = logger or get_logger("progress_observer")
        
        # Setup tracker jika belum diset
        if self.tracker.total <= 0:
            self.tracker.set_total(self.total)
    
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
                self.total = kwargs[self.total_key]
                self.tracker.set_total(self.total)
            
            # Set deskripsi jika disediakan
            if 'description' in kwargs:
                self.tracker.desc = kwargs['description']
                if self.tracker.pbar:
                    self.tracker.pbar.set_description(kwargs['description'])
                    
            # Reset tracker
            self.tracker.current = 0
            self.tracker.is_completed = False
            
            # Log start
            self.logger.info(f"🚀 {self.tracker.desc} dimulai")
                
        elif event_type == EventTopics.PROGRESS_UPDATE:
            # Dapatkan nilai progres
            if self.progress_key in kwargs:
                current_progress = kwargs[self.progress_key]
                
                # Jika total diberikan, update itu juga
                if self.total_key in kwargs:
                    new_total = kwargs[self.total_key]
                    if new_total != self.total:
                        self.total = new_total
                        self.tracker.set_total(self.total)
                
                # Jika progres adalah persentase (0-1)
                if 0 <= current_progress <= 1:
                    target = int(current_progress * self.total)
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


class ProgressEventEmitter:
    """Emitter untuk mengirim event progres ke EventDispatcher."""
    
    def __init__(
        self,
        total: int = 100,
        description: str = "Progress",
        event_topic_prefix: str = "progress",
        logger=None
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
        
        # Kirim event
        EventDispatcher.notify(
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
        EventDispatcher.notify(
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
        EventDispatcher.notify(
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
        EventDispatcher.notify(
            event_type=self.complete_event,
            sender=self,
            progress=self.current,
            total=self.total,
            progress_pct=1.0,
            message=message,
            metrics=self.metrics
        )
        
        self.logger.info(f"✅ {self.description} selesai")


def create_progress_tracker_for_observer(
    total: int = 100,
    desc: str = "Progress",
    display: bool = True
) -> Tuple[ProgressTracker, ProgressObserver]:
    """
    Buat ProgressTracker dan ProgressObserver yang terintegrasi.
    
    Args:
        total: Total nilai progres
        desc: Deskripsi progres
        display: Apakah menampilkan progress bar
        
    Returns:
        Tuple (ProgressTracker, ProgressObserver)
    """
    tracker = ProgressTracker(total=total, desc=desc, display=display)
    observer = ProgressObserver(tracker=tracker, total=total)
    
    # Daftarkan ke EventDispatcher
    for event_type in observer.event_types:
        EventDispatcher.register(event_type, observer)
        
    return tracker, observer