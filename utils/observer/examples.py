# File: smartcash/utils/observer/examples.py
# Author: Alfrida Sabar
# Deskripsi: Contoh implementasi dan penggunaan observer pattern di SmartCash

from typing import Dict, Any, List, Optional, Callable, Union
import threading
import time
from tqdm import tqdm

from smartcash.utils.observer.base_observer import BaseObserver
from smartcash.utils.observer.event_dispatcher import EventDispatcher
from smartcash.utils.observer.observer_manager import ObserverManager
from smartcash.utils.observer.decorators import observable, observe
from smartcash.utils.observer import EventTopics
from smartcash.utils.logger import get_logger


# Contoh 1: Observer dasar untuk training
class TrainingObserver(BaseObserver):
    """Observer untuk monitoring proses training."""
    
    def __init__(self, name: str = "TrainingObserver", priority: int = 0):
        super().__init__(name=name, priority=priority)
        self.logger = get_logger(name)
        self.epoch_start_time = None
        self.training_start_time = None
        self.metrics = {}
    
    def update(self, event_type: str, sender: Any, **kwargs):
        """Implementasi metode update."""
        # Training events
        if event_type == EventTopics.TRAINING_START:
            self.on_training_start(sender, **kwargs)
        elif event_type == EventTopics.TRAINING_END:
            self.on_training_end(sender, **kwargs)
        elif event_type == EventTopics.EPOCH_START:
            self.on_epoch_start(sender, **kwargs)
        elif event_type == EventTopics.EPOCH_END:
            self.on_epoch_end(sender, **kwargs)
        elif event_type == EventTopics.BATCH_END:
            self.on_batch_end(sender, **kwargs)
        elif event_type == EventTopics.VALIDATION_END:
            self.on_validation_end(sender, **kwargs)
    
    def on_training_start(self, sender: Any, **kwargs):
        self.training_start_time = time.time()
        self.logger.info(f"🏃 Training dimulai dengan {kwargs.get('epochs', '?')} epochs")
    
    def on_training_end(self, sender: Any, **kwargs):
        if self.training_start_time:
            duration = time.time() - self.training_start_time
            self.logger.info(f"🏁 Training selesai dalam {duration:.2f} detik")
            
            # Reset state
            self.training_start_time = None
            self.metrics = {}
    
    def on_epoch_start(self, sender: Any, **kwargs):
        self.epoch_start_time = time.time()
        epoch = kwargs.get('epoch', '?')
        self.logger.info(f"⏳ Epoch {epoch} dimulai")
    
    def on_epoch_end(self, sender: Any, **kwargs):
        epoch = kwargs.get('epoch', '?')
        metrics = kwargs.get('metrics', {})
        
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            self.logger.info(
                f"✅ Epoch {epoch} selesai dalam {duration:.2f} detik "
                f"dengan metrics: {metrics}"
            )
            
            # Simpan metrics terakhir
            self.metrics = metrics
            self.epoch_start_time = None
    
    def on_batch_end(self, sender: Any, **kwargs):
        # Batch events biasanya sangat sering, jadi hanya log pada level debug
        batch = kwargs.get('batch', '?')
        self.logger.debug(f"Batch {batch} selesai")
    
    def on_validation_end(self, sender: Any, **kwargs):
        metrics = kwargs.get('metrics', {})
        self.logger.info(f"📊 Validasi selesai dengan metrics: {metrics}")


# Contoh 2: Progress Observer dengan tqdm
class TqdmProgressObserver(BaseObserver):
    """Observer untuk menampilkan progress bar menggunakan tqdm."""
    
    def __init__(
        self, 
        name: str = "ProgressObserver", 
        priority: int = 0,
        total: Optional[int] = None,
        desc: str = "Progress",
        update_interval: float = 0.1
    ):
        super().__init__(name=name, priority=priority)
        self.total = total
        self.desc = desc
        self.update_interval = update_interval
        self.progress = 0
        self.pbar = None
        self.last_update_time = 0
    
    def update(self, event_type: str, sender: Any, **kwargs):
        # Buat progress bar jika belum ada
        if self.pbar is None and self.total is not None:
            self.pbar = tqdm(total=self.total, desc=self.desc)
        
        # Periksa jika ini adalah event progress
        if 'progress' in kwargs:
            progress = kwargs['progress']
            
            # Update nilai total jika ada
            if 'total' in kwargs and self.total != kwargs['total']:
                self.total = kwargs['total']
                if self.pbar:
                    self.pbar.total = self.total
            
            # Hitung increment
            increment = progress - self.progress
            self.progress = progress
            
            # Update progress bar dengan throttling
            current_time = time.time()
            if (increment > 0 and 
                (current_time - self.last_update_time >= self.update_interval or 
                 progress >= self.total or progress <= 0)):
                if self.pbar:
                    self.pbar.update(increment)
                self.last_update_time = current_time
        
        # Handle event khusus
        elif event_type.endswith('.start'):
            # Reset progress di event start
            self.progress = 0
            if self.pbar:
                self.pbar.reset()
        elif event_type.endswith('.end'):
            # Tutup progress bar di event end
            if self.pbar:
                self.pbar.close()
                self.pbar = None
                self.progress = 0
    
    def __del__(self):
        """Close progress bar when observer is destroyed."""
        if hasattr(self, 'pbar') and self.pbar is not None:
            self.pbar.close()


# Contoh 3: Metrics Collector Observer
class MetricsCollectorObserver(BaseObserver):
    """Observer untuk mengumpulkan metrics dari berbagai event."""
    
    def __init__(
        self, 
        name: str = "MetricsCollector", 
        priority: int = 0
    ):
        super().__init__(name=name, priority=priority)
        self.logger = get_logger(name)
        self.lock = threading.RLock()
        
        # Dictionary untuk menyimpan semua metrics
        self.all_metrics = {}
        
        # Dictionary untuk menyimpan history metrics per epoch
        self.epoch_metrics = {}
        
        # Dictionary untuk menyimpan metrics terbaik
        self.best_metrics = {}
    
    def update(self, event_type: str, sender: Any, **kwargs):
        metrics = kwargs.get('metrics', {})
        
        # Skip jika tidak ada metrics
        if not metrics:
            return
        
        with self.lock:
            # Simpan all metrics
            self.all_metrics.update(metrics)
            
            # Handle epoch metrics jika ini adalah event epoch
            if event_type == EventTopics.EPOCH_END and 'epoch' in kwargs:
                epoch = kwargs['epoch']
                self.epoch_metrics[epoch] = metrics.copy()
                
                # Update best metrics
                for metric_name, metric_value in metrics.items():
                    if metric_name not in self.best_metrics:
                        self.best_metrics[metric_name] = (metric_value, epoch)
                    else:
                        # Asumsi nilai lebih tinggi = lebih baik (ganti sesuai kebutuhan)
                        if metric_value > self.best_metrics[metric_name][0]:
                            self.best_metrics[metric_name] = (metric_value, epoch)
    
    def get_last_metrics(self) -> Dict[str, Any]:
        """Mendapatkan metrics terakhir."""
        with self.lock:
            return self.all_metrics.copy()
    
    def get_epoch_metrics(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """Mendapatkan metrics untuk epoch tertentu atau semua epoch."""
        with self.lock:
            if epoch is not None:
                return self.epoch_metrics.get(epoch, {}).copy()
            else:
                return {k: v.copy() for k, v in self.epoch_metrics.items()}
    
    def get_best_metrics(self) -> Dict[str, tuple]:
        """Mendapatkan metrics terbaik dengan epoch."""
        with self.lock:
            return self.best_metrics.copy()
    
    def reset(self) -> None:
        """Reset semua metrics."""
        with self.lock:
            self.all_metrics.clear()
            self.epoch_metrics.clear()
            self.best_metrics.clear()


# Contoh 4: Kelas dengan decorator @observable
class TrainingManager:
    """Contoh kelas dengan metode yang observable."""
    
    def __init__(self, model, dataset, epochs: int = 10):
        self.model = model
        self.dataset = dataset
        self.epochs = epochs
        self.current_epoch = 0
        self.logger = get_logger("training_manager")
    
    @observable(event_type=EventTopics.TRAINING_START, include_args=True)
    def start_training(self, optimizer=None, scheduler=None):
        """Mulai proses training."""
        self.logger.info(f"🏃 Memulai training untuk {self.epochs} epochs")
        
        # Notifikasi akan dikirim otomatis oleh decorator @observable
        
        # Lanjutkan dengan proses training
        self._run_training_loop(optimizer, scheduler)
        
        return {"status": "success"}
    
    @observable(event_type=EventTopics.EPOCH_START)
    def _start_epoch(self, epoch):
        """Mulai epoch baru."""
        self.current_epoch = epoch
        self.logger.info(f"⏳ Memulai epoch {epoch}/{self.epochs}")
        
        # Notifikasi akan dikirim otomatis
    
    @observable(event_type=EventTopics.EPOCH_END, include_result=True)
    def _end_epoch(self, epoch, metrics=None):
        """Akhiri epoch dengan metrics."""
        self.logger.info(f"✅ Epoch {epoch}/{self.epochs} selesai dengan metrics: {metrics}")
        
        # Notifikasi akan dikirim otomatis
        return metrics
    
    @observable(event_type=EventTopics.TRAINING_END, include_result=True)
    def _end_training(self, metrics=None):
        """Akhiri proses training."""
        self.logger.info(f"🏁 Training selesai dengan metrics akhir: {metrics}")
        
        # Notifikasi akan dikirim otomatis
        return {"status": "success", "metrics": metrics}
    
    def _run_training_loop(self, optimizer, scheduler):
        """Jalankan training loop (implementasi sederhana untuk contoh)."""
        try:
            final_metrics = {}
            
            for epoch in range(1, self.epochs + 1):
                # Mulai epoch
                self._start_epoch(epoch)
                
                # Simulasi training
                time.sleep(0.2)  # Simulasi waktu komputasi
                
                # Buat dummy metrics
                epoch_metrics = {
                    "loss": 1.0 - (epoch / self.epochs) * 0.8,
                    "accuracy": 0.5 + (epoch / self.epochs) * 0.4,
                    "val_loss": 1.2 - (epoch / self.epochs) * 0.7,
                    "val_accuracy": 0.4 + (epoch / self.epochs) * 0.45
                }
                
                # Akhiri epoch
                self._end_epoch(epoch, metrics=epoch_metrics)
                
                # Update final metrics
                final_metrics = epoch_metrics
            
            # Akhiri training
            self._end_training(metrics=final_metrics)
            
        except Exception as e:
            self.logger.error(f"❌ Error saat training: {str(e)}")
            raise


# Contoh 5: Kelas dengan decorator @observe
@observe(event_types=[EventTopics.TRAINING_START, EventTopics.TRAINING_END, 
                      EventTopics.EPOCH_START, EventTopics.EPOCH_END])
class TrainingMonitor:
    """
    Contoh kelas yang menggunakan decorator @observe untuk
    memonitor events training.
    """
    
    def __init__(self):
        self.logger = get_logger("training_monitor")
        self.is_monitoring = False
        self.start_time = None
        self.epoch_times = {}
    
    def update(self, event_type, sender, **kwargs):
        """
        Method ini akan dipanggil otomatis oleh observer
        saat event terjadi.
        """
        if event_type == EventTopics.TRAINING_START:
            self.start_time = time.time()
            self.is_monitoring = True
            self.logger.info("🔍 Training monitor aktif")
            
        elif event_type == EventTopics.TRAINING_END:
            if self.start_time:
                duration = time.time() - self.start_time
                self.logger.info(f"🔍 Training selesai dalam {duration:.2f} detik")
                
                # Reset state
                self.is_monitoring = False
                self.start_time = None
                self.epoch_times = {}
                
        elif event_type == EventTopics.EPOCH_START:
            epoch = kwargs.get('epoch', '?')
            self.epoch_times[epoch] = time.time()
            
        elif event_type == EventTopics.EPOCH_END:
            epoch = kwargs.get('epoch', '?')
            metrics = kwargs.get('metrics', {})
            
            if epoch in self.epoch_times:
                duration = time.time() - self.epoch_times[epoch]
                self.logger.info(
                    f"🔍 Epoch {epoch} selesai dalam {duration:.2f} detik "
                    f"dengan accuracy: {metrics.get('accuracy', '?'):.4f}"
                )


# Contoh 6: Contoh penggunaan ObserverManager untuk membuat berbagai jenis observer
def create_standard_observers():
    """Membuat set standar observer untuk aplikasi SmartCash."""
    manager = ObserverManager()
    
    # 1. Observer untuk training
    training_observer = manager.create_observer(
        observer_class=TrainingObserver,
        event_type=[
            EventTopics.TRAINING_START,
            EventTopics.TRAINING_END,
            EventTopics.EPOCH_START,
            EventTopics.EPOCH_END,
            EventTopics.VALIDATION_END
        ],
        group="training"
    )
    
    # 2. Observer untuk progress
    progress_observer = manager.create_observer(
        observer_class=TqdmProgressObserver,
        event_type=[
            EventTopics.PREPROCESSING_PROGRESS,
            EventTopics.DETECTION_PROGRESS
        ],
        total=100,
        desc="Processing",
        group="progress"
    )
    
    # 3. Observer untuk metrics collection
    metrics_observer = manager.create_observer(
        observer_class=MetricsCollectorObserver,
        event_type=[
            EventTopics.EPOCH_END,
            EventTopics.VALIDATION_END,
            EventTopics.EVALUATION_END
        ],
        group="metrics"
    )
    
    # 4. Observer untuk logging
    logging_observer = manager.create_simple_observer(
        event_type=EventTopics.get_all_topics(),
        callback=lambda event_type, sender, **kwargs: print(f"Event: {event_type}"),
        name="SimpleLogger",
        group="logging"
    )
    
    # Kembalikan semua observer
    return {
        'training': training_observer,
        'progress': progress_observer,
        'metrics': metrics_observer,
        'logging': logging_observer
    }


# Contoh 7: Penggunaan praktis dengan kasus nyata
def example_usage():
    """
    Contoh penggunaan observer pattern dalam kasus nyata.
    Ini menggambarkan bagaimana SmartCash dapat menggunakan
    observer pattern di beberapa komponen sekaligus.
    """
    # Buat manager
    manager = ObserverManager()
    
    # Buat set observer standar
    observers = create_standard_observers()
    
    # Buat training manager
    trainer = TrainingManager(model=None, dataset=None, epochs=5)
    
    # Monitor yang otomatis mendengarkan event training
    monitor = TrainingMonitor()
    
    # Jalankan training (akan mengirim event via @observable)
    trainer.start_training()
    
    # Unregister semua observer saat selesai
    manager.unregister_all()
    
    # Shutdown event dispatcher sebelum program berakhir
    EventDispatcher.shutdown()