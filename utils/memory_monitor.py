"""
File: smartcash/utils/memory_monitor.py
Author: Alfrida Sabar
Deskripsi: Utilitas untuk memonitor dan mengoptimalkan penggunaan memori di notebook dan browser
"""

import gc
import time
import threading
import weakref
from typing import Dict, List, Optional, Any, Set, Callable
import sys
import os

class MemoryMonitor:
    """
    Utilitas untuk memonitor dan mengoptimalkan penggunaan memori
    di notebook dan membantu mendeteksi memory leak
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Implementasi singleton untuk memastikan hanya ada satu instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MemoryMonitor, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, logger=None, auto_cleanup_interval: int = 300):
        """
        Inisialisasi monitor memori
        
        Args:
            logger: Logger untuk output informasi
            auto_cleanup_interval: Interval waktu untuk auto cleanup (detik)
        """
        # Hindari re-inisialisasi karena singleton
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.logger = logger
        self._tracked_objects = weakref.WeakSet()
        self._object_counts = {}
        self._last_cleanup_time = time.time()
        self._auto_cleanup_interval = auto_cleanup_interval
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self._initialized = True
        
        # Tracking memori
        self._memory_measurements = []
        
        # Log inisialisasi
        if self.logger:
            self.logger.info("🧠 Memory monitor diinisialisasi")
    
    def start_monitoring(self) -> None:
        """Mulai monitoring memori di background thread."""
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            if self.logger:
                self.logger.info("⚠️ Monitoring memori sudah berjalan")
            return
            
        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        if self.logger:
            self.logger.info("🔄 Monitoring memori dimulai")
    
    def stop_monitoring(self) -> None:
        """Hentikan monitoring memori."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            return
            
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=1.0)
        self._monitoring_thread = None
        
        if self.logger:
            self.logger.info("⏹️ Monitoring memori dihentikan")
    
    def _monitoring_loop(self) -> None:
        """Loop utama monitoring memori."""
        while not self._stop_monitoring.is_set():
            try:
                current_time = time.time()
                
                # Ambil data penggunaan memori
                mem_usage = self.get_memory_usage()
                self._memory_measurements.append((current_time, mem_usage))
                
                # Batasi jumlah measurement yang disimpan
                if len(self._memory_measurements) > 100:
                    self._memory_measurements = self._memory_measurements[-100:]
                
                # Check memory growth
                if len(self._memory_measurements) > 5:
                    first_time, first_usage = self._memory_measurements[0]
                    last_time, last_usage = self._memory_measurements[-1]
                    
                    duration = last_time - first_time
                    if duration > 30:  # Minimal 30 detik untuk menghindari false positive
                        usage_diff = {}
                        
                        for key in last_usage:
                            if key in first_usage and isinstance(last_usage[key], (int, float)) and isinstance(first_usage[key], (int, float)):
                                usage_diff[key] = last_usage[key] - first_usage[key]
                        
                        # Check for significant growth
                        if 'python_mem_mb' in usage_diff and usage_diff['python_mem_mb'] > 100:
                            if self.logger:
                                self.logger.warning(f"⚠️ Terdeteksi pertumbuhan memori Python: +{usage_diff['python_mem_mb']:.1f} MB dalam {duration:.1f} detik")
                            
                            # Jalankan garbage collection
                            collected = gc.collect()
                            if self.logger:
                                self.logger.info(f"🧹 Garbage collection: {collected} objek dibersihkan")
                
                # Jalankan auto-cleanup jika waktunya
                if current_time - self._last_cleanup_time > self._auto_cleanup_interval:
                    self.cleanup()
                    self._last_cleanup_time = current_time
                
                # Tunggu interval berikutnya
                time.sleep(10)  # Periksa setiap 10 detik
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"❌ Error dalam loop monitoring: {str(e)}")
                time.sleep(30)  # Jika error, tunggu lebih lama
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Dapatkan informasi penggunaan memori
        
        Returns:
            Dictionary berisi informasi penggunaan memori
        """
        memory_info = {'timestamp': time.time()}
        
        # Python memory info
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info['python_mem_mb'] = process.memory_info().rss / (1024 * 1024)
            memory_info['python_percent'] = process.memory_percent()
            
            # System memory
            system_mem = psutil.virtual_memory()
            memory_info['system_total_gb'] = system_mem.total / (1024**3)
            memory_info['system_used_gb'] = system_mem.used / (1024**3)
            memory_info['system_percent'] = system_mem.percent
        except ImportError:
            memory_info['python_mem_mb'] = 0
        
        # GPU info jika tersedia
        try:
            import torch
            if torch.cuda.is_available():
                memory_info['cuda_allocated_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
                memory_info['cuda_reserved_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
                
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_info['gpu_total_mb'] = info.total / (1024 * 1024)
                    memory_info['gpu_used_mb'] = info.used / (1024 * 1024)
                    memory_info['gpu_percent'] = (info.used / info.total) * 100
                except:
                    pass
        except ImportError:
            pass
            
        # Object tracking
        memory_info['tracked_objects'] = len(self._tracked_objects)
        memory_info['gc_objects'] = len(gc.get_objects())
        
        return memory_info
    
    def track_object(self, obj: Any, name: Optional[str] = None) -> None:
        """
        Lacak objek untuk memantau potensi memory leak.
        
        Args:
            obj: Objek yang akan dilacak
            name: Nama opsional untuk objek tersebut
        """
        self._tracked_objects.add(obj)
        
        # Update statistik
        obj_type = type(obj).__name__
        self._object_counts[obj_type] = self._object_counts.get(obj_type, 0) + 1
    
    def cleanup(self) -> Dict[str, Any]:
        """
        Bersihkan memori dan observer yang tidak terpakai.
        
        Returns:
            Dictionary dengan informasi cleanup
        """
        stats = {
            'timestamp': time.time(),
            'gc_collected': 0,
            'observers_cleaned': 0
        }
        
        # Jalankan garbage collection
        old_count = len(gc.get_objects())
        gc_collected = gc.collect(generation=2)
        stats['gc_collected'] = gc_collected
        
        # Coba bersihkan observer jika tersedia
        try:
            from smartcash.utils.observer.observer_manager import ObserverManager
            observer_manager = ObserverManager()
            # Dapatkan statistik awal
            observer_stats_before = observer_manager.get_stats()
            
            # Bersihkan grup observer yang mungkin tidak digunakan
            groups_to_check = [
                "download_observers", 
                "preprocessing_observers",
                "augmentation_observers", 
                "visualization_observers",
                "dataset_progress",
                "validation_progress",
                "augmentation_progress"
            ]
            
            for group in groups_to_check:
                observer_manager.unregister_group(group)
            
            # Dapatkan statistik akhir
            observer_stats_after = observer_manager.get_stats()
            
            # Hitung observer yang dibersihkan
            before_count = observer_stats_before.get('observer_count', 0)
            after_count = observer_stats_after.get('observer_count', 0)
            stats['observers_cleaned'] = before_count - after_count
        except:
            pass
        
        # Bersihkan tqdm progress bars jika ada
        try:
            from tqdm import tqdm
            original_instances = len(tqdm.get_instances())
            for inst in tqdm.get_instances():
                try:
                    inst.close()
                except:
                    pass
            stats['tqdm_closed'] = original_instances - len(tqdm.get_instances())
        except:
            pass
        
        # Bersihkan cache Matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.close('all')
            stats['matplotlib_figs_closed'] = True
        except:
            pass
        
        # Log hasil cleanup
        if self.logger:
            self.logger.info(f"🧹 Cleanup: {stats['gc_collected']} objek GC, "
                            f"{stats['observers_cleaned']} observer dibersihkan")
        
        return stats
    
    def detect_leaks(self) -> List[Dict[str, Any]]:
        """
        Deteksi potensi memory leak
        
        Returns:
            List dictionary dengan informasi potensi leak
        """
        # Jalankan GC untuk membersihkan objek yang sebenarnya sudah tidak terpakai
        gc.collect()
        
        # Dapatkan objek yang mereferensi objek lain
        leaks = []
        
        # Dapatkan semua objek
        all_objects = gc.get_objects()
        type_counts = {}
        
        # Hitung objek per tipe
        for obj in all_objects:
            obj_type = type(obj).__name__
            type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        
        # Cari tipe objek yang jumlahnya mencurigakan
        suspected_types = []
        for obj_type, count in type_counts.items():
            if count > 1000:  # Threshold yang perlu disesuaikan
                suspected_types.append((obj_type, count))
        
        # Tambahkan info ke hasil
        for obj_type, count in suspected_types:
            leaks.append({
                'type': obj_type,
                'count': count,
                'suspicion_level': 'high' if count > 10000 else 'medium'
            })
        
        # Tambahkan info tentang objek yang dilacak
        for obj_type, count in self._object_counts.items():
            remaining = sum(1 for obj in self._tracked_objects if type(obj).__name__ == obj_type)
            if remaining > 0 and count > 10:
                leaks.append({
                    'type': obj_type,
                    'original_count': count,
                    'remaining_count': remaining,
                    'leak_ratio': remaining / count,
                    'suspicion_level': 'medium' if remaining / count > 0.5 else 'low'
                })
        
        return leaks
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Dapatkan statistik memori lengkap
        
        Returns:
            Dictionary dengan statistik memori
        """
        stats = self.get_memory_usage()
        
        # Tambahkan info tambahan
        stats['gc_stats'] = {
            'garbage': len(gc.garbage),
            'objects': len(gc.get_objects()),
            'tracked_objects': len(self._tracked_objects)
        }
        
        # Deteksi dan tambahkan info leak
        leaks = self.detect_leaks()
        if leaks:
            stats['potential_leaks'] = leaks
        
        # Dapatkan growth info
        if len(self._memory_measurements) > 1:
            first_time, first_usage = self._memory_measurements[0]
            last_time, last_usage = self._memory_measurements[-1]
            
            stats['memory_growth'] = {}
            for key in last_usage:
                if key in first_usage and isinstance(last_usage[key], (int, float)) and isinstance(first_usage[key], (int, float)):
                    growth = last_usage[key] - first_usage[key]
                    stats['memory_growth'][key] = growth
        
        return stats
    
    def print_memory_usage(self) -> None:
        """Cetak informasi penggunaan memori ke konsol."""
        mem_info = self.get_memory_usage()
        
        print(f"===== Penggunaan Memori =====")
        if 'python_mem_mb' in mem_info:
            print(f"🐍 Python: {mem_info['python_mem_mb']:.1f} MB ({mem_info.get('python_percent', 0):.1f}%)")
        
        if 'system_used_gb' in mem_info and 'system_total_gb' in mem_info:
            print(f"💻 Sistem: {mem_info['system_used_gb']:.1f} GB / {mem_info['system_total_gb']:.1f} GB ({mem_info.get('system_percent', 0):.1f}%)")
        
        if 'cuda_allocated_mb' in mem_info:
            print(f"🖥️ CUDA: {mem_info['cuda_allocated_mb']:.1f} MB terpakai, {mem_info.get('cuda_reserved_mb', 0):.1f} MB dicadangkan")
        
        print(f"🔍 Objek: {mem_info.get('gc_objects', 0)} objek total, {mem_info.get('tracked_objects', 0)} dilacak")
        print(f"===========================")
    
    def plot_memory_history(self) -> Any:
        """
        Plot grafik riwayat penggunaan memori
        
        Returns:
            Matplotlib figure atau None jika matplotlib tidak ada
        """
        if not self._memory_measurements:
            return None
            
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from datetime import datetime
            
            # Extract data
            times = [datetime.fromtimestamp(t) for t, _ in self._memory_measurements]
            python_mem = [m.get('python_mem_mb', 0) for _, m in self._memory_measurements]
            system_mem = [m.get('system_used_gb', 0) * 1024 for _, m in self._memory_measurements]  # Convert to MB
            cuda_mem = [m.get('cuda_allocated_mb', 0) for _, m in self._memory_measurements]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(times, python_mem, 'b-', label='Python (MB)')
            
            if any(cuda_mem):
                ax.plot(times, cuda_mem, 'r-', label='CUDA (MB)')
            
            # Add labels
            ax.set_title('Penggunaan Memori')
            ax.set_xlabel('Waktu')
            ax.set_ylabel('Memori (MB)')
            ax.legend()
            ax.grid(True)
            
            # Format x-axis
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
            
        except ImportError:
            return None


# Fungsi global untuk akses mudah
_memory_monitor = None

def get_memory_monitor(logger=None):
    """
    Dapatkan instance MemoryMonitor (singleton)
    
    Args:
        logger: Logger untuk output informasi
        
    Returns:
        Instance MemoryMonitor
    """
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor(logger=logger)
    return _memory_monitor

def start_monitoring(logger=None):
    """
    Mulai monitoring memori dengan fungsi global
    
    Args:
        logger: Logger untuk output informasi
    """
    monitor = get_memory_monitor(logger)
    monitor.start_monitoring()
    return monitor

def cleanup_memory():
    """
    Bersihkan memori dengan fungsi global
    
    Returns:
        Dictionary dengan informasi cleanup
    """
    monitor = get_memory_monitor()
    return monitor.cleanup()

def print_memory_usage():
    """Cetak informasi penggunaan memori dengan fungsi global."""
    monitor = get_memory_monitor()
    monitor.print_memory_usage()

def track_object(obj, name=None):
    """
    Lacak objek untuk memantau memory leak dengan fungsi global
    
    Args:
        obj: Objek yang akan dilacak
        name: Nama opsional untuk objek tersebut
    """
    monitor = get_memory_monitor()
    monitor.track_object(obj, name)

def optimize_notebook_memory():
    """
    Optimasi memori notebook dengan membersihkan caches dan menutup figures
    
    Returns:
        Dictionary dengan informasi optimasi
    """
    stats = {
        'gc_collected': 0,
        'matplotlib_closed': 0,
        'others_cleared': []
    }
    
    # Run GC
    stats['gc_collected'] = gc.collect()
    
    # Matplotlib cleanup
    try:
        import matplotlib.pyplot as plt
        num_figs = len(plt.get_fignums())
        plt.close('all')
        stats['matplotlib_closed'] = num_figs
    except:
        pass
    
    # IPython cleanup
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic('reset_selective', '-f matplotlib_memory')
            stats['others_cleared'].append('ipython_matplotlib')
    except:
        pass
    
    # Pandas cleanup
    try:
        import pandas as pd
        old_versions = len(pd.core.common._files)
        pd.core.common._files = {}
        stats['others_cleared'].append(f'pandas_versions: {old_versions}')
    except:
        pass
    
    # Keras/TF cleanup
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        stats['others_cleared'].append('keras_session')
    except:
        pass
    
    # Observer cleanup
    try:
        from smartcash.utils.observer.observer_manager import ObserverManager
        observer_manager = ObserverManager()
        observer_manager.unregister_all()
        stats['others_cleared'].append('observers')
    except:
        pass
    
    # Final GC
    gc.collect()
    
    return statstou