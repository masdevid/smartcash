"""
File: smartcash/handlers/preprocessing/observers/progress_observer.py
Author: Alfrida Sabar
Deskripsi: Observer untuk monitoring progress pipeline preprocessing.
           Menampilkan progress bar dan status menggunakan tqdm.
"""

from typing import Dict, Any, Optional, List
import time
from tqdm.auto import tqdm

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.preprocessing.observers.base_observer import BaseObserver, PipelineEventType


class ProgressObserver(BaseObserver):
    """
    Observer untuk monitoring progress pipeline preprocessing.
    Menampilkan progress bar dan status menggunakan tqdm.
    """
    
    def __init__(
        self, 
        name: str = "ProgressObserver",
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi observer progress.
        
        Args:
            name: Nama observer
            logger: Logger kustom (opsional)
        """
        super().__init__(name)
        self.logger = logger or get_logger(name)
        self.current_progress_bar = None
        self.start_time = None
        self.current_component = None
        self.pipeline_name = None
        self.progress_bars = {}
    
    def on_pipeline_start(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat pipeline dimulai.
        
        Args:
            data: Data event dengan detail pipeline
        """
        self.pipeline_name = data.get('pipeline_name', 'Pipeline')
        self.start_time = time.time()
        
        components = data.get('components', [])
        if components:
            # Tampilkan informasi pipeline dan komponen
            component_names = [comp.get('name', f"Component {i+1}") for i, comp in enumerate(components)]
            self.logger.start(
                f"🚀 Memulai pipeline {self.pipeline_name} dengan "
                f"{len(components)} komponen: {', '.join(component_names)}"
            )
        else:
            self.logger.start(f"🚀 Memulai pipeline {self.pipeline_name}")
    
    def on_pipeline_end(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat pipeline selesai.
        
        Args:
            data: Data event dengan status pipeline
        """
        # Tutup progress bar yang masih terbuka
        self._close_progress_bars()
        
        # Hitung durasi
        elapsed = time.time() - self.start_time if self.start_time else 0
        status = data.get('status', 'completed')
        
        # Log sesuai status
        if status == 'success' or status == 'completed':
            self.logger.success(
                f"✅ Pipeline {self.pipeline_name} selesai dalam {elapsed:.2f} detik"
            )
        elif status == 'error':
            error_message = data.get('error', 'Unknown error')
            self.logger.error(
                f"❌ Pipeline {self.pipeline_name} gagal setelah {elapsed:.2f} detik: {error_message}"
            )
        elif status == 'warning':
            warning_message = data.get('warning', 'Some issues occurred')
            self.logger.warning(
                f"⚠️ Pipeline {self.pipeline_name} selesai dengan warning dalam {elapsed:.2f} detik: {warning_message}"
            )
    
    def on_component_start(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat komponen pipeline dimulai.
        
        Args:
            data: Data event dengan detail komponen
        """
        component_name = data.get('component_name', 'Component')
        self.current_component = component_name
        
        total = data.get('total_items', 0)
        description = data.get('description', f"Memproses {component_name}")
        
        # Buat progress bar baru
        if total > 0:
            progress_bar = tqdm(
                total=total,
                desc=description,
                unit=data.get('unit', 'items'),
                leave=True,
                dynamic_ncols=True
            )
            self.progress_bars[component_name] = progress_bar
        else:
            # Jika tidak ada total, hanya tampilkan log
            self.logger.info(f"🔄 Memulai komponen {component_name}")
    
    def on_component_end(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat komponen pipeline selesai.
        
        Args:
            data: Data event dengan status komponen
        """
        component_name = data.get('component_name', self.current_component)
        status = data.get('status', 'completed')
        elapsed = data.get('elapsed', 0)
        
        # Tutup progress bar
        if component_name in self.progress_bars:
            progress_bar = self.progress_bars[component_name]
            # Pastikan progress bar 100% selesai
            progress_bar.update(progress_bar.total - progress_bar.n)
            progress_bar.close()
            del self.progress_bars[component_name]
        
        # Log sesuai status
        if status == 'success' or status == 'completed':
            self.logger.success(
                f"✅ Komponen {component_name} selesai dalam {elapsed:.2f} detik"
            )
        elif status == 'error':
            error_message = data.get('error', 'Unknown error')
            self.logger.error(
                f"❌ Komponen {component_name} gagal setelah {elapsed:.2f} detik: {error_message}"
            )
        elif status == 'warning':
            warning_message = data.get('warning', 'Some issues occurred')
            self.logger.warning(
                f"⚠️ Komponen {component_name} selesai dengan warning dalam {elapsed:.2f} detik: {warning_message}"
            )
    
    def on_progress_update(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada update progress.
        
        Args:
            data: Data event dengan info progress
        """
        component_name = data.get('component_name', self.current_component)
        increment = data.get('increment', 1)
        current = data.get('current', None)
        total = data.get('total', None)
        
        # Update progress bar jika ada
        if component_name in self.progress_bars:
            progress_bar = self.progress_bars[component_name]
            
            # Update description jika ada info baru
            if 'description' in data:
                progress_bar.set_description(data['description'])
                
            # Update total jika ada
            if total is not None and total != progress_bar.total:
                progress_bar.total = total
            
            # Update progress
            if current is not None:
                # Set posisi absolut
                progress_bar.n = current
                progress_bar.refresh()
            else:
                # Increment
                progress_bar.update(increment)
    
    def on_error(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat terjadi error.
        
        Args:
            data: Data event dengan info error
        """
        component_name = data.get('component_name', self.current_component)
        error_message = data.get('message', 'Unknown error')
        self.logger.error(f"❌ Error pada {component_name}: {error_message}")
    
    def on_warning(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada warning.
        
        Args:
            data: Data event dengan info warning
        """
        component_name = data.get('component_name', self.current_component)
        warning_message = data.get('message', 'Warning')
        self.logger.warning(f"⚠️ Warning pada {component_name}: {warning_message}")
    
    def on_info(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada info.
        
        Args:
            data: Data event dengan info
        """
        component_name = data.get('component_name', self.current_component)
        info_message = data.get('message', '')
        self.logger.info(f"ℹ️ {component_name}: {info_message}")
    
    def on_metric_update(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada update metrik.
        
        Args:
            data: Data event dengan info metrik
        """
        component_name = data.get('component_name', self.current_component)
        metrics = data.get('metrics', {})
        
        # Format metrik untuk logging
        if metrics:
            metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
            self.logger.metric(f"📊 {component_name} - {metrics_str}")
    
    def _close_progress_bars(self) -> None:
        """Tutup semua progress bar yang masih terbuka."""
        for name, progress_bar in list(self.progress_bars.items()):
            progress_bar.close()
            del self.progress_bars[name]