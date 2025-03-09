"""
File: smartcash/handlers/preprocessing/observers/base_observer.py
Author: Alfrida Sabar
Deskripsi: Kelas dasar untuk observer pattern yang digunakan dalam monitoring
           pipeline preprocessing. Observer akan menerima notifikasi tentang
           progress dan status pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum


class PipelineEventType(Enum):
    """Tipe event pada pipeline."""
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    COMPONENT_START = "component_start"
    COMPONENT_END = "component_end"
    PROGRESS_UPDATE = "progress_update"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    METRIC_UPDATE = "metric_update"


class BaseObserver(ABC):
    """
    Kelas dasar untuk observer pattern dalam pipeline preprocessing.
    Observer akan menerima notifikasi tentang progress dan status pipeline.
    """
    
    def __init__(self, name: str = "BaseObserver"):
        """
        Inisialisasi observer.
        
        Args:
            name: Nama observer
        """
        self.name = name
        self.enabled = True
    
    def update(self, event_type: PipelineEventType, data: Dict[str, Any]) -> None:
        """
        Terima notifikasi dari pipeline. Method ini akan memanggil method spesifik
        sesuai tipe event.
        
        Args:
            event_type: Tipe event
            data: Data event
        """
        if not self.enabled:
            return
            
        if event_type == PipelineEventType.PIPELINE_START:
            self.on_pipeline_start(data)
        elif event_type == PipelineEventType.PIPELINE_END:
            self.on_pipeline_end(data)
        elif event_type == PipelineEventType.COMPONENT_START:
            self.on_component_start(data)
        elif event_type == PipelineEventType.COMPONENT_END:
            self.on_component_end(data)
        elif event_type == PipelineEventType.PROGRESS_UPDATE:
            self.on_progress_update(data)
        elif event_type == PipelineEventType.ERROR:
            self.on_error(data)
        elif event_type == PipelineEventType.WARNING:
            self.on_warning(data)
        elif event_type == PipelineEventType.INFO:
            self.on_info(data)
        elif event_type == PipelineEventType.METRIC_UPDATE:
            self.on_metric_update(data)
    
    def enable(self) -> None:
        """Aktifkan observer."""
        self.enabled = True
    
    def disable(self) -> None:
        """Nonaktifkan observer."""
        self.enabled = False
    
    @abstractmethod
    def on_pipeline_start(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat pipeline dimulai.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_pipeline_end(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat pipeline selesai.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_component_start(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat komponen pipeline dimulai.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_component_end(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat komponen pipeline selesai.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_progress_update(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada update progress.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_error(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat terjadi error.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_warning(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada warning.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_info(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada info.
        
        Args:
            data: Data event
        """
        pass
    
    @abstractmethod
    def on_metric_update(self, data: Dict[str, Any]) -> None:
        """
        Panggil saat ada update metrik.
        
        Args:
            data: Data event
        """
        pass