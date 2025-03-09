"""
File: smartcash/handlers/preprocessing/pipeline/preprocessing_pipeline.py
Author: Alfrida Sabar
Deskripsi: Pipeline preprocessing dasar yang menggabungkan komponen-komponen preprocessing
           dengan observer pattern untuk monitoring progress.
"""

from typing import Dict, Any, Optional, List, Union, Type
import time
from pathlib import Path

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.handlers.preprocessing.core.preprocessing_component import PreprocessingComponent
from smartcash.handlers.preprocessing.observers.base_observer import BaseObserver, PipelineEventType


class PreprocessingPipeline:
    """
    Pipeline preprocessing dasar yang menggabungkan komponen-komponen preprocessing
    dengan observer pattern untuk monitoring progress.
    """
    
    def __init__(
        self, 
        name: str = "PreprocessingPipeline",
        logger: Optional[SmartCashLogger] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Inisialisasi pipeline preprocessing.
        
        Args:
            name: Nama pipeline
            logger: Logger kustom (opsional)
            config: Konfigurasi pipeline (opsional)
        """
        self.name = name
        self.logger = logger or get_logger(name)
        self.config = config or {}
        self.components = []
        self.observers = []
        self.results = {}
    
    def add_component(self, component: PreprocessingComponent) -> 'PreprocessingPipeline':
        """
        Tambahkan komponen ke pipeline.
        
        Args:
            component: Komponen preprocessing
            
        Returns:
            PreprocessingPipeline: Self (untuk method chaining)
        """
        self.components.append(component)
        return self
    
    def add_observer(self, observer: BaseObserver) -> 'PreprocessingPipeline':
        """
        Tambahkan observer ke pipeline.
        
        Args:
            observer: Observer
            
        Returns:
            PreprocessingPipeline: Self (untuk method chaining)
        """
        self.observers.append(observer)
        return self
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Jalankan pipeline preprocessing.
        
        Args:
            **kwargs: Parameter tambahan untuk komponen
            
        Returns:
            Dict[str, Any]: Hasil pipeline
        """
        start_time = time.time()
        
        # Notifikasi pipeline dimulai
        self._notify_observers(PipelineEventType.PIPELINE_START, {
            'pipeline_name': self.name,
            'components': [{'name': comp.__class__.__name__} for comp in self.components],
            'start_time': start_time,
            'kwargs': kwargs
        })
        
        # Inisialisasi hasil
        pipeline_results = {
            'name': self.name,
            'start_time': start_time,
            'components': [],
            'status': 'running'
        }
        
        try:
            # Jalankan setiap komponen
            for i, component in enumerate(self.components):
                component_name = component.__class__.__name__
                component_start_time = time.time()
                
                # Notifikasi komponen dimulai
                self._notify_observers(PipelineEventType.COMPONENT_START, {
                    'component_name': component_name,
                    'component_index': i,
                    'pipeline_name': self.name,
                    'start_time': component_start_time,
                    'description': f"Memproses {component_name}"
                })
                
                try:
                    # Jalankan komponen
                    component_result = component.execute(**kwargs)
                    component_status = 'success'
                    component_elapsed = time.time() - component_start_time
                    
                    # Notifikasi komponen selesai
                    self._notify_observers(PipelineEventType.COMPONENT_END, {
                        'component_name': component_name,
                        'component_index': i,
                        'pipeline_name': self.name,
                        'status': component_status,
                        'elapsed': component_elapsed,
                        'result': component_result
                    })
                    
                    # Tambahkan hasil komponen ke hasil pipeline
                    pipeline_results['components'].append({
                        'name': component_name,
                        'status': component_status,
                        'elapsed': component_elapsed,
                        'result': component_result
                    })
                    
                except Exception as e:
                    # Komponen gagal
                    component_status = 'error'
                    component_elapsed = time.time() - component_start_time
                    error_message = str(e)
                    
                    # Notifikasi error
                    self._notify_observers(PipelineEventType.ERROR, {
                        'component_name': component_name,
                        'message': error_message,
                        'exception': e
                    })
                    
                    # Notifikasi komponen selesai (dengan error)
                    self._notify_observers(PipelineEventType.COMPONENT_END, {
                        'component_name': component_name,
                        'component_index': i,
                        'pipeline_name': self.name,
                        'status': component_status,
                        'elapsed': component_elapsed,
                        'error': error_message
                    })
                    
                    # Tambahkan hasil komponen ke hasil pipeline
                    pipeline_results['components'].append({
                        'name': component_name,
                        'status': component_status,
                        'elapsed': component_elapsed,
                        'error': error_message
                    })
                    
                    # Pipeline juga dianggap gagal
                    pipeline_results['status'] = 'error'
                    pipeline_results['error'] = error_message
                    break
            
            # Jika semua komponen berhasil dan status belum diubah
            if pipeline_results['status'] == 'running':
                pipeline_results['status'] = 'success'
                
        except Exception as e:
            # Pipeline gagal di luar komponen
            error_message = str(e)
            pipeline_results['status'] = 'error'
            pipeline_results['error'] = error_message
            
            # Notifikasi error
            self._notify_observers(PipelineEventType.ERROR, {
                'component_name': self.name,
                'message': error_message,
                'exception': e
            })
            
        finally:
            # Hitung waktu total
            end_time = time.time()
            elapsed = end_time - start_time
            pipeline_results['end_time'] = end_time
            pipeline_results['elapsed'] = elapsed
            
            # Notifikasi pipeline selesai
            self._notify_observers(PipelineEventType.PIPELINE_END, {
                'pipeline_name': self.name,
                'status': pipeline_results['status'],
                'elapsed': elapsed,
                'results': pipeline_results
            })
            
            return pipeline_results
    
    def _notify_observers(self, event_type: PipelineEventType, data: Dict[str, Any]) -> None:
        """
        Notifikasi semua observer tentang event.
        
        Args:
            event_type: Tipe event
            data: Data event
        """
        for observer in self.observers:
            try:
                observer.update(event_type, data)
            except Exception as e:
                self.logger.error(f"❌ Error pada observer {observer.name}: {str(e)}")