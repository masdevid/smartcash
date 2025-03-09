# File: smartcash/handlers/evaluation/observers/progress_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer untuk monitoring progres evaluasi dengan tqdm

import time
from typing import Dict, Any, Optional
from tqdm.auto import tqdm

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.evaluation.observers.base_observer import BaseObserver

class ProgressObserver(BaseObserver):
    """
    Observer untuk monitoring progres evaluasi.
    Menampilkan progress bar dan informasi runtime.
    """
    
    def __init__(
        self, 
        name: str = "ProgressObserver",
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi progress observer.
        
        Args:
            name: Nama observer
            logger: Logger kustom (opsional)
        """
        super().__init__(name)
        self.logger = logger or get_logger("progress_observer")
        
        # Progress tracking
        self.progress_bars = {}
        self.start_times = {}
        self.total_steps = {}
        self.current_steps = {}
        
        # Status flag
        self.is_running = False
    
    def update(self, event: str, data: Dict[str, Any] = None):
        """
        Update dari pipeline evaluasi.
        
        Args:
            event: Nama event
            data: Data tambahan (opsional)
        """
        data = data or {}
        
        # Handle berbagai event
        if event == 'pipeline_start':
            self._handle_pipeline_start(data)
        elif event == 'pipeline_complete':
            self._handle_pipeline_complete(data)
        elif event == 'pipeline_error':
            self._handle_pipeline_error(data)
        elif event == 'component_start':
            self._handle_component_start(data)
        elif event == 'component_complete':
            self._handle_component_complete(data)
        elif event == 'batch_start':
            self._handle_batch_start(data)
        elif event == 'batch_complete':
            self._handle_batch_complete(data)
        elif event == 'research_start':
            self._handle_research_start(data)
        elif event == 'research_complete':
            self._handle_research_complete(data)
        elif event == 'batch_process':
            self._handle_batch_process(data)
        elif event == 'evaluation_dataloader_ready':
            self._handle_evaluation_dataloader_ready(data)
        elif event == 'batch_start':
            self._handle_batch_start(data)
        elif event == 'batch_complete':
            self._handle_batch_complete(data)
    
    def _handle_pipeline_start(self, data: Dict[str, Any]):
        """
        Handle pipeline_start event.
        
        Args:
            data: Event data
        """
        pipeline_name = data.get('pipeline_name', 'Pipeline')
        self.start_times[pipeline_name] = data.get('start_time', time.time())
        self.is_running = True
        
        # Log info
        self.logger.info(f"⏳ Memulai {pipeline_name} evaluasi")
    
    def _handle_pipeline_complete(self, data: Dict[str, Any]):
        """
        Handle pipeline_complete event.
        
        Args:
            data: Event data
        """
        pipeline_name = data.get('pipeline_name', 'Pipeline')
        execution_time = data.get('execution_time', 0)
        
        # Close progress bar jika ada
        if pipeline_name in self.progress_bars:
            self.progress_bars[pipeline_name].close()
            del self.progress_bars[pipeline_name]
        
        # Reset status
        self.is_running = False
        
        # Log info
        self.logger.success(f"✅ {pipeline_name} selesai dalam {execution_time:.2f}s")
    
    def _handle_pipeline_error(self, data: Dict[str, Any]):
        """
        Handle pipeline_error event.
        
        Args:
            data: Event data
        """
        pipeline_name = data.get('pipeline_name', 'Pipeline')
        error = data.get('error', 'Unknown error')
        
        # Close progress bar jika ada
        if pipeline_name in self.progress_bars:
            self.progress_bars[pipeline_name].close()
            del self.progress_bars[pipeline_name]
        
        # Reset status
        self.is_running = False
        
        # Log error
        self.logger.error(f"❌ {pipeline_name} gagal: {error}")
    
    def _handle_component_start(self, data: Dict[str, Any]):
        """
        Handle component_start event.
        
        Args:
            data: Event data
        """
        component_name = data.get('component_name', 'Component')
        
        # Log info
        self.logger.info(f"▶️ Menjalankan komponen: {component_name}")
    
    def _handle_component_complete(self, data: Dict[str, Any]):
        """
        Handle component_complete event.
        
        Args:
            data: Event data
        """
        component_name = data.get('component_name', 'Component')
        execution_time = data.get('execution_time', 0)
        
        # Log info
        self.logger.info(f"✅ Komponen {component_name} selesai dalam {execution_time:.2f}s")
    
    def _handle_batch_start(self, data: Dict[str, Any]):
        """
        Handle batch_start event.
        
        Args:
            data: Event data
        """
        num_models = data.get('num_models', 0)
        dataset_path = data.get('dataset_path', 'unknown')
        
        # Setup progress bar
        self.progress_bars['batch'] = tqdm(total=num_models, desc="Evaluasi Model", unit="model")
        self.total_steps['batch'] = num_models
        self.current_steps['batch'] = 0
        
        # Log info
        self.logger.info(f"🚀 Memulai evaluasi batch untuk {num_models} model")
    
    def _handle_batch_complete(self, data: Dict[str, Any]):
        """
        Handle batch_complete event.
        
        Args:
            data: Event data
        """
        # Close progress bar
        if 'batch' in self.progress_bars:
            self.progress_bars['batch'].close()
            del self.progress_bars['batch']
        
        # Log summary
        summary = data.get('summary', {})
        if 'best_model' in summary:
            self.logger.success(f"🏆 Model terbaik: {summary['best_model']} (mAP={summary.get('best_map', 0):.4f})")
    
    def _handle_research_start(self, data: Dict[str, Any]):
        """
        Handle research_start event.
        
        Args:
            data: Event data
        """
        num_scenarios = data.get('num_scenarios', 0)
        
        # Setup progress bar
        self.progress_bars['research'] = tqdm(total=num_scenarios, desc="Evaluasi Skenario", unit="skenario")
        self.total_steps['research'] = num_scenarios
        self.current_steps['research'] = 0
        
        # Log info
        self.logger.info(f"🔬 Memulai evaluasi {num_scenarios} skenario penelitian")
    
    def _handle_research_complete(self, data: Dict[str, Any]):
        """
        Handle research_complete event.
        
        Args:
            data: Event data
        """
        # Close progress bar
        if 'research' in self.progress_bars:
            self.progress_bars['research'].close()
            del self.progress_bars['research']
        
        # Log summary
        summary = data.get('summary', {})
        if 'best_scenario' in summary:
            self.logger.success(f"🏆 Skenario terbaik: {summary['best_scenario']} (mAP={summary.get('best_map', 0):.4f})")
    
    def _handle_batch_process(self, data: Dict[str, Any]):
        """
        Handle batch_process event.
        
        Args:
            data: Event data
        """
        # Update progress bar
        if 'batch' in self.progress_bars:
            self.current_steps['batch'] += 1
            self.progress_bars['batch'].update(1)
            
            # Add postfix
            if 'metrics' in data:
                metrics = data['metrics']
                self.progress_bars['batch'].set_postfix({
                    'mAP': f"{metrics.get('mAP', 0):.4f}",
                    'F1': f"{metrics.get('f1', 0):.4f}"
                })
    
    def _handle_evaluation_dataloader_ready(self, data: Dict[str, Any]):
        """
        Handle evaluation_dataloader_ready event.
        
        Args:
            data: Event data
        """
        num_batches = data.get('num_batches', 0)
        num_samples = data.get('num_samples', 0)
        
        # Setup progress bar untuk batch evaluation
        if num_batches > 0:
            self.progress_bars['evaluation'] = tqdm(total=num_batches, desc="Evaluasi", unit="batch")
            self.total_steps['evaluation'] = num_batches
            self.current_steps['evaluation'] = 0
            
            # Log info
            self.logger.info(f"🔄 Dataloader siap dengan {num_batches} batch ({num_samples} sampel)")
    
    def _handle_batch_start(self, data: Dict[str, Any]):
        """
        Handle batch_start event.
        
        Args:
            data: Event data
        """
        batch_idx = data.get('batch_idx', 0)
        batch_size = data.get('batch_size', 0)
        
        # Update progress bar
        if 'evaluation' in self.progress_bars:
            # Jika batch_idx = 0, ini adalah batch pertama
            if batch_idx == 0:
                # Tidak perlu update karena baru mulai
                pass
            else:
                # Update progress bar
                self.progress_bars['evaluation'].update(1)
                self.current_steps['evaluation'] = batch_idx
    
    def _handle_batch_complete(self, data: Dict[str, Any]):
        """
        Handle batch_complete event.
        
        Args:
            data: Event data
        """
        batch_idx = data.get('batch_idx', 0)
        inference_time = data.get('inference_time', 0)
        
        # Update progress bar
        if 'evaluation' in self.progress_bars:
            # Update postfix dengan waktu
            fps = 1.0 / inference_time if inference_time > 0 else 0
            self.progress_bars['evaluation'].set_postfix({
                'FPS': f"{fps:.2f}",
                'Time': f"{inference_time*1000:.2f}ms"
            })