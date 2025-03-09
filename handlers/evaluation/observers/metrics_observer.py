# File: smartcash/handlers/evaluation/observers/metrics_observer.py
# Author: Alfrida Sabar
# Deskripsi: Observer untuk monitoring metrik evaluasi

from typing import Dict, Any, Optional, List
import pandas as pd
import time
import os
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.handlers.evaluation.observers.base_observer import BaseObserver

class MetricsObserver(BaseObserver):
    """
    Observer untuk monitoring dan pencatatan metrik evaluasi.
    Berguna untuk tracking eksperimen dan visualisasi hasil.
    """
    
    def __init__(
        self, 
        name: str = "MetricsObserver",
        logger: Optional[SmartCashLogger] = None,
        output_dir: Optional[str] = None,
        save_to_csv: bool = True,
        track_metrics: Optional[List[str]] = None
    ):
        """
        Inisialisasi metrics observer.
        
        Args:
            name: Nama observer
            logger: Logger kustom (opsional)
            output_dir: Direktori output untuk file hasil (opsional)
            save_to_csv: Simpan metrik ke CSV
            track_metrics: List metrik yang akan ditrack (opsional, track semua jika None)
        """
        super().__init__(name)
        self.logger = logger or get_logger("metrics_observer")
        
        # Output path
        self.output_dir = Path(output_dir) if output_dir else Path("results/evaluation/metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Settings
        self.save_to_csv = save_to_csv
        self.track_metrics = track_metrics or ['mAP', 'f1', 'precision', 'recall', 'accuracy', 'inference_time']
        
        # Metrics storage
        self.metrics_history = []
        self.current_run_info = {}
        
        # Timestamps
        self.start_time = None
        
        self.logger.debug(f"🔧 {name} diinisialisasi (output_dir={self.output_dir})")
    
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
        elif event == 'evaluation_complete':
            self._handle_evaluation_complete(data)
        elif event == 'batch_complete':
            self._handle_batch_complete(data)
        elif event == 'research_complete':
            self._handle_research_complete(data)
    
    def _handle_pipeline_start(self, data: Dict[str, Any]):
        """
        Handle pipeline_start event.
        
        Args:
            data: Event data
        """
        # Reset untuk run baru
        self.start_time = data.get('start_time', time.time())
        
        # Simpan info run
        self.current_run_info = {
            'pipeline_name': data.get('pipeline_name', 'Unknown'),
            'model_path': data.get('model_path', 'Unknown'),
            'dataset_path': data.get('dataset_path', 'Unknown'),
            'start_time': self.start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _handle_pipeline_complete(self, data: Dict[str, Any]):
        """
        Handle pipeline_complete event.
        
        Args:
            data: Event data
        """
        # Simpan waktu eksekusi
        execution_time = data.get('execution_time', time.time() - self.start_time)
        self.current_run_info['execution_time'] = execution_time
        
        # Jika ada hasil langsung, simpan
        if 'results' in data:
            self._save_metrics(data['results'])
        
        # Reset current run info
        self.current_run_info = {}
    
    def _handle_pipeline_error(self, data: Dict[str, Any]):
        """
        Handle pipeline_error event.
        
        Args:
            data: Event data
        """
        # Tandai run sebagai error
        self.current_run_info['error'] = data.get('error', 'Unknown error')
        self.current_run_info['status'] = 'error'
        
        # Simpan ke history
        self.metrics_history.append(self.current_run_info.copy())
        
        # Reset current run info
        self.current_run_info = {}
    
    def _handle_evaluation_complete(self, data: Dict[str, Any]):
        """
        Handle evaluation_complete event.
        
        Args:
            data: Event data
        """
        self._save_metrics(data)
    
    def _handle_batch_complete(self, data: Dict[str, Any]):
        """
        Handle batch_complete event dari batch process.
        
        Args:
            data: Event data
        """
        # Jika ada summary, simpan
        if 'summary' in data:
            summary = data['summary']
            
            # Jika ada metrics_table, tambahkan ke history
            if 'metrics_table' in summary and isinstance(summary['metrics_table'], pd.DataFrame):
                df = summary['metrics_table']
                
                # Simpan ke CSV
                if self.save_to_csv:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    csv_path = self.output_dir / f"batch_metrics_{timestamp}.csv"
                    df.to_csv(csv_path, index=False)
                    self.logger.info(f"📊 Metrik batch disimpan ke {csv_path}")
    
    def _handle_research_complete(self, data: Dict[str, Any]):
        """
        Handle research_complete event.
        
        Args:
            data: Event data
        """
        # Jika ada summary, simpan
        if 'summary' in data:
            summary = data['summary']
            
            # Jika ada metrics_table, tambahkan ke history
            if 'metrics_table' in summary and isinstance(summary['metrics_table'], pd.DataFrame):
                df = summary['metrics_table']
                
                # Simpan ke CSV
                if self.save_to_csv:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    csv_path = self.output_dir / f"research_metrics_{timestamp}.csv"
                    df.to_csv(csv_path, index=False)
                    self.logger.info(f"📊 Metrik penelitian disimpan ke {csv_path}")
    
    def _save_metrics(self, data: Dict[str, Any]):
        """
        Simpan metrik dari hasil evaluasi.
        
        Args:
            data: Data hasil evaluasi
        """
        # Extract metrik
        metrics_entry = self.current_run_info.copy()
        metrics_entry['status'] = 'success'
        
        # Tambahkan timestamp jika tidak ada
        if 'timestamp' not in metrics_entry:
            metrics_entry['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract metrik dari results
        if 'metrics' in data:
            metrics = data['metrics']
            for metric_name in self.track_metrics:
                if metric_name in metrics:
                    # Konversi waktu inferensi ke ms
                    if metric_name == 'inference_time':
                        metrics_entry[metric_name] = metrics[metric_name] * 1000
                    else:
                        metrics_entry[metric_name] = metrics[metric_name]
        
        # Tambahkan ke history
        self.metrics_history.append(metrics_entry)
        
        # Simpan ke CSV
        if self.save_to_csv and self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            
            # Hapus kolom yang tidak perlu
            if 'start_time' in df.columns:
                df = df.drop(columns=['start_time'])
            
            # Simpan ke CSV
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            csv_path = self.output_dir / f"evaluation_metrics_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            self.logger.info(f"📊 Metrik evaluasi disimpan ke {csv_path}")
    
    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """
        Dapatkan history metrik evaluasi.
        
        Returns:
            List metrics entries
        """
        return self.metrics_history.copy()
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """
        Dapatkan metrics history sebagai DataFrame.
        
        Returns:
            DataFrame metrics
        """
        return pd.DataFrame(self.metrics_history)