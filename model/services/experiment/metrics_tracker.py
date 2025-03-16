"""
File: smartcash/model/services/experiment/metrics_tracker.py
Deskripsi: Komponen untuk melacak dan visualisasi metrik eksperimen
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import time

from smartcash.common.logger import get_logger


class ExperimentMetricsTracker:
    """
    Komponen untuk melacak dan visualisasi metrik eksperimen.
    
    Bertanggung jawab untuk:
    - Pelacakan metrik selama training
    - Perhitungan statistik
    - Simpan dan load metrik
    - Visualisasi hasil
    """
    
    def __init__(
        self,
        experiment_dir: str,
        metrics_to_track: List[str] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi metrics tracker.
        
        Args:
            experiment_dir: Direktori untuk menyimpan metrik
            metrics_to_track: List metrik yang akan dilacak (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.experiment_dir = Path(experiment_dir)
        self.metrics_dir = self.experiment_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or get_logger("experiment_metrics_tracker")
        
        # Default metrik yang dilacak
        self.metrics_to_track = metrics_to_track or [
            "train_loss", "val_loss", "mAP", "precision", 
            "recall", "f1", "learning_rate"
        ]
        
        # Struktur untuk menyimpan metrik
        self.metrics = {metric: [] for metric in self.metrics_to_track}
        self.metrics["epoch"] = []
        
        # Timestamp untuk tracking waktu
        self.start_time = None
        self.epoch_times = []
        
        self.logger.debug(f"📊 ExperimentMetricsTracker diinisialisasi")
    
    def start_tracking(self) -> None:
        """Mulai tracking waktu."""
        self.start_time = time.time()
        self.logger.debug(f"⏱️ Mulai tracking metrik pada {self.start_time}")
    
    def track_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        save: bool = True
    ) -> Dict[str, float]:
        """
        Catat metrik untuk satu epoch.
        
        Args:
            epoch: Nomor epoch
            metrics: Dictionary metrik epoch
            save: Flag untuk menyimpan metrik ke disk
            
        Returns:
            Dictionary metrik epoch dengan statistik tambahan
        """
        # Catat epoch
        self.metrics["epoch"].append(epoch)
        
        # Catat semua metrik yang diminta
        for metric in self.metrics_to_track:
            if metric in metrics:
                self.metrics[metric].append(metrics[metric])
            else:
                # Placeholder jika metrik tidak tersedia
                self.metrics[metric].append(None)
        
        # Catat waktu epoch
        epoch_time = time.time() - (self.start_time or time.time())
        self.epoch_times.append(epoch_time)
        
        # Tambahkan statistik untuk logging
        enhanced_metrics = {
            **metrics,
            "epoch": epoch,
            "elapsed_time": epoch_time
        }
        
        # Tambahkan best metrics jika ada history
        if len(self.metrics["epoch"]) > 1:
            for metric in ["val_loss", "mAP", "f1"]:
                if metric in self.metrics and self.metrics[metric] and not all(x is None for x in self.metrics[metric]):
                    # Filter None values
                    metric_history = [x for x in self.metrics[metric] if x is not None]
                    
                    if metric_history:
                        if metric == "val_loss":
                            best_value = min(metric_history)
                            is_best = metrics.get(metric) == best_value
                        else:
                            best_value = max(metric_history)
                            is_best