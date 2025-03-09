# File: smartcash/handlers/evaluation/integration/metrics_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk MetricsCalculator dari modul utils

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from smartcash.utils.logger import SmartCashLogger, get_logger
from smartcash.utils.metrics import MetricsCalculator

class MetricsAdapter:
    """
    Adapter untuk MetricsCalculator dari utils.metrics.
    Menyediakan antarmuka yang konsisten untuk menghitung dan mengelola metrik evaluasi.
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi MetricsAdapter.
        
        Args:
            config: Konfigurasi untuk evaluasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("metrics_adapter")
        
        # Inisialisasi MetricsCalculator dari utils
        self.metrics_calculator = MetricsCalculator()
        
        # Konfigurasi metrik
        self.metrics_config = self.config.get('evaluation', {}).get('metrics', {})
        self.iou_threshold = self.metrics_config.get('iou_threshold', 0.5)
        self.conf_threshold = self.metrics_config.get('conf_threshold', 0.25)
        
        self.logger.debug(f"🔧 MetricsAdapter diinisialisasi (iou_threshold={self.iou_threshold:.2f}, conf_threshold={self.conf_threshold:.2f})")
    
    def reset(self):
        """Reset metrics calculator untuk perhitungan baru."""
        self.metrics_calculator.reset()
        self.logger.debug("🔄 Metrics calculator direset")
    
    def update(
        self, 
        predictions: Union[torch.Tensor, np.ndarray], 
        targets: Union[torch.Tensor, np.ndarray]
    ):
        """
        Update metrik dengan batch prediksi dan target baru.
        
        Args:
            predictions: Tensor prediksi dari model
            targets: Tensor target ground truth
        """
        # Pastikan tipe data konsisten
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
        
        # Update metrik
        self.metrics_calculator.update(
            predictions, 
            targets, 
            iou_threshold=self.iou_threshold,
            conf_threshold=self.conf_threshold
        )
    
    def compute(self) -> Dict[str, Any]:
        """
        Hitung metrik berdasarkan semua batch yang telah diupdate.
        
        Returns:
            Dictionary berisi metrik evaluasi
        """
        # Hitung semua metrik
        metrics = self.metrics_calculator.compute()
        
        # Format hasil untuk lebih mudah dibaca
        formatted_metrics = {
            'accuracy': metrics.get('accuracy', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'f1': metrics.get('f1', 0.0),
            'mAP': metrics.get('mAP', 0.0),
            'inference_time': metrics.get('inference_time', 0.0),
            # Breakdown per kelas
            'class_metrics': metrics.get('class_metrics', {}),
            # Tambahkan metrik tambahan jika ada
            'additional_metrics': {
                k: v for k, v in metrics.items() 
                if k not in ['accuracy', 'precision', 'recall', 'f1', 'mAP', 'inference_time', 'class_metrics']
            }
        }
        
        return formatted_metrics
    
    def format_metrics_log(self, metrics: Dict[str, Any]) -> str:
        """
        Format metrik untuk output log.
        
        Args:
            metrics: Dictionary berisi metrik evaluasi
            
        Returns:
            String terformat untuk output log
        """
        return (
            f"📊 Hasil Evaluasi:\n"
            f"  Akurasi: \033[1;32m{metrics['accuracy']:.4f}\033[0m\n"
            f"  Presisi: \033[1;32m{metrics['precision']:.4f}\033[0m\n"
            f"  Recall: \033[1;32m{metrics['recall']:.4f}\033[0m\n"
            f"  F1-Score: \033[1;32m{metrics['f1']:.4f}\033[0m\n"
            f"  mAP: \033[1;32m{metrics['mAP']:.4f}\033[0m\n"
            f"  Waktu Inferensi: \033[1;36m{metrics['inference_time']*1000:.2f} ms\033[0m"
        )
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Log metrik ke logger.
        
        Args:
            metrics: Dictionary berisi metrik evaluasi
        """
        formatted_log = self.format_metrics_log(metrics)
        self.logger.info(formatted_log)
    
    def get_confusion_matrix(self) -> np.ndarray:
        """
        Dapatkan confusion matrix dari metrics calculator.
        
        Returns:
            Confusion matrix sebagai numpy array
        """
        return self.metrics_calculator.get_confusion_matrix()