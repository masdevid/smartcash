# File: utils/early_stopping.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk early stopping dengan support multiple metrics

import numpy as np
from typing import Dict, Optional
from smartcash.utils.logger import SmartCashLogger

class EarlyStopping:
    """
    Early stopping handler dengan dukungan multiple metrics monitoring.
    Mendukung mode minimization dan maximization.
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 5,
        mode: str = 'min',
        logger: Optional[SmartCashLogger] = None
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.logger = logger or SmartCashLogger(__name__)
        
        self.best_score = np.inf if mode == 'min' else -np.inf
        self.counter = 0
        self.early_stop = False
        
        # Validasi mode
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode {mode} tidak valid. Gunakan 'min' atau 'max'")
            
        # Setup operator perbandingan
        self.monitor_op = np.less if mode == 'min' else np.greater
        
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Check apakah training harus dihentikan
        Args:
            metrics: Dict metrik evaluasi
        Returns:
            bool: True jika training harus dihentikan
        """
        if self.monitor not in metrics:
            self.logger.warning(
                f"⚠️ Metrik {self.monitor} tidak ditemukan di {metrics.keys()}"
            )
            return False
            
        current = metrics[self.monitor]
        
        if self.monitor_op(current - self.min_delta, self.best_score):
            # Metrik membaik
            self._improvement_detected(current)
        else:
            # Tidak ada improvement
            self._no_improvement_detected()
            
        return self.early_stop
        
    def _improvement_detected(self, current: float) -> None:
        """Handle ketika ada improvement"""
        diff = abs(current - self.best_score)
        self.best_score = current
        self.counter = 0
        
        self.logger.success(
            f"✨ {self.monitor} membaik: {current:.4f} "
            f"(+{diff:.4f})"
        )
        
    def _no_improvement_detected(self) -> None:
        """Handle ketika tidak ada improvement"""
        self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            self.logger.warning(
                f"⚠️ Early stopping setelah {self.counter} epoch "
                f"tanpa improvement di {self.monitor}"
            )
        else:
            remaining = self.patience - self.counter
            self.logger.info(
                f"📉 Tidak ada improvement. "
                f"Akan early stop dalam {remaining} epoch"
            )
            
    def reset(self) -> None:
        """Reset state early stopping"""
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.counter = 0
        self.early_stop = False