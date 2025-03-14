"""
File: smartcash/model/services/training/early_stopping.py
Deskripsi: Handler untuk early stopping yang dapat mencegah overfitting dengan menghentikan training saat tidak ada peningkatan.
"""
import numpy as np
from typing import Dict, Union, Optional, Any


class EarlyStoppingHandler:
    """
    Handler untuk early stopping yang memonitor metrik selama training dan
    menghentikan proses jika tidak ada peningkatan.
    
    * old: utils.early_stopping.EarlyStopping
    * migrated: Improved with more flexibility and better heuristics
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = 'val_loss',
        mode: str = 'min',
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi early stopping handler.
        
        Args:
            patience: Jumlah epoch tanpa peningkatan sebelum berhenti
            min_delta: Perubahan minimum yang dianggap signifikan
            monitor: Metrik yang akan dimonitor
            mode: Mode evaluasi ('min' atau 'max')
            logger: Logger untuk mencatat aktivitas
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.logger = logger
        
        # Verifikasi mode valid
        if mode not in ['min', 'max']:
            raise ValueError(f"Mode {mode} tidak valid. Gunakan 'min' atau 'max'")
        
        # Setup best value dan counter
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False
        
        # Setup operator pembanding
        if mode == 'min':
            self.is_improvement = lambda current, best: current < best - min_delta
        else:
            self.is_improvement = lambda current, best: current > best + min_delta
            
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Periksa apakah training harus dihentikan.
        
        Args:
            metrics: Dictionary dari metrik yang telah dihitung
            
        Returns:
            Boolean yang menunjukkan apakah harus berhenti
        """
        # Pastikan metrik yang dimonitor ada
        if self.monitor not in metrics:
            if self.logger:
                self.logger.warning(f"⚠️ Metrik {self.monitor} tidak ditemukan dalam {metrics.keys()}")
            return False
        
        current_value = metrics[self.monitor]
        
        # Periksa apakah ada peningkatan
        if self.is_improvement(current_value, self.best_value):
            self._handle_improvement(current_value)
        else:
            self._handle_no_improvement(current_value)
            
        return self.early_stop
    
    def _handle_improvement(self, current_value: float) -> None:
        """
        Menangani kasus saat ada peningkatan dalam metrik yang dimonitor.
        
        Args:
            current_value: Nilai metrik saat ini
        """
        # Calculate improvement
        if self.mode == 'min':
            improvement = self.best_value - current_value
            sign = "-"
        else:
            improvement = current_value - self.best_value
            sign = "+"
        
        # Log improvement
        if self.logger:
            self.logger.success(
                f"✨ {self.monitor} membaik: {current_value:.4f} "
                f"({sign}{improvement:.4f})"
            )
        
        # Update best value dan reset counter
        self.best_value = current_value
        self.counter = 0
    
    def _handle_no_improvement(self, current_value: float) -> None:
        """
        Menangani kasus saat tidak ada peningkatan dalam metrik yang dimonitor.
        
        Args:
            current_value: Nilai metrik saat ini
        """
        # Increment counter
        self.counter += 1
        
        # Check if early stopping should be triggered
        if self.counter >= self.patience:
            self.early_stop = True
            if self.logger:
                self.logger.warning(
                    f"⚠️ Early stopping setelah {self.counter} epoch "
                    f"tanpa peningkatan di {self.monitor}"
                )
        else:
            remaining = self.patience - self.counter
            if self.logger:
                self.logger.info(
                    f"📉 Tidak ada peningkatan pada {self.monitor}. "
                    f"Akan early stop dalam {remaining} epoch"
                )
    
    def reset(self) -> None:
        """Reset state early stopping."""
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.counter = 0
        self.early_stop = False