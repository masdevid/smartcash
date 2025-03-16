"""
File: smartcash/model/services/training/warmup_scheduler_training_service.py
Deskripsi: Modul pemanasan penjadwalan untuk layanan pelatihan
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


class CosineDecayWithWarmup(_LRScheduler):
    """
    Cosine learning rate decay dengan warmup phase.
    
    * new: Custom scheduler with warmup support
    """
    
    def __init__(
        self, 
        optimizer: optim.Optimizer, 
        warmup_epochs: int = 3,
        max_epochs: int = 100,
        min_lr_factor: float = 0.01,
        last_epoch: int = -1
    ):
        """
        Inisialisasi scheduler.
        
        Args:
            optimizer: Optimizer PyTorch
            warmup_epochs: Jumlah epoch untuk fase warmup
            max_epochs: Total jumlah epoch
            min_lr_factor: Faktor untuk learning rate minimum (0.01 = 1% dari lr awal)
            last_epoch: Epoch terakhir
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr_factor = min_lr_factor
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Update learning rate berdasarkan schedule."""
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase: linear dari 0 ke lr awal
            warmup_factor = (self.last_epoch + 1) / (self.warmup_epochs + 1)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            decay_factor = self.min_lr_factor + (1 - self.min_lr_factor) * cosine_factor
            return [base_lr * decay_factor for base_lr in self.base_lrs]