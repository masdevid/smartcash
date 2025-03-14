"""
File: smartcash/model/services/training/scheduler.py
Deskripsi: Factory untuk learning rate scheduler yang dioptimalkan untuk proses training deteksi mata uang.
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, Type, Any, Optional

from smartcash.model.services.training.warmup_scheduler import CosineDecayWithWarmup


class SchedulerFactory:
    """
    Factory class untuk membuat learning rate scheduler.
    
    * old: handlers.model.core.training_utils.get_scheduler()
    * migrated: Simplified factory-based scheduler creation
    """
    
    # Mapping dari nama scheduler ke class
    SCHEDULER_MAP = {
        'cosine': optim.lr_scheduler.CosineAnnealingLR,
        'cosine_with_restarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'step': optim.lr_scheduler.StepLR,
        'multistep': optim.lr_scheduler.MultiStepLR,
        'exponential': optim.lr_scheduler.ExponentialLR,
        'plateau': optim.lr_scheduler.ReduceLROnPlateau,
        'warmup_cosine': CosineDecayWithWarmup
    }
    
    @classmethod
    def create(cls, 
               scheduler_type: str, 
               optimizer: optim.Optimizer,
               **kwargs) -> _LRScheduler:
        """
        Buat scheduler berdasarkan tipe dan parameter.
        
        Args:
            scheduler_type: Tipe scheduler ('cosine', 'step', etc)
            optimizer: Optimizer PyTorch
            **kwargs: Parameter tambahan untuk scheduler
            
        Returns:
            Instance scheduler
            
        Raises:
            ValueError: Jika tipe scheduler tidak didukung
        """
        # Validasi tipe scheduler
        scheduler_type = scheduler_type.lower()
        if scheduler_type not in cls.SCHEDULER_MAP:
            raise ValueError(f"Tipe scheduler '{scheduler_type}' tidak didukung. "
                           f"Pilihan: {list(cls.SCHEDULER_MAP.keys())}")
        
        # Dapatkan class scheduler
        scheduler_class = cls.SCHEDULER_MAP[scheduler_type]
        
        # Parameter khusus per scheduler
        if scheduler_type == 'cosine':
            T_max = kwargs.pop('T_max', 100)
            eta_min = kwargs.pop('eta_min', 0)
            
            return scheduler_class(
                optimizer, 
                T_max=T_max,
                eta_min=eta_min,
                **kwargs
            )
            
        elif scheduler_type == 'cosine_with_restarts':
            T_0 = kwargs.pop('T_0', 10)
            T_mult = kwargs.pop('T_mult', 2)
            eta_min = kwargs.pop('eta_min', 0)
            
            return scheduler_class(
                optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min,
                **kwargs
            )
            
        elif scheduler_type == 'step':
            step_size = kwargs.pop('step_size', 30)
            gamma = kwargs.pop('gamma', 0.1)
            
            return scheduler_class(
                optimizer,
                step_size=step_size,
                gamma=gamma,
                **kwargs
            )
            
        elif scheduler_type == 'multistep':
            milestones = kwargs.pop('milestones', [30, 60, 90])
            gamma = kwargs.pop('gamma', 0.1)
            
            return scheduler_class(
                optimizer,
                milestones=milestones,
                gamma=gamma,
                **kwargs
            )
            
        elif scheduler_type == 'exponential':
            gamma = kwargs.pop('gamma', 0.95)
            
            return scheduler_class(
                optimizer,
                gamma=gamma,
                **kwargs
            )
            
        elif scheduler_type == 'plateau':
            mode = kwargs.pop('mode', 'min')
            factor = kwargs.pop('factor', 0.1)
            patience = kwargs.pop('patience', 10)
            threshold = kwargs.pop('threshold', 0.0001)
            min_lr = kwargs.pop('min_lr', 0)
            
            return scheduler_class(
                optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                min_lr=min_lr,
                **kwargs
            )
            
        elif scheduler_type == 'warmup_cosine':
            warmup_epochs = kwargs.pop('warmup_epochs', 3)
            max_epochs = kwargs.pop('max_epochs', 100)
            min_lr_factor = kwargs.pop('min_lr_factor', 0.01)
            
            return scheduler_class(
                optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
                min_lr_factor=min_lr_factor,
                **kwargs
            )
            
        # Fallback jika tidak ada case khusus
        return scheduler_class(optimizer, **kwargs)
    
    @classmethod
    def create_one_cycle_scheduler(
        cls, 
        optimizer: optim.Optimizer, 
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        **kwargs
    ) -> optim.lr_scheduler.OneCycleLR:
        """
        Buat One Cycle LR scheduler yang efektif untuk konvergensi cepat.
        
        Args:
            optimizer: Optimizer PyTorch
            max_lr: Learning rate maksimum
            total_steps: Total jumlah step training
            pct_start: Persentase total steps untuk fase warmup
            **kwargs: Parameter tambahan
            
        Returns:
            OneCycleLR scheduler
        """
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            **kwargs
        )