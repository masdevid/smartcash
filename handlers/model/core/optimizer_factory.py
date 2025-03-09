# File: smartcash/handlers/model/core/optimizer_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory untuk membuat optimizer dan scheduler dengan implementasi yang dioptimasi

import torch
from typing import Dict, Optional, Any, Union, Tuple

from smartcash.utils.logger import get_logger
from smartcash.exceptions.base import ModelError
from smartcash.handlers.model.core.model_component import ModelComponent

class OptimizerFactory(ModelComponent):
    """Factory untuk membuat optimizer dan scheduler."""
    
    def _initialize(self) -> None:
        """Inisialisasi parameter default dari config."""
        self.defaults = self.config.get('training', {})
    
    def process(self, model, optimizer_type=None, **kwargs):
        """Alias untuk create_optimizer."""
        return self.create_optimizer(model, optimizer_type, **kwargs)
    
    def create_optimizer(self, model, optimizer_type=None, lr=None, weight_decay=None, 
                        momentum=None, clip_grad_norm=None):
        """
        Membuat optimizer untuk model.
        
        Args:
            model: Model yang akan dioptimasi
            optimizer_type: Tipe optimizer ('adam', 'adamw', 'sgd')
            lr: Learning rate
            weight_decay: Weight decay
            momentum: Momentum (untuk SGD)
            clip_grad_norm: Nilai maksimum untuk gradient clipping
            
        Returns:
            Optimizer yang sesuai dengan konfigurasi
        """
        # Ambil parameter dari config jika tidak ada yang diberikan
        optimizer_type = optimizer_type or self.defaults.get('optimizer', 'adam').lower()
        lr = lr or self.defaults.get('lr0', 0.001)
        weight_decay = weight_decay or self.defaults.get('weight_decay', 0.0005)
        momentum = momentum or self.defaults.get('momentum', 0.9)
        
        self.logger.info(f"🔧 Optimizer: {optimizer_type}, LR: {lr:.6f}")
        
        try:
            # Pilih optimizer berdasarkan tipe
            optimizers = {
                'adam': lambda: torch.optim.Adam(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                ),
                'adamw': lambda: torch.optim.AdamW(
                    model.parameters(), lr=lr, weight_decay=weight_decay
                ),
                'sgd': lambda: torch.optim.SGD(
                    model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
                )
            }
            
            # Default ke Adam jika tipe tidak dikenal
            optimizer_fn = optimizers.get(optimizer_type, optimizers['adam'])
            optimizer = optimizer_fn()
            
            # Tambahkan gradient clipping jika diminta
            if clip_grad_norm is not None:
                # Wrap step method dengan gradient clipping
                original_step = optimizer.step
                def step_with_clip(*args, **kwargs):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    return original_step(*args, **kwargs)
                optimizer.step = step_with_clip
                self.logger.info(f"🔒 Gradient clipping: max_norm={clip_grad_norm}")
                
            return optimizer
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat optimizer: {str(e)}")
            raise ModelError(f"Gagal membuat optimizer: {str(e)}")
    
    def create_scheduler(self, optimizer, scheduler_type=None, **kwargs):
        """
        Membuat learning rate scheduler.
        
        Args:
            optimizer: Optimizer yang akan dijadwalkan
            scheduler_type: Tipe scheduler ('plateau', 'step', 'cosine', 'onecycle')
            **kwargs: Parameter tambahan untuk scheduler
            
        Returns:
            Learning rate scheduler
        """
        scheduler_type = scheduler_type or self.defaults.get('scheduler', 'cosine').lower()
        self.logger.info(f"🔧 Scheduler: {scheduler_type}")
        
        try:
            # Mapping tipe scheduler ke implementasi
            epochs = kwargs.get('epochs', self.defaults.get('epochs', 30))
            schedulers = {
                'plateau': lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', 
                    factor=kwargs.get('factor', 0.5),
                    patience=kwargs.get('patience', self.defaults.get('patience', 3)),
                    verbose=True
                ),
                'step': lambda: torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=kwargs.get('step_size', self.defaults.get('lr_step_size', 10)),
                    gamma=kwargs.get('gamma', self.defaults.get('lr_gamma', 0.1))
                ),
                'cosine': lambda: torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs
                ),
                'onecycle': lambda: torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=kwargs.get('max_lr', lr * 10),
                    epochs=epochs,
                    steps_per_epoch=kwargs.get('steps_per_epoch', 100)
                ) if kwargs.get('steps_per_epoch') is not None else None
            }
            
            # Get scheduler function
            scheduler_fn = schedulers.get(scheduler_type)
            if not scheduler_fn:
                self.logger.warning(f"⚠️ Scheduler {scheduler_type} tidak tersedia, gunakan cosine")
                scheduler_fn = schedulers['cosine']
                
            # Create scheduler
            scheduler = scheduler_fn()
            if scheduler is None and scheduler_type == 'onecycle':
                self.logger.warning("⚠️ OneCycleLR gagal (steps_per_epoch?), gunakan ReduceLROnPlateau")
                scheduler = schedulers['plateau']()
                
            return scheduler
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat scheduler: {str(e)}")
            raise ModelError(f"Gagal membuat scheduler: {str(e)}")
            
    def create_optimizer_and_scheduler(self, model, **kwargs):
        """
        Membuat optimizer dan scheduler secara bersamaan.
        
        Args:
            model: Model yang akan dioptimasi
            **kwargs: Parameter untuk optimizer dan scheduler
            
        Returns:
            Tuple (optimizer, scheduler)
        """
        # Buat optimizer
        optimizer = self.create_optimizer(model, **kwargs)
        
        # Buat scheduler
        scheduler = self.create_scheduler(optimizer, **kwargs)
        
        return optimizer, scheduler