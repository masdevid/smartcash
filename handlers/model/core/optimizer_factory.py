# File: smartcash/handlers/model/core/optimizer_factory.py
# Author: Alfrida Sabar
# Deskripsi: Factory untuk membuat optimizer dan scheduler dengan berbagai konfigurasi

import torch
from typing import Dict, Optional, Any, Union, Tuple

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError
from smartcash.handlers.model.core.model_component import ModelComponent

class OptimizerFactory(ModelComponent):
    """
    Factory untuk membuat optimizer dan scheduler dengan berbagai konfigurasi.
    """
    
    def __init__(
        self, 
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi factory.
        
        Args:
            config: Konfigurasi training
            logger: Custom logger (opsional)
        """
        super().__init__(config, logger, "optimizer_factory")
        
    def _initialize(self) -> None:
        """Inisialisasi internal komponen."""
        # Ambil konfigurasi training
        self.training_config = self.config.get('training', {})
    
    def process(
        self, 
        model: torch.nn.Module,
        optimizer_type: Optional[str] = None,
        **kwargs
    ) -> torch.optim.Optimizer:
        """
        Membuat optimizer untuk model.
        Alias untuk create_optimizer().
        
        Args:
            model: Model yang akan dioptimasi
            optimizer_type: Tipe optimizer
            **kwargs: Parameter tambahan
            
        Returns:
            Optimizer yang sesuai dengan konfigurasi
        """
        return self.create_optimizer(
            model=model,
            optimizer_type=optimizer_type,
            **kwargs
        )
    
    def create_optimizer(
        self, 
        model: torch.nn.Module, 
        optimizer_type: Optional[str] = None,
        lr: Optional[float] = None,
        weight_decay: Optional[float] = None,
        momentum: Optional[float] = None,
        clip_grad_norm: Optional[float] = None
    ) -> torch.optim.Optimizer:
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
        optimizer_type = optimizer_type or self.training_config.get('optimizer', 'adam').lower()
        learning_rate = lr or self.training_config.get('lr0', 0.001)
        weight_decay = weight_decay or self.training_config.get('weight_decay', 0.0005)
        momentum = momentum or self.training_config.get('momentum', 0.9)
        
        self.logger.info(f"🔧 Membuat optimizer {optimizer_type} dengan learning rate {learning_rate}")
        
        # Buat optimizer sesuai tipe
        try:
            if optimizer_type == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=learning_rate,
                    momentum=momentum,
                    weight_decay=weight_decay
                )
            else:
                self.logger.warning(f"⚠️ Tipe optimizer '{optimizer_type}' tidak dikenal, menggunakan Adam")
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=weight_decay
                )
                
            # Tambahkan gradient clipping jika diminta
            if clip_grad_norm is not None:
                for group in optimizer.param_groups:
                    group.setdefault('max_grad_norm', clip_grad_norm)
                
                # Wrap step method dengan gradient clipping
                original_step = optimizer.step
                
                def step_with_clip(*args, **kwargs):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                    return original_step(*args, **kwargs)
                    
                optimizer.step = step_with_clip
                self.logger.info(f"🔒 Gradient clipping diaktifkan dengan max_norm={clip_grad_norm}")
                
            return optimizer
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat optimizer: {str(e)}")
            raise ModelError(f"Gagal membuat optimizer: {str(e)}")
    
    def create_scheduler(
        self, 
        optimizer: torch.optim.Optimizer,
        scheduler_type: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Membuat learning rate scheduler berdasarkan konfigurasi.
        
        Args:
            optimizer: Optimizer yang akan dijadwalkan
            scheduler_type: Tipe scheduler ('plateau', 'step', 'cosine', 'onecycle')
            **kwargs: Parameter tambahan untuk scheduler
            
        Returns:
            Learning rate scheduler
        """
        scheduler_type = scheduler_type or self.training_config.get('scheduler', 'cosine').lower()
        
        self.logger.info(f"🔧 Membuat scheduler {scheduler_type}")
        
        try:
            if scheduler_type == 'plateau':
                patience = kwargs.get('patience', self.training_config.get('patience', 3))
                factor = kwargs.get('factor', 0.5)
                
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=factor,
                    patience=patience,
                    verbose=True
                )
            elif scheduler_type == 'step':
                step_size = kwargs.get('step_size', self.training_config.get('lr_step_size', 10))
                gamma = kwargs.get('gamma', self.training_config.get('lr_gamma', 0.1))
                
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
            elif scheduler_type == 'cosine':
                epochs = kwargs.get('epochs', self.training_config.get('epochs', 30))
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs
                )
            elif scheduler_type == 'onecycle':
                epochs = kwargs.get('epochs', self.training_config.get('epochs', 30))
                steps_per_epoch = kwargs.get('steps_per_epoch', None)
                max_lr = kwargs.get('max_lr', self.training_config.get('lr0', 0.01) * 10)
                
                if steps_per_epoch is None:
                    self.logger.warning("⚠️ OneCycleLR membutuhkan steps_per_epoch, gunakan ReduceLROnPlateau")
                    return self.create_scheduler(optimizer, 'plateau')
                
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=max_lr,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch
                )
            else:
                self.logger.warning(f"⚠️ Tipe scheduler '{scheduler_type}' tidak dikenal, menggunakan CosineAnnealingLR")
                epochs = kwargs.get('epochs', self.training_config.get('epochs', 30))
                
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=epochs
                )
                
            return scheduler
            
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat scheduler: {str(e)}")
            raise ModelError(f"Gagal membuat scheduler: {str(e)}")
            
    def create_optimizer_and_scheduler(
        self, 
        model: torch.nn.Module,
        **kwargs
    ) -> Tuple[torch.optim.Optimizer, Any]:
        """
        Membuat optimizer dan scheduler secara bersamaan.
        
        Args:
            model: Model yang akan dioptimasi
            **kwargs: Parameter untuk optimizer dan scheduler
            
        Returns:
            Tuple dari (optimizer, scheduler)
        """
        # Buat optimizer
        optimizer = self.create_optimizer(model, **kwargs)
        
        # Tambahkan steps_per_epoch untuk OneCycleLR jika ada
        steps_per_epoch = kwargs.get('steps_per_epoch', None)
        
        # Buat scheduler
        scheduler = self.create_scheduler(optimizer, steps_per_epoch=steps_per_epoch, **kwargs)
        
        return optimizer, scheduler