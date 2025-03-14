"""
File: smartcash/model/services/training/optimizer_training_service.py
Deskripsi: Modul optimasi untuk layanan pelatihan
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Iterator, List, Union, Type, Optional


class OptimizerFactory:
    """
    Factory class untuk membuat dan mengkonfigurasi optimizer.
    
    * old: handlers.model.core.training_utils.get_optimizer()
    * migrated: Simplified factory-based optimizer creation
    """
    
    # Mapping dari nama optimizer ke class
    OPTIMIZER_MAP: Dict[str, Type[optim.Optimizer]] = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop
    }
    
    @classmethod
    def create(cls, 
               optimizer_type: str, 
               model_params: Iterator[torch.nn.Parameter],
               **kwargs) -> optim.Optimizer:
        """
        Buat optimizer berdasarkan tipe dan parameter.
        
        Args:
            optimizer_type: Tipe optimizer ('adam', 'sgd', etc)
            model_params: Parameter model untuk dioptimasi
            **kwargs: Parameter tambahan untuk optimizer
            
        Returns:
            Instance optimizer
            
        Raises:
            ValueError: Jika tipe optimizer tidak didukung
        """
        # Validasi tipe optimizer
        optimizer_type = optimizer_type.lower()
        if optimizer_type not in cls.OPTIMIZER_MAP:
            raise ValueError(f"Tipe optimizer '{optimizer_type}' tidak didukung. "
                           f"Pilihan: {list(cls.OPTIMIZER_MAP.keys())}")
        
        # Dapatkan class optimizer
        optimizer_class = cls.OPTIMIZER_MAP[optimizer_type]
        
        # Ekstrak parameter umum
        lr = kwargs.pop('lr', 0.001)
        weight_decay = kwargs.pop('weight_decay', 0)
        
        # Parameter khusus per optimizer
        if optimizer_type == 'adam' or optimizer_type == 'adamw':
            betas = kwargs.pop('betas', (0.9, 0.999))
            eps = kwargs.pop('eps', 1e-8)
            amsgrad = kwargs.pop('amsgrad', False)
            
            return optimizer_class(
                model_params, 
                lr=lr, 
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                **kwargs
            )
            
        elif optimizer_type == 'sgd':
            momentum = kwargs.pop('momentum', 0.9)
            dampening = kwargs.pop('dampening', 0)
            nesterov = kwargs.pop('nesterov', False)
            
            return optimizer_class(
                model_params,
                lr=lr,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=nesterov,
                **kwargs
            )
            
        elif optimizer_type == 'rmsprop':
            alpha = kwargs.pop('alpha', 0.99)
            eps = kwargs.pop('eps', 1e-8)
            momentum = kwargs.pop('momentum', 0)
            centered = kwargs.pop('centered', False)
            
            return optimizer_class(
                model_params,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
                **kwargs
            )
            
        # Fallback jika tidak ada case khusus
        return optimizer_class(model_params, lr=lr, **kwargs)
    
    @classmethod
    def create_optimizer_with_layer_lr(cls,
                                     model: torch.nn.Module,
                                     base_lr: float = 0.001,
                                     backbone_lr_factor: float = 0.1,
                                     optimizer_type: str = 'adam',
                                     **kwargs) -> optim.Optimizer:
        """
        Buat optimizer dengan learning rate berbeda untuk setiap komponen model.
        
        Args:
            model: Model PyTorch
            base_lr: Learning rate dasar
            backbone_lr_factor: Faktor untuk learning rate backbone (0.1 berarti 10x lebih kecil)
            optimizer_type: Tipe optimizer
            **kwargs: Parameter tambahan untuk optimizer
            
        Returns:
            Optimizer dengan parameter group berbeda
        """
        # Split parameter untuk grup berbeda
        backbone_params = []
        head_params = []
        
        # Iterasi parameter model dan kelompokkan berdasarkan nama
        for name, param in model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Buat parameter groups
        param_groups = [
            {'params': backbone_params, 'lr': base_lr * backbone_lr_factor},
            {'params': head_params, 'lr': base_lr}
        ]
        
        # Buat optimizer
        optimizer_class = cls.OPTIMIZER_MAP.get(optimizer_type.lower(), optim.Adam)
        return optimizer_class(param_groups, **kwargs)