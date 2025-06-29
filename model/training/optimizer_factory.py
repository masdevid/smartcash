"""
File: smartcash/model/training/optimizer_factory.py
Deskripsi: Factory untuk membuat optimizers dan schedulers dengan mixed precision support
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, ReduceLROnPlateau, 
    ExponentialLR, MultiStepLR, CyclicLR
)
from torch.cuda.amp import GradScaler
from typing import Dict, List, Tuple, Optional, Any, Union
import math

class OptimizerFactory:
    """Factory untuk membuat optimizers dengan parameter groups dan scheduling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.training_config = config.get('training', {})
        
        # Optimizer parameters
        self.learning_rate = self.training_config.get('learning_rate', 0.001)
        self.weight_decay = self.training_config.get('weight_decay', 0.0005)
        self.optimizer_type = self.training_config.get('optimizer', 'adam').lower()
        self.scheduler_type = self.training_config.get('scheduler', 'cosine').lower()
        self.mixed_precision = self.training_config.get('mixed_precision', True)
        self.gradient_clip = self.training_config.get('gradient_clip', 10.0)
        
        # Scaler untuk mixed precision
        self.scaler = GradScaler() if self.mixed_precision else None
    
    def create_optimizer(self, model: nn.Module, custom_lr: Optional[float] = None) -> torch.optim.Optimizer:
        """
        Create optimizer dengan parameter groups yang berbeda untuk backbone dan head
        
        Args:
            model: Model yang akan dioptimize
            custom_lr: Custom learning rate (override config)
            
        Returns:
            Configured optimizer
        """
        lr = custom_lr or self.learning_rate
        
        # Create parameter groups dengan learning rate yang berbeda
        param_groups = self._create_parameter_groups(model, lr)
        
        # Create optimizer berdasarkan type
        if self.optimizer_type == 'adam':
            optimizer = optim.Adam(
                param_groups,
                lr=lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                param_groups,
                lr=lr,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif self.optimizer_type == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                lr=lr,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True
            )
        elif self.optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                param_groups,
                lr=lr,
                weight_decay=self.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"❌ Optimizer type tidak didukung: {self.optimizer_type}")
        
        return optimizer
    
    def _create_parameter_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        """Create parameter groups dengan learning rate yang berbeda"""
        backbone_params = []
        head_params = []
        other_params = []
        
        # Iterate through model parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Backbone parameters (biasanya freezed atau lower LR)
            if 'backbone' in name.lower():
                backbone_params.append(param)
            # Detection head parameters
            elif any(keyword in name.lower() for keyword in ['head', 'detect', 'cls', 'obj', 'box']):
                head_params.append(param)
            # Other parameters (neck, etc.)
            else:
                other_params.append(param)
        
        # Parameter groups dengan learning rate yang berbeda
        param_groups = []
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr * 0.1,  # Lower learning rate untuk backbone
                'name': 'backbone'
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': base_lr,  # Full learning rate untuk head
                'name': 'head'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr * 0.5,  # Medium learning rate untuk other components
                'name': 'other'
            })
        
        # Fallback: jika tidak ada groups, gunakan semua parameters
        if not param_groups:
            param_groups.append({
                'params': model.parameters(),
                'lr': base_lr,
                'name': 'all'
            })
        
        return param_groups
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, 
                        total_epochs: int) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Create learning rate scheduler
        
        Args:
            optimizer: Optimizer yang akan di-schedule
            total_epochs: Total training epochs
            
        Returns:
            Configured scheduler atau None
        """
        if self.scheduler_type == 'none':
            return None
        
        elif self.scheduler_type == 'cosine':
            return CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
                eta_min=self.learning_rate * 0.01  # Minimum LR = 1% dari initial
            )
        
        elif self.scheduler_type == 'step':
            step_size = max(total_epochs // 3, 10)  # Step setiap 1/3 total epochs
            return StepLR(
                optimizer,
                step_size=step_size,
                gamma=0.1  # Reduce LR by 10x
            )
        
        elif self.scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='max',  # Monitor mAP (higher is better)
                factor=0.5,  # Reduce LR by half
                patience=5,  # Wait 5 epochs
                min_lr=self.learning_rate * 0.001,
                verbose=True
            )
        
        elif self.scheduler_type == 'exponential':
            gamma = (0.01) ** (1 / total_epochs)  # Decay ke 1% dari initial LR
            return ExponentialLR(optimizer, gamma=gamma)
        
        elif self.scheduler_type == 'multistep':
            milestones = [total_epochs // 3, 2 * total_epochs // 3]
            return MultiStepLR(
                optimizer,
                milestones=milestones,
                gamma=0.1
            )
        
        elif self.scheduler_type == 'cyclic':
            return CyclicLR(
                optimizer,
                base_lr=self.learning_rate * 0.1,
                max_lr=self.learning_rate,
                step_size_up=total_epochs // 4,
                mode='triangular2'
            )
        
        else:
            raise ValueError(f"❌ Scheduler type tidak didukung: {self.scheduler_type}")
    
    def setup_mixed_precision(self) -> Optional[GradScaler]:
        """Setup mixed precision training"""
        if self.mixed_precision and torch.cuda.is_available():
            return GradScaler()
        return None
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor,
                      retain_graph: bool = False) -> Dict[str, Any]:
        """
        Perform optimizer step dengan gradient clipping dan mixed precision
        
        Args:
            optimizer: Optimizer untuk update parameters
            loss: Loss tensor
            retain_graph: Retain computation graph
            
        Returns:
            Dictionary dengan step information
        """
        step_info = {'grad_norm': 0.0, 'scaled': False}
        
        if self.scaler is not None:
            # Mixed precision training
            self.scaler.scale(loss).backward(retain_graph=retain_graph)
            
            # Gradient clipping
            if self.gradient_clip > 0:
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'], 
                    self.gradient_clip
                )
                step_info['grad_norm'] = grad_norm.item()
            
            # Optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
            step_info['scaled'] = True
        
        else:
            # Regular training
            loss.backward(retain_graph=retain_graph)
            
            # Gradient clipping
            if self.gradient_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group['params']], 
                    self.gradient_clip
                )
                step_info['grad_norm'] = grad_norm.item()
            
            # Optimizer step
            optimizer.step()
        
        # Clear gradients
        optimizer.zero_grad()
        
        return step_info
    
    def get_current_lr(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Get current learning rates untuk semua parameter groups"""
        lr_dict = {}
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            lr_dict[group_name] = group['lr']
        
        return lr_dict
    
    def warmup_lr(self, optimizer: torch.optim.Optimizer, epoch: int, warmup_epochs: int = 3) -> None:
        """Apply learning rate warmup untuk early epochs"""
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            
            for group in optimizer.param_groups:
                base_lr = group.get('initial_lr', group['lr'])
                group['lr'] = base_lr * warmup_factor
    
    def freeze_backbone(self, model: nn.Module, freeze: bool = True) -> int:
        """Freeze/unfreeze backbone parameters"""
        frozen_params = 0
        
        for name, param in model.named_parameters():
            if 'backbone' in name.lower():
                param.requires_grad = not freeze
                if freeze:
                    frozen_params += param.numel()
        
        return frozen_params
    
    def gradually_unfreeze(self, model: nn.Module, current_epoch: int, 
                          freeze_epochs: int = 5, total_layers: int = None) -> Dict[str, Any]:
        """Gradually unfreeze backbone layers selama training"""
        if current_epoch < freeze_epochs:
            return {'unfrozen_layers': 0, 'total_frozen': self.freeze_backbone(model, True)}
        
        # Get backbone layers
        backbone_layers = []
        for name, module in model.named_modules():
            if 'backbone' in name.lower() and len(list(module.children())) == 0:
                backbone_layers.append((name, module))
        
        if not backbone_layers:
            return {'unfrozen_layers': 0, 'total_frozen': 0}
        
        total_layers = total_layers or len(backbone_layers)
        
        # Calculate berapa layers yang harus unfrozen
        unfreeze_progress = (current_epoch - freeze_epochs) / max(1, total_layers)
        layers_to_unfreeze = min(int(unfreeze_progress * total_layers) + 1, total_layers)
        
        # Unfreeze layers dari akhir (closest to head)
        unfrozen_count = 0
        for i in range(total_layers - layers_to_unfreeze, total_layers):
            if i < len(backbone_layers):
                name, module = backbone_layers[i]
                for param in module.parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen_count += 1
        
        return {
            'unfrozen_layers': layers_to_unfreeze,
            'total_layers': total_layers,
            'unfrozen_params': unfrozen_count
        }
    
    def get_optimizer_info(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Get comprehensive optimizer information"""
        info = {
            'optimizer_type': type(optimizer).__name__,
            'num_param_groups': len(optimizer.param_groups),
            'total_params': sum(len(group['params']) for group in optimizer.param_groups),
            'current_lr': self.get_current_lr(optimizer),
            'mixed_precision': self.scaler is not None,
            'gradient_clip': self.gradient_clip
        }
        
        # Parameter group details
        info['param_groups'] = []
        for i, group in enumerate(optimizer.param_groups):
            group_info = {
                'name': group.get('name', f'group_{i}'),
                'lr': group['lr'],
                'num_params': len(group['params']),
                'weight_decay': group.get('weight_decay', 0)
            }
            info['param_groups'].append(group_info)
        
        return info

class WarmupScheduler:
    """Custom scheduler dengan warmup support"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int = 3,
                 base_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        self.current_epoch = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch: int = None, metrics: Optional[float] = None) -> None:
        """Step scheduler dengan warmup logic"""
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            # Warmup phase
            warmup_factor = (self.current_epoch + 1) / self.warmup_epochs
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * warmup_factor
        
        elif self.base_scheduler is not None:
            # Use base scheduler setelah warmup
            if isinstance(self.base_scheduler, ReduceLROnPlateau):
                if metrics is not None:
                    self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]

# Convenience functions
def create_optimizer_and_scheduler(model: nn.Module, config: Dict[str, Any], 
                                  total_epochs: int) -> Tuple[torch.optim.Optimizer, 
                                                             Optional[torch.optim.lr_scheduler._LRScheduler],
                                                             Optional[GradScaler]]:
    """One-liner untuk create optimizer, scheduler, dan scaler"""
    factory = OptimizerFactory(config)
    optimizer = factory.create_optimizer(model)
    scheduler = factory.create_scheduler(optimizer, total_epochs)
    scaler = factory.setup_mixed_precision()
    
    return optimizer, scheduler, scaler

def get_parameter_count(model: nn.Module) -> Dict[str, int]:
    """Get parameter count breakdown"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params
    }

def setup_optimizer_with_warmup(model: nn.Module, config: Dict[str, Any], 
                               total_epochs: int) -> Tuple[torch.optim.Optimizer, WarmupScheduler, Optional[GradScaler]]:
    """Setup optimizer dengan warmup scheduler"""
    factory = OptimizerFactory(config)
    optimizer = factory.create_optimizer(model)
    
    # Create base scheduler
    warmup_epochs = config.get('training', {}).get('warmup_epochs', 3)
    base_scheduler = factory.create_scheduler(optimizer, total_epochs)
    
    # Wrap dengan warmup
    warmup_scheduler = WarmupScheduler(optimizer, warmup_epochs, base_scheduler)
    
    scaler = factory.setup_mixed_precision()
    
    return optimizer, warmup_scheduler, scaler