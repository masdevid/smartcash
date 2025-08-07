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
        self.phase = self.training_config.get('phase', 1)  # 1 or 2 for phase-based training
        
        # Phase-based defaults - AdamW with CosineAnnealingLR as standard
        if self.phase == 1:
            # Phase 1: Head LR = 0.001, Backbone LR = 1e-05
            self.learning_rate = self.training_config.get('learning_rate', 0.001)  # Head LR
            self.weight_decay = self.training_config.get('weight_decay', 1e-2)
        else:
            # Phase 2: Head LR = 0.0001, Backbone LR = 1e-05
            self.learning_rate = self.training_config.get('learning_rate', 0.0001)  # Head LR
            self.weight_decay = self.training_config.get('weight_decay', 1e-2)
            
        self.optimizer_type = self.training_config.get('optimizer', 'adamw').lower()
        self.scheduler_type = self.training_config.get('scheduler', 'cosine').lower()
        self.mixed_precision = self.training_config.get('mixed_precision', True)
        self.gradient_clip = self.training_config.get('gradient_clip', 10.0)
        self.total_epochs = self.training_config.get('total_epochs', 100)  # For scheduler
        
        # Batch size and LR scaling
        self.batch_size = self.training_config.get('batch_size', 16)  # Default batch size
        self.backbone_lr = 1e-05  # Fixed backbone LR as specified
        
        # Configurable AdamW optimizer parameters
        self.adamw_betas = self.training_config.get('adamw_betas', (0.9, 0.999))
        self.adamw_eps = self.training_config.get('adamw_eps', 1e-8)
        
        # Configurable CosineAnnealingLR scheduler parameters
        self.cosine_eta_min = self.training_config.get('cosine_eta_min', 1e-6)
        
        # Scaler untuk mixed precision
        self.scaler = GradScaler('cuda') if self.mixed_precision else None
    
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
        
        # Default to AdamW with the specified configuration (lr=5e-4 for phase 2)
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=self.weight_decay,
            betas=self.adamw_betas,
            eps=self.adamw_eps
        )
        
        return optimizer
    
    def _create_parameter_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        """Create parameter groups with specified learning rates"""
        backbone_params = []
        head_params = []
        other_params = []
        
        # Iterate through model parameters
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Backbone parameters
            if 'backbone' in name.lower():
                backbone_params.append(param)
            # Detection head parameters
            elif any(keyword in name.lower() for keyword in ['head', 'detect', 'cls', 'obj', 'box']):
                head_params.append(param)
            # Other parameters (neck, etc.)
            else:
                other_params.append(param)
        
        # Apply batch size scaling: if batch_size = 32, multiply LR by 3
        lr_scale = 3.0 if self.batch_size == 32 else 1.0
        scaled_head_lr = base_lr * lr_scale
        scaled_backbone_lr = self.backbone_lr * lr_scale  # Fixed 1e-05 backbone LR
        
        # Parameter groups with specified learning rates
        param_groups = []
        
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': scaled_backbone_lr,  # Fixed backbone LR: 1e-05 (scaled by 3x if batch_size=32)
                'name': 'backbone'
            })
        
        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': scaled_head_lr,  # Head LR: Phase 1=0.001, Phase 2=0.0001 (scaled by 3x if batch_size=32)
                'name': 'head'
            })
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': scaled_head_lr * 0.5,  # Medium learning rate for other components
                'name': 'other'
            })
        
        # Fallback: if no groups found, use all parameters with head LR
        if not param_groups:
            param_groups.append({
                'params': model.parameters(),
                'lr': scaled_head_lr,
                'name': 'all'
            })
        
        return param_groups
    
    def create_scheduler(self, optimizer: torch.optim.Optimizer, total_epochs: Optional[int] = None) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler"""
        total_epochs = total_epochs or self.total_epochs
        
        # Use scheduler for both phases by default (cosine annealing)
            
        if self.scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=total_epochs,
                eta_min=self.cosine_eta_min  # Configurable minimum learning rate
            )
        elif self.scheduler_type == 'step':
            return StepLR(optimizer, step_size=30, gamma=0.1)
        
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
            return GradScaler('cuda')
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