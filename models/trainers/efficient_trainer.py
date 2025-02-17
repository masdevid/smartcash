# File: models/trainers/efficient_trainer.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi trainer untuk YOLOv5 dengan EfficientNet-B4 backbone

import torch
from typing import Dict, List, Optional, Tuple
import math


from .yolo_trainer import YOLOv5Trainer
from utils.logger import SmartCashLogger

class EfficientYOLOTrainer(YOLOv5Trainer):
    """Trainer untuk YOLOv5 dengan EfficientNet-B4 backbone"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__(model, config, logger)
        
        # Setup learning rate untuk backbone dan adapter
        self.backbone_lr = self.config['training'].get(
            'backbone_lr',
            self.config['training']['lr0'] * 0.1
        )
        
    def configure_optimizers(
        self
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Setup optimizer dengan different learning rates untuk
        backbone, adapter, dan detection heads
        """
        # Split parameter groups
        backbone_params = []
        adapter_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            elif 'adapter' in name:
                adapter_params.append(param)
            else:
                head_params.append(param)
                
        # Parameter groups dengan different learning rates
        param_groups = [
            {'params': backbone_params, 'lr': self.backbone_lr},
            {'params': adapter_params, 'lr': self.config['training']['lr0']},
            {'params': head_params, 'lr': self.config['training']['lr0']}
        ]
        
        # Setup optimizer
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup scheduler
        lrf = self.config['training']['lrf']
        total_steps = self.config['training']['epochs']
        warmup_steps = self.config['training'].get('warmup_epochs', 3)
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            final_lr_factor=lrf
        )
        
        return optimizer, scheduler
        
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step dengan gradient clipping dan
        learning rate warmup untuk EfficientNet backbone
        """
        metrics = super().train_step(batch, batch_idx)
        
        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config['training'].get('max_grad_norm', 10.0)
        )
        
        return metrics
        
    def _compute_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses dengan tambahan regularization untuk
        feature adaptation
        """
        # Base losses dari YOLOv5
        losses = super()._compute_losses(predictions, targets)
        
        # Feature adaptation loss
        if self.current_epoch >= self.config['training'].get('warmup_epochs', 3):
            adapter_loss = self._compute_adapter_loss()
            losses['adapter_loss'] = adapter_loss * self.config['training'].get(
                'adapter_loss_weight',
                0.1
            )
            
        return losses
        
    def _compute_adapter_loss(self) -> torch.Tensor:
        """
        Compute regularization loss untuk feature adaptation
        antara EfficientNet dan YOLOv5
        """
        adapter_loss = torch.tensor(0.0, device=self.device)
        
        # Get feature maps
        efficient_features = self.model.backbone.features
        adapted_features = self.model.adapter(efficient_features)
        
        # Compute similarity loss
        for ef, af in zip(efficient_features, adapted_features):
            # Channel attention similarity
            ef_att = self._compute_channel_attention(ef)
            af_att = self._compute_channel_attention(af)
            
            adapter_loss += torch.nn.functional.mse_loss(
                ef_att,
                af_att
            )
            
        return adapter_loss
        
    def _compute_channel_attention(
        self,
        feature: torch.Tensor
    ) -> torch.Tensor:
        """Compute channel attention weights"""
        b, c, h, w = feature.shape
        
        # Global average pooling
        pool = feature.mean(dim=[2, 3])
        
        # Channel attention
        attention = torch.nn.functional.softmax(pool, dim=1)
        
        return attention

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Learning rate scheduler dengan warmup dan cosine decay"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        final_lr_factor: float = 0.01,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.final_lr_factor = final_lr_factor
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        if not self._get_lr_called_within_step:
            self.logger.warning(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
            
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            factor = float(step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            factor = self.final_lr_factor + (1 - self.final_lr_factor) * (
                1 + math.cos(math.pi * progress)
            ) / 2
            
        return [base_lr * factor for base_lr in self.base_lrs]