# File: models/trainers/yolo_trainer.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi trainer untuk YOLOv5 dengan CSPDarknet backbone

import torch
from typing import Dict, Optional, Tuple
import numpy as np

from .base_trainer import BaseTrainer
from utils.logger import SmartCashLogger

class YOLOv5Trainer(BaseTrainer):
    """Trainer untuk YOLOv5 dengan CSPDarknet backbone"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__(model, config, logger)
        
        # Setup loss weights
        self.loss_weights = {
            'box': self.config['training'].get('box_loss_weight', 0.05),
            'obj': self.config['training'].get('obj_loss_weight', 1.0),
            'cls': self.config['training'].get('cls_loss_weight', 0.5)
        }
        
        # Setup optimizer dan scheduler
        self.optimizer = self.configure_optimizers()
        
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Setup optimizer dan learning rate scheduler"""
        # Setup optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['training']['lr0'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup scheduler
        lrf = self.config['training']['lrf']  # final learning rate
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['training']['lr0'],
            total_steps=self.config['training']['epochs'],
            pct_start=self.config['training'].get('warmup_epochs', 3) / 
                      self.config['training']['epochs'],
            final_div_factor=1/lrf
        )
        
        return optimizer, scheduler
        
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Training step untuk YOLOv5
        Args:
            batch: Tuple (images, targets)
            batch_idx: Index batch
        Returns:
            Dict metrics
        """
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        predictions = self.model(images)
        
        # Calculate losses
        losses = self._compute_losses(predictions, targets)
        total_loss = sum(losses.values())
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        metrics = {
            'loss': total_loss,
            **losses
        }
        
        return metrics
        
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step untuk YOLOv5
        Args:
            batch: Tuple (images, targets)
            batch_idx: Index batch
        Returns:
            Dict metrics
        """
        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            predictions = self.model(images)
            
        # Calculate losses dan metrics
        losses = self._compute_losses(predictions, targets)
        metrics = self._compute_metrics(predictions, targets)
        
        return {
            'val_loss': sum(losses.values()),
            **{f'val_{k}': v for k, v in losses.items()},
            **{f'val_{k}': v for k, v in metrics.items()}
        }
        
    def _compute_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Hitung losses YOLOv5
        Args:
            predictions: Output model
            targets: Ground truth
        Returns:
            Dict losses
        """
        # Box loss (CIoU)
        box_loss = self._compute_box_loss(predictions, targets)
        
        # Objectness loss (BCE)
        obj_loss = self._compute_obj_loss(predictions, targets)
        
        # Classification loss (BCE)
        cls_loss = self._compute_cls_loss(predictions, targets)
        
        return {
            'box_loss': box_loss * self.loss_weights['box'],
            'obj_loss': obj_loss * self.loss_weights['obj'],
            'cls_loss': cls_loss * self.loss_weights['cls']
        }
        
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Hitung metrics evaluasi
        Args:
            predictions: Output model
            targets: Ground truth
        Returns:
            Dict metrics
        """
        # Compute precision, recall, mAP
        metrics = {}
        
        # Precision & Recall per class
        for cls in range(self.config['model']['nc']):
            prec, rec = self._compute_precision_recall(
                predictions,
                targets,
                cls
            )
            metrics[f'precision_cls_{cls}'] = prec
            metrics[f'recall_cls_{cls}'] = rec
            
        # Mean metrics
        metrics.update({
            'mAP50': self._compute_map(predictions, targets, iou_thresh=0.5),
            'mAP50_95': self._compute_map(
                predictions,
                targets,
                iou_threshs=np.arange(0.5, 1.0, 0.05)
            )
        })
        
        return metrics
        
    def _compute_box_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Complete IoU loss untuk bounding boxes"""
        pass  # TODO: Implementasi CIoU loss
        
    def _compute_obj_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate objectness loss (BCE)"""
        pass  # TODO: Implementasi objectness loss
        
    def _compute_cls_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculate classification loss (BCE)"""
        pass  # TODO: Implementasi classification loss