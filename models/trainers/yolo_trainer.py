# File: models/trainers/yolo_trainer.py
# Author: Alfrida Sabar
# Deskripsi: Base trainer untuk YOLOv5 yang mendukung berbagai backbone

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np
from tqdm.auto import tqdm

from .base_trainer import BaseTrainer
from utils.logger import SmartCashLogger

class YOLOv5BaseTrainer(BaseTrainer):
    """Base trainer untuk YOLOv5 dengan komponen umum"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__(model, config, logger)
        
        # Loss weights
        self.loss_weights = {
            'box': self.config['training'].get('box_loss_weight', 0.05),
            'obj': self.config['training'].get('obj_loss_weight', 1.0),
            'cls': self.config['training'].get('cls_loss_weight', 0.5)
        }
        
        # Setup optimizer dan scheduler
        self.optimizer, self.scheduler = self.configure_optimizers()
        
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Common training step untuk semua YOLOv5 variants"""
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
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.config['training'].get('max_grad_norm', 10.0)
        )
        
        self.optimizer.step()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
            
        metrics = {
            'loss': total_loss,
            **losses
        }
        
        return metrics
        
    def _compute_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Abstract method untuk loss calculation"""
        raise NotImplementedError
        
class YOLOv5CSPTrainer(YOLOv5BaseTrainer):
    """Trainer untuk YOLOv5 dengan CSPDarknet backbone"""
    
    def _compute_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        CSPDarknet specific loss calculation
        - Single feature map output
        - Standard YOLO anchor boxes
        """
        # Unpack predictions
        pred_box = predictions[..., :4]  # x,y,w,h
        pred_conf = predictions[..., 4]  # objectness
        pred_cls = predictions[..., 5:]  # class probabilities
        
        # Calculate losses menggunakan format CSPDarknet
        box_loss = self._box_loss(pred_box, targets[..., :4])
        obj_loss = self._objectness_loss(pred_conf, targets[..., 4])
        cls_loss = self._classification_loss(pred_cls, targets[..., 5])
        
        return {
            'box_loss': box_loss * self.loss_weights['box'],
            'obj_loss': obj_loss * self.loss_weights['obj'],
            'cls_loss': cls_loss * self.loss_weights['cls']
        }
        
class YOLOv5EfficientTrainer(YOLOv5BaseTrainer):
    """Trainer untuk YOLOv5 dengan EfficientNet backbone"""
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        super().__init__(model, config, logger)
        
        # Anchor boxes per scale
        self.anchors = self._generate_anchors()
        
    def _generate_anchors(self) -> List[torch.Tensor]:
        """Generate anchor boxes untuk setiap skala deteksi"""
        base_anchors = torch.tensor(self.config['model'].get('anchors', [
            [[10, 13], [16, 30], [33, 23]],     # P3 anchors
            [[30, 61], [62, 45], [59, 119]],    # P4 anchors
            [[116, 90], [156, 198], [373, 326]]  # P5 anchors
        ]))
        return [anchors.float().to(self.device) for anchors in base_anchors]
        
    def _compute_losses(
        self,
        predictions: List[torch.Tensor],  # List dari P3, P4, P5 predictions
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        EfficientNet specific loss calculation
        - Multi-scale feature maps
        - Scale-specific anchor boxes
        """
        scale_losses = []
        
        # Calculate loss untuk setiap skala
        for scale_idx, (pred, anchors) in enumerate(zip(predictions, self.anchors)):
            scale_losses.append(self._compute_scale_losses(
                predictions=pred,
                targets=targets,
                anchors=anchors,
                scale_idx=scale_idx
            ))
            
        # Combine losses dari semua skala
        combined_losses = {}
        for key in ['box_loss', 'obj_loss', 'cls_loss']:
            combined_losses[key] = sum(loss[key] for loss in scale_losses)
            
        return combined_losses
        
    def _compute_scale_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        anchors: torch.Tensor,
        scale_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Compute losses untuk satu skala deteksi"""
        # Unpack predictions
        pred_box = predictions[..., :4]
        pred_conf = predictions[..., 4]
        pred_cls = predictions[..., 5:]
        
        # Match predictions dengan targets untuk skala ini
        matches, match_indices = self._match_predictions(
            pred_box,
            targets,
            anchors,
            scale_idx
        )
        
        # Calculate losses
        box_loss = self._box_loss(
            pred_box[matches],
            targets[match_indices][..., :4]
        )
        
        obj_loss = self._objectness_loss(
            pred_conf,
            matches
        )
        
        cls_loss = self._classification_loss(
            pred_cls[matches],
            targets[match_indices][..., 4]
        )
        
        return {
            'box_loss': box_loss * self.loss_weights['box'],
            'obj_loss': obj_loss * self.loss_weights['obj'],
            'cls_loss': cls_loss * self.loss_weights['cls']
        }