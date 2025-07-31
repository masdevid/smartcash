"""
File: smartcash/model/training/losses/base_loss.py
Description: Core YOLO loss implementation for currency detection
Responsibility: Pure YOLO loss computation with IoU, classification, and objectness losses
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional, Union

from smartcash.common.logger import get_logger


class YOLOLoss(nn.Module):
    """Core YOLO loss implementation for currency detection"""
    
    def __init__(self, num_classes: int = 7, anchors: Optional[List] = None, 
                 box_weight: float = 0.05, obj_weight: float = 4.0, cls_weight: float = 0.5,
                 focal_loss: bool = False, label_smoothing: float = 0.0, logger=None):
        """
        Initialize YOLO loss function.
        
        Args:
            num_classes: Number of classes for detection
            anchors: Anchor boxes for different scales
            box_weight: Weight for bounding box regression loss
            obj_weight: Weight for objectness loss
            cls_weight: Weight for classification loss
            focal_loss: Whether to use focal loss for classification
            label_smoothing: Label smoothing factor
            logger: Logger instance
        """
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.focal_loss = focal_loss
        self.label_smoothing = label_smoothing
        
        # Initialize logger
        self.logger = logger or self._create_default_logger()
        
        # Default anchors optimized for banknote detection (smaller objects)
        if anchors is None:
            self.anchors = torch.tensor([
                [[8, 11], [12, 18], [18, 28]],       # P3/8 - Smaller for banknotes
                [[24, 36], [36, 48], [48, 72]],      # P4/16 - Medium banknote sizes  
                [[72, 96], [96, 128], [144, 192]]    # P5/32 - Large banknote sizes
            ]).float()
        else:
            self.anchors = torch.tensor(anchors).float()
            
        # Store original anchor count for validation
        self.original_anchor_count = len(self.anchors)
        
        # Loss functions
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Balance weights for different scales
        self.balance = [4.0, 1.0, 0.4]
        
    def _create_default_logger(self):
        """Create a default logger if none is provided"""
        import logging
        import sys
        
        logger = logging.getLogger('YOLOLoss')
        logger.setLevel(logging.WARNING)  # Default to warning level
        
        # Create console handler
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.WARNING)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        return logger
    
    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor, 
                img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate YOLO loss
        
        Args:
            predictions: List of predictions from 3 scales [P3, P4, P5]
            targets: [num_targets, 6] format: [batch_idx, class, x, y, w, h]
            img_size: Input image size
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with breakdown loss components
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        # Build targets for each scale
        from .target_builder import build_targets_for_yolo
        tcls, tbox, indices, anchors = build_targets_for_yolo(
            predictions, targets, self.anchors, img_size, self.logger
        )
        
        # Calculate loss for each scale
        max_scales = len(self.anchors) if hasattr(self, 'anchors') and self.anchors is not None else 3
        
        # Auto-expand anchors if we have more prediction scales
        if len(predictions) > max_scales:
            self._expand_anchors_for_scales(len(predictions))
            max_scales = len(self.anchors)
            self.logger.info(f"Expanded anchors to handle {len(predictions)} prediction scales")
        
        for i, pred in enumerate(predictions[:max_scales]):  # Only process scales with anchors
            # Safe indexing with bounds checking
            if i >= len(indices) or i >= len(anchors) or i >= len(tcls) or i >= len(tbox):
                continue
            
            # Ensure predictions are on the same device as targets
            pred = pred.to(device=targets.device)
            
            # Process predictions and compute losses
            scale_losses = self._compute_scale_losses(
                pred, tcls[i], tbox[i], indices[i], anchors[i], i, device
            )
            
            lcls += scale_losses['cls_loss']
            lbox += scale_losses['box_loss'] 
            lobj += scale_losses['obj_loss']
        
        # Apply loss weights
        lbox = lbox * self.box_weight
        lobj = lobj * self.obj_weight
        lcls = lcls * self.cls_weight
        
        # Calculate total loss
        total_loss = lbox + lobj + lcls
        
        # Return loss breakdown
        return total_loss, {
            'box_loss': lbox.detach(),
            'obj_loss': lobj.detach(),
            'cls_loss': lcls.detach(),
            'total_loss': total_loss.detach()
        }
    
    def _expand_anchors_for_scales(self, num_scales: int) -> None:
        """
        Expand anchors to handle more prediction scales than originally defined
        
        Args:
            num_scales: Total number of scales needed
        """
        current_scales = len(self.anchors)
        if num_scales <= current_scales:
            return
            
        # Generate additional anchor scales by interpolating/extrapolating
        device = self.anchors.device
        anchors_list = self.anchors.tolist()
        
        for scale_idx in range(current_scales, num_scales):
            if scale_idx < 3:
                # Use predefined patterns for common scales
                if scale_idx == 0:
                    new_anchors = [[10, 13], [16, 30], [33, 23]]  # P3
                elif scale_idx == 1:
                    new_anchors = [[30, 61], [62, 45], [59, 119]]  # P4
                else:
                    new_anchors = [[116, 90], [156, 198], [373, 326]]  # P5
            else:
                # Extrapolate for additional scales by scaling up the last set
                scale_factor = 1.5 ** (scale_idx - 2)  # Exponential scaling
                base_anchors = anchors_list[-1]  # Use last anchor set as base
                new_anchors = [[int(w * scale_factor), int(h * scale_factor)] 
                              for w, h in base_anchors]
            
            anchors_list.append(new_anchors)
        
        # Update the anchor tensor
        self.anchors = torch.tensor(anchors_list, device=device, dtype=torch.float32)
    
    def _compute_scale_losses(self, pred: torch.Tensor, tcls: torch.Tensor, 
                             tbox: torch.Tensor, indices: Tuple, anchors: torch.Tensor,
                             scale_idx: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Compute losses for a single scale
        
        Args:
            pred: Predictions for this scale
            tcls: Target classes for this scale
            tbox: Target boxes for this scale
            indices: Target indices (batch, anchor, grid_y, grid_x)
            anchors: Anchors for this scale
            scale_idx: Scale index
            device: Device for computations
            
        Returns:
            Dictionary containing scale losses
        """
        # Initialize losses
        cls_loss = torch.zeros(1, device=device)
        box_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)
        
        # Get the current scale's predictions and ensure correct shape
        pred = self._ensure_prediction_format(pred, device)
        
        # Initialize target objectness
        tobj = torch.zeros_like(pred[..., 0])
        
        # Get target indices for this scale if available
        if len(indices) == 4:
            b, a, gj, gi = indices  # batch, anchor, grid_y, grid_x
            
            if len(b) > 0:
                # Ensure indices are within bounds
                b = b.clamp(0, pred.shape[0] - 1)
                a = a.clamp(0, pred.shape[1] - 1)
                gj = gj.clamp(0, pred.shape[2] - 1)
                gi = gi.clamp(0, pred.shape[3] - 1)
                
                # Get predictions for matched anchors
                ps = pred[b, a, gj, gi]  # [num_matches, num_classes + 5]
                
                # Compute box loss
                box_loss = self._compute_box_loss(ps, tbox, anchors, device)
                
                # Compute classification loss
                if self.num_classes > 1 and len(tcls) > 0:
                    cls_loss = self._compute_classification_loss(ps, tcls, device)
                
                # Update objectness targets
                tobj = self._update_objectness_targets(tobj, ps, tbox, b, a, gj, gi, device)
        
        # Calculate objectness loss for this scale
        if tobj.numel() > 0:
            pred_obj = pred[..., 4]  # Shape: [batch, anchors, grid, grid]
            
            # Ensure pred_obj is 4D [batch, anchors, grid, grid]
            if pred_obj.dim() == 3:
                pred_obj = pred_obj.unsqueeze(1)  # Add anchor dimension if missing
            
            # Ensure tobj has the same shape as pred_obj
            if pred_obj.shape != tobj.shape:
                new_tobj = torch.zeros_like(pred_obj)
                if tobj.any().item():
                    # Copy valid values with bounds checking
                    max_batch = min(new_tobj.size(0), tobj.size(0))
                    max_anchor = min(new_tobj.size(1), tobj.size(1))
                    max_h = min(new_tobj.size(2), tobj.size(2))
                    max_w = min(new_tobj.size(3), tobj.size(3))
                    
                    new_tobj[:max_batch, :max_anchor, :max_h, :max_w] = \
                        tobj[:max_batch, :max_anchor, :max_h, :max_w]
                tobj = new_tobj
            
            # Calculate BCE loss with proper reduction
            obj_loss_raw = self.bce_obj(pred_obj, tobj).mean()
            
            # Apply balance weight if needed
            balance_weight = self.balance[scale_idx] if scale_idx < len(self.balance) else 1.0
            obj_loss = obj_loss_raw * balance_weight
        
        return {
            'cls_loss': cls_loss,
            'box_loss': box_loss,
            'obj_loss': obj_loss
        }
    
    def _ensure_prediction_format(self, pred: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Ensure prediction tensor is in correct 5D format"""
        pred_shape = pred.shape
        
        if len(pred_shape) == 5:  # [batch, anchors, grid_y, grid_x, features]
            return pred
        elif len(pred_shape) == 4:  # [batch, num_anchors * (num_classes + 5), grid_y, grid_x]
            # Reshape to [batch, anchors, grid_y, grid_x, num_classes + 5]
            na = 3  # Number of anchors per scale
            pred = pred.view(
                pred_shape[0], na, -1, pred_shape[2], pred_shape[3]
            ).permute(0, 1, 3, 4, 2).contiguous()
            return pred
        elif len(pred_shape) == 3:  # [batch, num_anchors * grid_y * grid_x, num_classes + 5]
            batch_size, total_predictions, num_features = pred_shape
            
            # Calculate grid size
            na = 3
            grid_size_squared = total_predictions // na
            grid_size = int(math.isqrt(grid_size_squared))
            grid_size = max(8, min(80, grid_size))
            grid_size = 2 ** int(round(math.log2(grid_size)))
            
            try:
                pred = pred.view(
                    batch_size, na, grid_size, grid_size, num_features
                ).contiguous()
                return pred
            except RuntimeError as e:
                self.logger.warning(f"Failed to reshape prediction tensor {pred_shape}: {e}")
                # Return zero tensor as fallback
                return torch.zeros((batch_size, na, grid_size, grid_size, num_features), 
                                  device=device)
        else:
            raise ValueError(f"Unexpected prediction shape: {pred_shape}")
    
    def _compute_box_loss(self, ps: torch.Tensor, tbox: torch.Tensor, 
                         anchors: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute bounding box regression loss"""
        if ps.numel() == 0 or tbox.numel() == 0:
            return torch.zeros(1, device=device)
        
        # Regression (box) loss with gradient clipping for stability
        pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
        anchor_tensor = anchors.to(device=ps.device)
        pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchor_tensor
        pbox = torch.cat((pxy, pwh), 1).clamp(0, 1)
        
        # Calculate IoU and box loss
        iou = self._bbox_iou(pbox.T, tbox.to(device=pbox.device), x1y1x2y2=False)
        return (1.0 - iou).mean()
    
    def _compute_classification_loss(self, ps: torch.Tensor, tcls: torch.Tensor, 
                                   device: torch.device) -> torch.Tensor:
        """Compute classification loss"""
        if ps.numel() == 0 or tcls.numel() == 0:
            return torch.zeros(1, device=device)
        
        t = torch.full_like(ps[:, 5:], self.cn, device=ps.device)  # targets
        if t.numel() > 0 and tcls.numel() > 0:
            valid_indices = tcls.to(device=ps.device).clamp(0, t.shape[1] - 1)
            # Ensure we only index as many rows as we have target classes
            num_targets = min(t.shape[0], len(valid_indices))
            if num_targets > 0:
                t[range(num_targets), valid_indices[:num_targets]] = self.cp
        
        return self._classification_loss(ps[:, 5:], t)
    
    def _update_objectness_targets(self, tobj: torch.Tensor, ps: torch.Tensor, 
                                  tbox: torch.Tensor, b: torch.Tensor, a: torch.Tensor,
                                  gj: torch.Tensor, gi: torch.Tensor, 
                                  device: torch.device) -> torch.Tensor:
        """Update objectness targets based on IoU"""
        # Calculate IoU for objectness targets
        pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
        
        # Get the appropriate anchors for each prediction using the anchor indices
        if len(a) > 0 and hasattr(self, 'anchors') and self.anchors is not None:
            # Use the correct anchor for each prediction based on anchor index
            anchor_scale = self.anchors[0].to(device=ps.device)  # Get anchor scale
            if len(a) <= anchor_scale.shape[0]:
                # Use anchor indices to select appropriate anchor sizes
                selected_anchors = anchor_scale[a.clamp(0, anchor_scale.shape[0] - 1)]
            else:
                # If we have more predictions than anchor indices, repeat the pattern
                selected_anchors = anchor_scale[a % anchor_scale.shape[0]]
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * selected_anchors
        else:
            # Fallback: use a default anchor size
            default_anchor = torch.tensor([16.0, 16.0], device=ps.device)
            pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * default_anchor
        
        pbox = torch.cat((pxy, pwh), 1).clamp(0, 1)
        
        iou = self._bbox_iou(pbox.T, tbox.to(device=pbox.device), x1y1x2y2=False)
        score_iou = iou.detach().clamp(0).type(tobj.dtype)
        
        # Create a target tensor for objectness
        target_obj = torch.zeros_like(tobj)
        
        # Ensure indices are within bounds
        b = b.clamp(0, target_obj.size(0) - 1)
        a = a.clamp(0, target_obj.size(1) - 1)
        gj = gj.clamp(0, target_obj.size(2) - 1)
        gi = gi.clamp(0, target_obj.size(3) - 1)
        
        # Create a mask for valid indices
        valid_mask = (b >= 0) & (b < target_obj.size(0)) & \
                    (a >= 0) & (a < target_obj.size(1)) & \
                    (gj >= 0) & (gj < target_obj.size(2)) & \
                    (gi >= 0) & (gi < target_obj.size(3))
        
        if valid_mask.any().item():
            # Update objectness target only for valid indices
            target_obj[b[valid_mask], a[valid_mask], gj[valid_mask], gi[valid_mask]] = 1.0
            
            # Smooth label if enabled
            if self.label_smoothing > 0:
                target_obj = target_obj * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        return target_obj.to(tobj.device, non_blocking=True)
    
    def _bbox_iou(self, box1: torch.Tensor, box2: torch.Tensor, 
                  x1y1x2y2: bool = True, eps: float = 1e-7) -> torch.Tensor:
        """
        Calculate IoU between two sets of boxes
        
        Args:
            box1: Tensor of shape [N, 4] or [4] for first set of boxes
            box2: Tensor of shape [M, 4] or [4] for second set of boxes
            x1y1x2y2: If True, boxes are in [x1, y1, x2, y2] format
                     If False, boxes are in [x_center, y_center, width, height] format
            eps: Small value to avoid division by zero
            
        Returns:
            iou: IoU values between box1 and box2
        """
        # Ensure boxes are at least 2D and contiguous
        box1 = box1.reshape(-1, 4).contiguous()
        box2 = box2.reshape(-1, 4).contiguous()
        
        if x1y1x2y2:
            # Already in x1y1x2y2 format
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
        else:
            # Convert from xywh to x1y1x2y2
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        
        # Expand dimensions for broadcasting
        b1_x1, b1_y1 = b1_x1.unsqueeze(1), b1_y1.unsqueeze(1)
        b1_x2, b1_y2 = b1_x2.unsqueeze(1), b1_y2.unsqueeze(1)
        b2_x1, b2_y1 = b2_x1.unsqueeze(0), b2_y1.unsqueeze(0)
        b2_x2, b2_y2 = b2_x2.unsqueeze(0), b2_y2.unsqueeze(0)
        
        # Intersection coordinates
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        # Intersection area
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
        
        # Union Area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + eps
        
        # IoU
        iou = inter_area / union_area
        
        # If one of the inputs was a single box, squeeze the output
        if iou.dim() > 1 and (iou.size(0) == 1 or iou.size(1) == 1):
            iou = iou.squeeze()
            
        return iou
    
    def _classification_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate classification loss with optional focal loss"""
        if self.focal_loss:
            return self._focal_loss(pred, target)
        else:
            return self.bce_cls(pred, target).mean()
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss implementation"""
        bce_loss = self.bce_cls(pred, target)
        p_t = torch.exp(-bce_loss)
        focal_weight = alpha * (1 - p_t) ** gamma
        return (focal_weight * bce_loss).mean()
    
    @property
    def gr(self) -> float:
        """Objectness gradient ratio"""
        return 1.0
    
    @property
    def cp(self) -> float:
        """Class positive label value"""
        return 1.0 - 0.5 * self.label_smoothing
    
    @property
    def cn(self) -> float:
        """Class negative label value"""
        return 0.5 * self.label_smoothing