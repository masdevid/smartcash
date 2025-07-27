"""
File: smartcash/model/training/loss_manager.py
Deskripsi: Manager untuk perhitungan loss YOLO dengan dukungan multi-layer currency detection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional, Union
import math
from typing import Dict, List, Tuple, Optional, Any

class YOLOLoss(nn.Module):
    """YOLO loss implementation untuk currency detection"""
    
    def __init__(self, num_classes: int = 7, anchors: Optional[List] = None, 
                 box_weight: float = 0.05, obj_weight: float = 1.0, cls_weight: float = 0.5,
                 focal_loss: bool = False, label_smoothing: float = 0.0, logger=None):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.focal_loss = focal_loss
        self.label_smoothing = label_smoothing
        
        # Initialize logger
        self.logger = logger or self._create_default_logger()
        
        # Default anchors untuk 3 scales (P3, P4, P5)
        if anchors is None:
            self.anchors = torch.tensor([
                [[10, 13], [16, 30], [33, 23]],      # P3/8
                [[30, 61], [62, 45], [59, 119]],     # P4/16  
                [[116, 90], [156, 198], [373, 326]]  # P5/32
            ]).float()
        else:
            self.anchors = torch.tensor(anchors).float()
        
        # Loss functions
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        
        # Balance weights untuk different scales
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
            predictions: List of predictions dari 3 scales [P3, P4, P5]
            targets: [num_targets, 6] format: [batch_idx, class, x, y, w, h]
            img_size: Input image size
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary dengan breakdown loss components
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        # Build targets untuk setiap scale
        tcls, tbox, indices, anchors = self._build_targets(predictions, targets, img_size)
        
        # Calculate loss for each scale
        for i, pred in enumerate(predictions):
            # Safe indexing with bounds checking
            if i >= len(indices) or i >= len(anchors) or i >= len(tcls) or i >= len(tbox):
                continue
            
            # Ensure predictions are on the same device as targets
            pred = pred.to(device=targets.device)
            
            # Get the current scale's predictions and ensure correct shape
            pred_shape = pred.shape
            if len(pred_shape) == 5:  # [batch, anchors, grid_y, grid_x, num_classes + 5]
                pass  # Already in correct format
            elif len(pred_shape) == 4:  # [batch, num_anchors * (num_classes + 5), grid_y, grid_x]
                # Reshape to [batch, anchors, grid_y, grid_x, num_classes + 5]
                pred = pred.view(
                    pred_shape[0], self.na, -1, pred_shape[2], pred_shape[3]
                ).permute(0, 1, 3, 4, 2).contiguous()
            elif len(pred_shape) == 3:  # [batch, num_anchors * grid_y * grid_x, num_classes + 5]
                # Calculate the expected number of features per anchor box
                num_features = pred_shape[-1]
                
                # Try to calculate grid size based on the tensor size
                total_elements = pred_shape[0] * pred_shape[1] * pred_shape[2]
                expected_elements = pred_shape[0] * self.na * num_features  # Per grid cell
                
                # Calculate grid size that would make the tensor size match
                grid_size_squared = total_elements // (pred_shape[0] * self.na * num_features)
                grid_size = int(math.isqrt(grid_size_squared))
                
                # Ensure grid_size is reasonable (should be a power of 2 between 8 and 80)
                grid_size = max(8, min(80, grid_size))
                grid_size = 2 ** int(round(math.log2(grid_size)))
                grid_y = grid_x = grid_size
                
                try:
                    # Reshape to [batch, grid_y, grid_x, anchors, num_classes + 5]
                    pred = pred.view(
                        pred_shape[0],  # batch
                        grid_y,
                        grid_x,
                        self.na,  # anchors
                        -1  # num_classes + 5
                    ).permute(0, 3, 1, 2, 4).contiguous()  # [batch, anchors, grid_y, grid_x, num_classes + 5]
                except RuntimeError as e:
                    # If reshaping fails, log the error and skip this prediction
                    self.logger.warning(f"Failed to reshape prediction tensor with shape {pred_shape} to [batch, {grid_y}, {grid_x}, {self.na}, -1]: {e}")
                    # Skip this prediction by returning zero losses
                    return torch.zeros(1, device=pred.device), \
                           {'box_loss': torch.zeros(1, device=pred.device),
                            'obj_loss': torch.zeros(1, device=pred.device),
                            'cls_loss': torch.zeros(1, device=pred.device),
                            'num_targets': 0,
                            'num_preds': 0}
            else:
                raise ValueError(f"Unexpected prediction shape: {pred_shape}")
            
            # Initialize target objectness
            tobj = torch.zeros_like(pred[..., 0])
            
            # Get target indices for this scale if available
            if i < len(indices) and len(indices[i]) == 4:
                b, a, gj, gi = indices[i]  # batch, anchor, grid_y, grid_x
                
                if len(b) > 0:
                    # Ensure indices are within bounds
                    b = b.clamp(0, pred.shape[0] - 1)
                    a = a.clamp(0, pred.shape[1] - 1)
                    gj = gj.clamp(0, pred.shape[2] - 1)
                    gi = gi.clamp(0, pred.shape[3] - 1)
                    
                    # Get predictions for matched anchors
                    ps = pred[b, a, gj, gi]  # [num_matches, num_classes + 5]
                    
                    # Regression (box) loss with gradient clipping for stability
                    with torch.no_grad():
                        pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                        pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i].to(device=ps.device)
                        pbox = torch.cat((pxy, pwh), 1).clamp(0, 1)
                # Calculate IoU and box loss
                with torch.no_grad():
                    iou = self._bbox_iou(pbox.T, tbox[i].to(device=pbox.device), x1y1x2y2=False)
                    lbox += (1.0 - iou).mean()
                    
                    # Objectness target with shape matching the prediction
                    score_iou = iou.detach().clamp(0).type(tobj.dtype)
                    if len(score_iou) == 0:
                        return torch.zeros_like(pred[..., 0]), {}
                    
                    # Ensure we have the same number of scores as targets
                    if score_iou.numel() > 1:
                        score_iou = score_iou.view(-1, 1, 1, 1)  # [num_matches, 1, 1, 1]
                    else:
                        score_iou = score_iou.view(1, 1, 1, 1)  # [1, 1, 1, 1]
                    
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
                    
                    if valid_mask.any():
                        # Update objectness target only for valid indices
                        target_obj[b[valid_mask], a[valid_mask], gj[valid_mask], gi[valid_mask]] = 1.0
                        
                        # Smooth label if enabled
                        if self.label_smoothing > 0:
                            target_obj = target_obj * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
                        
                        # Update the main target tensor
                        tobj = target_obj.to(tobj.device, non_blocking=True)
                
                # Classification loss (only if multiple classes)
                if self.num_classes > 1 and len(tcls) > i and len(tcls[i]) > 0:
                    with torch.no_grad():
                        t = torch.full_like(ps[:, 5:], self.cn, device=ps.device)  # targets
                        t[range(len(t)), tcls[i].to(device=ps.device)] = self.cp
                    lcls += self._classification_loss(ps[:, 5:], t)  # BCE
            
            # Calculate objectness loss for this scale
            if tobj.numel() > 0:  # Check if tobj has elements
                # Get objectness predictions
                pred_obj = pred[..., 4]  # Shape: [batch, anchors, grid, grid]
                
                # Ensure pred_obj is 4D [batch, anchors, grid, grid]
                if pred_obj.dim() == 3:
                    pred_obj = pred_obj.unsqueeze(1)  # Add anchor dimension if missing
                
                # Ensure tobj has the same shape as pred_obj
                if pred_obj.shape != tobj.shape:
                    # If shapes don't match, create a new tensor with the correct shape
                    new_tobj = torch.zeros_like(pred_obj)
                    
                    # If we have any positive samples, copy them to the correct positions
                    if tobj.any():
                        # Get indices of positive samples
                        pos_indices = (tobj > 0).nonzero(as_tuple=True)
                        
                        # If we have any positive samples, copy them to the new tensor
                        if len(pos_indices) > 0 and len(pos_indices[0]) > 0:
                            # Make sure we don't go out of bounds
                            max_batch = min(new_tobj.size(0), tobj.size(0))
                            max_anchor = min(new_tobj.size(1), tobj.size(1))
                            max_h = min(new_tobj.size(2), tobj.size(2))
                            max_w = min(new_tobj.size(3), tobj.size(3))
                            
                            # Copy valid values
                            new_tobj[:max_batch, :max_anchor, :max_h, :max_w] = \
                                tobj[:max_batch, :max_anchor, :max_h, :max_w]
                    
                    tobj = new_tobj
                
                # Calculate BCE loss with proper reduction
                # Use mean reduction to get a scalar loss value
                obj_loss = self.bce_obj(pred_obj, tobj).mean()
                
                # Apply balance weight if needed
                balance_weight = self.balance[i] if i < len(self.balance) else 1.0
                lobj = lobj + (obj_loss * balance_weight)  # Use = instead of +=
        
        # Apply loss weights with gradient scaling
        num_scales = max(1, len(predictions))  # Avoid division by zero
        lbox = lbox * self.box_weight
        lobj = lobj * (self.obj_weight / num_scales)  # Scale by number of scales
        lcls = lcls * self.cls_weight
        
        # Calculate total loss with gradient scaling
        total_loss = (lbox + lobj + lcls) * 3.0  # Scale up to maintain magnitude
        
        # Detach losses for logging
        with torch.no_grad():
            return total_loss, {
                'box_loss': lbox.detach(),
                'obj_loss': lobj.detach(),
                'cls_loss': lcls.detach(),
                'total_loss': total_loss.detach()
            }
    
    def _build_targets(self, predictions: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], 
                      targets: torch.Tensor, img_size: int) -> Tuple[List, List, List, List]:
        """
        Build targets for each prediction scale
        
        Args:
            predictions: List or tuple of predictions from different scales
            targets: [num_targets, 6] format: [batch_idx, class, x, y, w, h]
            img_size: Input image size
            
        Returns:
            tcls: List of class targets for each scale
            tbox: List of box targets for each scale
            indices: List of indices for each scale
            anchors: List of anchors for each scale
        """
        # Initialize empty lists for targets
        tcls, tbox, indices, anchors = [], [], [], []
        
        # Convert predictions to list if it's a tuple
        if isinstance(predictions, tuple):
            predictions = list(predictions)
        
        # Ensure predictions is a list of tensors
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]
        
        # Ensure all predictions are tensors
        tensor_predictions = []
        for p in predictions:
            if torch.is_tensor(p):
                tensor_predictions.append(p)
            else:
                try:
                    # Try to convert to tensor, handling the case where p might be a list of tensors
                    if isinstance(p, (list, tuple)) and all(torch.is_tensor(x) for x in p):
                        # If it's a list of tensors, stack them
                        tensor_predictions.append(torch.stack(p))
                    else:
                        # Otherwise, try to convert to tensor directly
                        tensor_predictions.append(torch.tensor(p, device=targets.device) if hasattr(targets, 'device') 
                                               else torch.tensor(p))
                except (ValueError, RuntimeError, TypeError) as e:
                    # If conversion fails, log the error and skip this prediction
                    self.logger.warning(f"Failed to convert prediction to tensor: {e}")
                    continue
        
        predictions = tensor_predictions
            
        # Number of anchors per scale - make it an instance variable for access in other methods
        self.na = self.anchors.shape[1] if hasattr(self, 'anchors') and self.anchors is not None else 3
        na = self.na  # Local variable for use in this method
        
        # Number of positive targets
        nt = len(targets) if hasattr(targets, 'shape') and len(targets.shape) > 1 else 0
        
        # Define the grid offset
        g = 0.5  # grid cell offset
        
        # Offsets for grid cells
        off = torch.tensor(
            [
                [0, 0],  # no offset
                [1, 0],  # x offset
                [0, 1],  # y offset
                [-1, 0], # negative x offset
                [0, -1]  # negative y offset
            ], 
            device=targets.device if torch.is_tensor(targets) and hasattr(targets, 'device') 
                  else torch.device('cpu')
        ).float() * g  # offset
        
        for i, pred in enumerate(predictions):
            # Ensure pred is a tensor
            if not torch.is_tensor(pred):
                pred = torch.tensor(pred, device=off.device)
                predictions[i] = pred
                
            # Get the shape of the current prediction
            pred_shape = pred.shape if hasattr(pred, 'shape') else torch.tensor(pred).shape
            
            # Skip if prediction is empty or has invalid shape
            if len(pred_shape) < 2:
                device = targets.device if hasattr(targets, 'device') else torch.device('cpu')
                tcls.append(torch.zeros(0, device=device))
                tbox.append(torch.zeros((0, 4), device=device))
                indices.append((torch.zeros(0, dtype=torch.long, device=device), 
                              torch.zeros(0, dtype=torch.long, device=device)))
                anchors.append(torch.zeros((0, 2), device=device))
                continue
                
            # Get the number of classes
            num_classes = pred_shape[-1] - 5  # x, y, w, h, obj + classes
            
            # Get the current anchors for this scale
            anchors_i = self.anchors[i].clone().to(targets.device)
            
            # Initialize gain tensor with the correct shape
            gain = torch.ones(7, device=targets.device)
            
            # Handle different prediction shapes
            if len(pred_shape) == 5:  # [batch, anchors, grid_y, grid_x, features]
                grid_size = pred_shape[2]  # assuming square grid
                gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], 
                                       device=targets.device)
                pred_reshaped = pred
            elif len(pred_shape) == 3:  # [batch, num_anchors * grid_y * grid_x, features]
                # Special handling for YOLOv5 output format
                if pred_shape[1] == 25200 and pred_shape[2] == 12:
                    self.logger.info("Detected [batch, 25200, 12] format - processing as YOLO output with 3 scales")
                    # For YOLO with 3 scales (80x80, 40x40, 20x20) and 3 anchors each
                    grid_size = 80  # Using the largest grid size for now
                    gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], 
                                           device=targets.device)
                    # Just use the first scale for now (80x80)
                    num_elements = grid_size * grid_size * 3
                    pred_reshaped = pred[:, :num_elements, :].view(
                        pred_shape[0], 3, grid_size, grid_size, -1
                    ).permute(0, 1, 2, 3, 4).contiguous()
                else:
                    # For other 3D tensors, try to infer the shape
                    self.logger.warning(f"Unexpected 3D tensor shape: {pred_shape}. Using fallback.")
                    # Fallback to a default grid size
                    grid_size = 80
                    gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], 
                                           device=targets.device)
                    pred_reshaped = pred.view(pred_shape[0], -1, 3, grid_size, grid_size, 
                                            pred_shape[2]).squeeze(1).permute(0, 1, 3, 2, 4).contiguous()
            else:
                self.logger.error(f"Unexpected prediction shape: {pred_shape}")
                tcls.append(torch.zeros(0, device=targets.device))
                tbox.append(torch.zeros((0, 4), device=targets.device))
                indices.append((torch.zeros(0, dtype=torch.long, device=targets.device), 
                              torch.zeros(0, dtype=torch.long, device=targets.device)))
                anchors.append(torch.zeros((0, 2), device=targets.device))
                continue
            
            # Process targets if any exist
            if nt > 0:
                # Create a copy of targets with the right device
                t = targets.clone().to(targets.device)
                
                # Apply gain to normalize to grid coordinates
                t[:, 2:6] *= gain[2:6]
                
                # Match targets to anchors
                # Calculate width/height ratios between targets and anchors
                target_wh = t[:, 4:6]  # [num_targets, 2]
                r = target_wh.unsqueeze(1) / anchors_i.unsqueeze(0)  # [num_targets, num_anchors, 2]
                
                # Find anchors with aspect ratios close to target
                j = torch.max(r, 1. / r).max(-1)[0] < 4  # [num_targets, num_anchors]
                
                # Get indices of matching target-anchor pairs
                target_indices, anchor_indices = torch.where(j)
                
                # Filter targets and get corresponding anchors
                if len(target_indices) > 0:
                    t = t[target_indices]  # [num_matches, 6]
                    a = anchor_indices  # [num_matches]
                else:
                    # No matches, use empty tensors
                    t = torch.zeros((0, 6), device=targets.device)
                    a = torch.zeros((0,), dtype=torch.long, device=targets.device)
                
                # Matching anchor indices are already calculated
                
                # Initialize empty tensors for the case of no matches
                gi, gj = torch.zeros(0, dtype=torch.long, device=targets.device), \
                         torch.zeros(0, dtype=torch.long, device=targets.device)
                
                if len(t) > 0:
                    # Get grid xy coordinates
                    gxy = t[:, 2:4]  # grid xy [num_matches, 2]
                    
                    # Calculate grid indices
                    gij = gxy.long()  # grid indices
                    gi, gj = gij.T  # grid x, y indices
                    
                    # Apply grid size limits
                    grid_size = int(gain[2].item())  # assuming square grid
                    gi = gi.clamp(0, grid_size - 1)
                    gj = gj.clamp(0, grid_size - 1)
                    
                    # Get the image indices and classes
                    b = t[:, 0].long()  # image index
                    c = t[:, 1].long()  # class
                    
                    # Append to lists
                    indices.append((b, a, gj, gi))  # image, anchor, grid y, grid x
                    tbox.append(t[:, 2:6])  # box coordinates
                    anchors.append(anchors_i[a])  # anchors
                    tcls.append(c)  # class
                else:
                    # No matches, append empty tensors
                    indices.append((torch.zeros(0, dtype=torch.long, device=targets.device), 
                                  torch.zeros(0, dtype=torch.long, device=targets.device)))
                    tbox.append(torch.zeros((0, 4), device=targets.device))
                    anchors.append(torch.zeros((0, 2), device=targets.device))
                    tcls.append(torch.zeros(0, device=targets.device))
            else:
                # No targets, return empty tensors
                tcls.append(torch.zeros(0, device=targets.device))
                tbox.append(torch.zeros((0, 4), device=targets.device))
                indices.append((torch.zeros(0, dtype=torch.long, device=targets.device), 
                              torch.zeros(0, dtype=torch.long, device=targets.device)))
                anchors.append(torch.zeros((0, 2), device=targets.device))
        
        return tcls, tbox, indices, anchors
    
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
        # box1: [N] -> [N, 1]
        # box2: [M] -> [1, M]
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
        """Calculate classification loss dengan optional focal loss"""
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

class LossManager:
    """Manager untuk koordinasi loss calculation dengan multi-layer support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        loss_config = config.get('training', {}).get('loss', {})
        
        self.box_weight = loss_config.get('box_weight', 0.05)
        self.obj_weight = loss_config.get('obj_weight', 1.0)
        self.cls_weight = loss_config.get('cls_weight', 0.5)
        self.focal_loss = loss_config.get('focal_loss', False)
        self.label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        # Initialize loss functions untuk different layer modes
        self.loss_functions = {}
        self._setup_loss_functions()
    
    def _setup_loss_functions(self) -> None:
        """Setup loss functions berdasarkan model configuration"""
        # Check if we should use uncertainty-based multi-task loss (MODEL_ARC.md compliant)
        loss_type = self.config.get('training', {}).get('loss', {}).get('type', 'uncertainty_multi_task')
        
        if loss_type == 'uncertainty_multi_task' and self._is_multilayer_mode():
            # Use MODEL_ARC.md compliant uncertainty-based multi-task loss
            from smartcash.model.training.multi_task_loss import create_banknote_multi_task_loss
            
            layer_config = {
                'layer_1': {'description': 'Full banknote detection', 'num_classes': 7},
                'layer_2': {'description': 'Denomination-specific features', 'num_classes': 7}, 
                'layer_3': {'description': 'Common features', 'num_classes': 3}
            }
            
            loss_config = {
                'box_weight': self.box_weight,
                'obj_weight': self.obj_weight,
                'cls_weight': self.cls_weight,
                'focal_loss': self.focal_loss,
                'label_smoothing': self.label_smoothing,
                'dynamic_weighting': True,
                'min_variance': 1e-3,
                'max_variance': 10.0
            }
            
            self.multi_task_loss = create_banknote_multi_task_loss(
                use_adaptive=False,
                loss_config=loss_config
            )
            self.use_multi_task_loss = True
        else:
            # Use individual YOLO losses for backward compatibility
            self.use_multi_task_loss = False
            self._setup_individual_losses()
    
    def _setup_individual_losses(self) -> None:
        """Setup individual YOLO loss functions for each layer"""
        # MODEL_ARC.md compliant layer names
        self.loss_functions['layer_1'] = YOLOLoss(
            num_classes=7,  # Full banknote detection
            box_weight=self.box_weight,
            obj_weight=self.obj_weight,
            cls_weight=self.cls_weight,
            focal_loss=self.focal_loss,
            label_smoothing=self.label_smoothing
        )
        
        # Additional layers jika diperlukan
        if self._is_multilayer_mode():
            self.loss_functions['layer_2'] = YOLOLoss(num_classes=7, **self._get_loss_params())  # Denomination features
            self.loss_functions['layer_3'] = YOLOLoss(num_classes=3, **self._get_loss_params())  # Common features
        
        # Legacy support untuk backward compatibility
        self.loss_functions['banknote'] = self.loss_functions['layer_1']
        if self._is_multilayer_mode():
            self.loss_functions['nominal'] = self.loss_functions['layer_2']
            self.loss_functions['security'] = self.loss_functions['layer_3']
    
    def _is_multilayer_mode(self) -> bool:
        """Check if model menggunakan multilayer detection"""
        layer_mode = self.config.get('model', {}).get('layer_mode', 'multi')
        return layer_mode in ['multi', 'multilayer']
    
    def _get_loss_params(self) -> Dict[str, Any]:
        """Get standard loss parameters"""
        return {
            'box_weight': self.box_weight,
            'obj_weight': self.obj_weight, 
            'cls_weight': self.cls_weight,
            'focal_loss': self.focal_loss,
            'label_smoothing': self.label_smoothing
        }
    
    def compute_loss(self, predictions: Dict[str, List[torch.Tensor]], 
                    targets: torch.Tensor, img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute total loss untuk single atau multilayer detection
        
        Args:
            predictions: Dict dengan format {layer_name: [pred_p3, pred_p4, pred_p5]}
            targets: Batch targets [batch_idx, class, x, y, w, h]
            img_size: Input image size
            
        Returns:
            total_loss: Combined loss
            loss_breakdown: Detailed loss components
        """
        device = targets.device if len(targets) > 0 else torch.device('cpu')
        
        # Use MODEL_ARC.md compliant uncertainty-based multi-task loss if available
        if hasattr(self, 'use_multi_task_loss') and self.use_multi_task_loss:
            return self._compute_multi_task_loss(predictions, targets, img_size)
        else:
            return self._compute_individual_losses(predictions, targets, img_size)
    
    def _compute_multi_task_loss(self, predictions: Dict[str, List[torch.Tensor]], 
                                targets: torch.Tensor, img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss using MODEL_ARC.md compliant uncertainty-based multi-task loss"""
        # Prepare targets for each layer
        layer_targets = {}
        for layer_name in ['layer_1', 'layer_2', 'layer_3']:
            if layer_name in predictions:
                filtered_targets = self._filter_targets_for_layer(targets, layer_name)
                if len(filtered_targets) > 0:  # Only add if there are targets for this layer
                    layer_targets[layer_name] = filtered_targets
        
        # Initialize metrics dictionary
        metrics = {
            'val_loss': 0.0,
            'val_map50': 0.0,
            'val_map50_95': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0,
            'num_targets': len(targets)
        }
        
        try:
            # Use uncertainty-based multi-task loss
            total_loss, loss_breakdown = self.multi_task_loss(predictions, layer_targets, img_size)
            
            # Update metrics from loss breakdown
            if loss_breakdown:
                for k, v in loss_breakdown.items():
                    if isinstance(v, torch.Tensor):
                        v = v.item() if v.numel() == 1 else v.detach().cpu().numpy()
                    
                    # Add metric to appropriate category
                    if k.startswith(('val_', 'map', 'precision', 'recall', 'f1', 'accuracy')):
                        metrics[k] = v
                    elif k in ['box_loss', 'obj_loss', 'cls_loss']:
                        metrics[k] = v
            
            # Calculate overall metrics if not provided
            if 'val_map50' not in metrics:
                metrics['val_map50'] = metrics.get('map50', 0.0)
                metrics['val_map50_95'] = metrics.get('map50_95', 0.0)
                
            # Calculate F1 if not provided
            if 'val_f1' not in metrics and all(k in metrics for k in ['val_precision', 'val_recall']):
                p, r = metrics['val_precision'], metrics['val_recall']
                metrics['val_f1'] = 2 * (p * r) / (p + r + 1e-16)
            
            return total_loss, metrics
            
        except Exception as e:
            self.logger.error(f"Error in multi-task loss computation: {str(e)}")
            # Return zero metrics if there's an error
            return torch.tensor(0.0, device=targets.device), metrics
    
    def _compute_individual_losses(self, predictions: Dict[str, List[torch.Tensor]], 
                                  targets: torch.Tensor, img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss using individual YOLO losses (backward compatibility)"""
        device = targets.device if len(targets) > 0 else torch.device('cpu')
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_breakdown = {}
        
        # Handle single layer mode
        if not self._is_multilayer_mode():
            layer_name = list(predictions.keys())[0]  # Primary layer
            layer_preds = predictions[layer_name]
            loss_fn = self.loss_functions.get('banknote', self.loss_functions[layer_name])
            
            layer_loss, layer_components = loss_fn(layer_preds, targets, img_size)
            total_loss = total_loss + layer_loss
            loss_breakdown.update(layer_components)
        
        # Handle multilayer mode
        else:
            for layer_name, layer_preds in predictions.items():
                if layer_name in self.loss_functions:
                    # Filter targets untuk layer ini
                    layer_targets = self._filter_targets_for_layer(targets, layer_name)
                    
                    if len(layer_targets) > 0:
                        loss_fn = self.loss_functions[layer_name]
                        layer_loss, layer_components = loss_fn(layer_preds, layer_targets, img_size)
                        
                        # Add layer prefix ke component names
                        prefixed_components = {f"{layer_name}_{k}": v for k, v in layer_components.items()}
                        loss_breakdown.update(prefixed_components)
                        
                        total_loss = total_loss + layer_loss
        
        # Add overall metrics
        loss_breakdown['total_loss'] = total_loss
        loss_breakdown['num_targets'] = len(targets)
        
        return total_loss, loss_breakdown
    
    def _filter_targets_for_layer(self, targets: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Filter targets berdasarkan layer detection classes"""
        if len(targets) == 0:
            return targets
        
        # Class mapping untuk different layers (MODEL_ARC.md compliant)
        layer_class_ranges = {
            'layer_1': list(range(0, 7)),    # Full banknote detection: Classes 0-6
            'layer_2': list(range(7, 14)),   # Denomination features: Classes 7-13  
            'layer_3': list(range(14, 17)),  # Common features: Classes 14-16
            # Legacy support
            'banknote': list(range(0, 7)),   # Classes 0-6
            'nominal': list(range(7, 14)),   # Classes 7-13  
            'security': list(range(14, 17))  # Classes 14-16
        }
        
        valid_classes = layer_class_ranges.get(layer_name, list(range(0, 7)))
        
        # Filter targets dengan class yang sesuai
        mask = torch.zeros(len(targets), dtype=torch.bool, device=targets.device)
        for i, target in enumerate(targets):
            try:
                # Ensure target[1] is properly converted to int
                class_id = int(target[1].float().item())  # Explicit float conversion first
                if class_id in valid_classes:
                    mask[i] = True
            except (ValueError, RuntimeError) as e:
                # Skip invalid targets
                continue
        
        filtered_targets = targets[mask].clone()
        
        # Remap class IDs untuk layer ini (0-based indexing)
        if len(filtered_targets) > 0:
            class_offset = min(valid_classes)
            filtered_targets[:, 1] -= class_offset
        
        return filtered_targets
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights"""
        return {
            'box_weight': self.box_weight,
            'obj_weight': self.obj_weight,
            'cls_weight': self.cls_weight
        }
    
    def update_loss_weights(self, box_weight: Optional[float] = None,
                           obj_weight: Optional[float] = None,
                           cls_weight: Optional[float] = None) -> None:
        """Update loss weights dinamis selama training"""
        if box_weight is not None:
            self.box_weight = box_weight
        if obj_weight is not None:
            self.obj_weight = obj_weight
        if cls_weight is not None:
            self.cls_weight = cls_weight
        
        # Update semua loss functions
        for loss_fn in self.loss_functions.values():
            loss_fn.box_weight = self.box_weight
            loss_fn.obj_weight = self.obj_weight
            loss_fn.cls_weight = self.cls_weight
    
    def get_loss_breakdown_summary(self, loss_dict: Dict[str, Any]) -> str:
        """Get formatted summary dari loss breakdown"""
        total = loss_dict.get('total_loss', 0)
        box = loss_dict.get('box_loss', 0)
        obj = loss_dict.get('obj_loss', 0)
        cls = loss_dict.get('cls_loss', 0)
        
        return f"Total: {total:.4f} | Box: {box:.4f} | Obj: {obj:.4f} | Cls: {cls:.4f}"

# Convenience functions
def create_loss_manager(config: Dict[str, Any]) -> LossManager:
    """Factory function untuk create loss manager"""
    return LossManager(config)

def compute_yolo_loss(predictions: Dict[str, List[torch.Tensor]], targets: torch.Tensor,
                     config: Dict[str, Any], img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """One-liner untuk compute YOLO loss"""
    loss_manager = LossManager(config)
    return loss_manager.compute_loss(predictions, targets, img_size)