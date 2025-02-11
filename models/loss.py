# File: src/models/loss.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi fungsi loss untuk YOLO dengan fokus deteksi objek kecil

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOLoss(nn.Module):
    def __init__(self, num_classes=7, anchors=None):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors or [
            [[10,13], [16,30], [33,23]],
            [[30,61], [62,45], [59,119]],
            [[116,90], [156,198], [373,326]]
        ]
        self.nl = len(anchors)
        self.na = len(anchors[0])
        self.no = num_classes + 5  # outputs per anchor
        
        # Loss weights
        self.lambda_box = 0.05
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
        self.lambda_small = 2.0  # weight for small objects
        
    def forward(self, outputs, targets):
        total_loss = torch.tensor(0., device=outputs[0].device)
        lcls = torch.zeros(1, device=outputs[0].device)
        lbox = torch.zeros(1, device=outputs[0].device)
        lobj = torch.zeros(1, device=outputs[0].device)
        
        # Process each detection layer
        for i in range(self.nl):
            pred = outputs[i]
            batch_size, _, grid_h, grid_w = pred.shape
            
            # Extract predictions
            pred = pred.view(batch_size, self.na, self.no, grid_h, grid_w)
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]
            pred_box = pred[..., :4]
            
            # Match targets to anchors
            targets_i = self._build_targets(targets, pred, i)
            
            if targets_i.shape[0] > 0:
                # Box loss
                box_loss = self._compute_box_loss(pred_box, targets_i)
                box_loss *= self.lambda_box
                
                # Small object weighting
                box_area = targets_i[:, 2] * targets_i[:, 3]
                small_obj_weight = torch.exp(-box_area * self.lambda_small)
                box_loss *= small_obj_weight.mean()
                
                lbox += box_loss
                
                # Classification loss
                cls_loss = self._compute_cls_loss(pred_cls, targets_i)
                lcls += cls_loss * self.lambda_cls
                
                # Objectness loss
                obj_loss = self._compute_obj_loss(pred_obj, targets_i)
                lobj += obj_loss * self.lambda_obj
        
        total_loss = lbox + lobj + lcls
        loss_dict = {
            'box_loss': lbox.item(),
            'obj_loss': lobj.item(),
            'cls_loss': lcls.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def _build_targets(self, targets, pred, layer_idx):
        """Convert targets to matched anchor format"""
        anchors = torch.tensor(self.anchors[layer_idx])
        targets_out = []
        
        if targets.shape[0] == 0:
            return torch.zeros((0, 6))
            
        # Calculate anchor IoU for matching
        box_wh = targets[:, 2:4]
        anchor_wh = anchors.unsqueeze(0)
        intersect = torch.min(box_wh, anchor_wh).prod(1)
        union = box_wh.prod(1).unsqueeze(1) + anchor_wh.prod(1) - intersect
        iou = intersect / union
        
        # Match each target to best anchor
        best_anchor = iou.argmax(1)
        
        for ti, target in enumerate(targets):
            if target.sum() == 0:
                continue
                
            # Add anchor index to target
            target_out = torch.zeros(6)
            target_out[:4] = target[:4]
            target_out[4] = best_anchor[ti]
            target_out[5] = target[4]  # class
            targets_out.append(target_out)
            
        return torch.stack(targets_out) if targets_out else torch.zeros((0, 6))
    
    def _compute_box_loss(self, pred, targets):
        if targets.shape[0] == 0:
            return torch.tensor(0)
            
        # Extract predictions corresponding to targets
        batch_idx = targets[:, 0].long()
        anchor_idx = targets[:, 4].long()
        grid_y = targets[:, 2].long()
        grid_x = targets[:, 1].long()
        
        pred_boxes = pred[batch_idx, anchor_idx, :, grid_y, grid_x]
        target_boxes = targets[:, 1:5]
        
        # CIoU loss
        return (1.0 - self._box_ciou(pred_boxes, target_boxes)).mean()
    
    def _compute_cls_loss(self, pred, targets):
        if targets.shape[0] == 0:
            return torch.tensor(0)
            
        batch_idx = targets[:, 0].long()
        anchor_idx = targets[:, 4].long()
        grid_y = targets[:, 2].long()
        grid_x = targets[:, 1].long()
        
        pred_cls = pred[batch_idx, anchor_idx, :, grid_y, grid_x]
        target_cls = F.one_hot(targets[:, 5].long(), self.num_classes)
        
        return F.binary_cross_entropy_with_logits(pred_cls, target_cls.float())
    
    def _compute_obj_loss(self, pred, targets):
        if targets.shape[0] == 0:
            return torch.tensor(0)
            
        batch_idx = targets[:, 0].long()
        anchor_idx = targets[:, 4].long()
        grid_y = targets[:, 2].long()
        grid_x = targets[:, 1].long()
        
        obj_targets = torch.zeros_like(pred)
        obj_targets[batch_idx, anchor_idx, grid_y, grid_x] = 1
        
        return F.binary_cross_entropy_with_logits(pred, obj_targets)
    
    def _box_ciou(self, box1, box2):
        """Complete IoU loss"""
        b1_xy = box1[:, :2]
        b1_wh = box1[:, 2:4]
        b2_xy = box2[:, :2] 
        b2_wh = box2[:, 2:4]
        
        # Intersection area
        intersect_xy = torch.max(b1_xy - b1_wh/2, b2_xy - b2_wh/2)
        intersect_wh = torch.min(b1_xy + b1_wh/2, b2_xy + b2_wh/2) - intersect_xy
        intersect = torch.prod(torch.relu(intersect_wh), 1)
        
        # Union area
        b1_area = torch.prod(b1_wh, 1)
        b2_area = torch.prod(b2_wh, 1)
        union = b1_area + b2_area - intersect
        
        # IoU
        iou = intersect / union
        
        # Diagonal distance
        c_wh = torch.max(b1_xy + b1_wh/2, b2_xy + b2_wh/2) - torch.min(b1_xy - b1_wh/2, b2_xy - b2_wh/2)
        c2 = torch.sum(c_wh**2, 1)
        
        # Center distance
        rho2 = torch.sum((b1_xy - b2_xy)**2, 1)
        
        # Aspect ratio
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(b1_wh[:, 0] / b1_wh[:, 1]) - torch.atan(b2_wh[:, 0] / b2_wh[:, 1]), 2
        )
        alpha = v / (1 - iou + v)
        
        return iou - (rho2 / c2 + v * alpha)