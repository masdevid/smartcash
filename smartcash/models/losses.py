# File: models/losses.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi fungsi loss untuk YOLOv5

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

class YOLOLoss(nn.Module):
    """YOLOv5 Loss Function."""
    
    def __init__(
        self,
        num_classes: int = 7,
        anchors: List[List[int]] = None,
        anchor_t: float = 4.0,
        balance: List[float] = [4.0, 1.0, 0.4],
        box_weight: float = 0.05,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        # Default anchors for YOLOv5s
        if anchors is None:
            anchors = [
                [[10, 13], [16, 30], [33, 23]],  # P3/8
                [[30, 61], [62, 45], [59, 119]],  # P4/16
                [[116, 90], [156, 198], [373, 326]]  # P5/32
            ]
        
        self.num_classes = num_classes
        self.balance = balance
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.label_smoothing = label_smoothing
        
        # Convert anchors to tensor
        self.register_buffer('anchors', torch.tensor(anchors).float().view(len(anchors), -1, 2))
        self.register_buffer('anchor_t', torch.tensor(anchor_t))
        
        self.na = len(anchors[0])  # number of anchors
        self.nl = len(anchors)  # number of layers
        
        # Initialize BCEWithLogitsLoss for classification and objectness
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]))
        
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions: List of predictions from each scale [P3, P4, P5]
            targets: [num_targets, 6] (image_idx, class_idx, x, y, w, h)
        
        Returns:
            total_loss: Combined loss
            loss_items: Dictionary containing individual loss components
        """
        device = targets.device
        lcls = torch.zeros(1, device=device)  # class loss
        lbox = torch.zeros(1, device=device)  # box loss
        lobj = torch.zeros(1, device=device)  # object loss
        
        # Calculate losses for each scale
        for i, pred in enumerate(predictions):
            # Get targets for this scale
            batch_size, _, grid_h, grid_w = pred.shape
            
            # Transform predictions
            pred = pred.view(batch_size, self.na, 5 + self.num_classes, grid_h, grid_w)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()
            
            # Calculate losses
            t = self._build_targets(pred, targets, i)
            
            # Box loss
            if len(t['box_target']) > 0:
                pbox = pred[t['batch_idx'], t['anchor_idx'], t['grid_y'], t['grid_x']][:, :4]
                iou = self._box_iou(pbox, t['box_target'])
                lbox += (1.0 - iou).mean()
            
            # Class loss
            if len(t['cls_target']) > 0:
                pcls = pred[t['batch_idx'], t['anchor_idx'], t['grid_y'], t['grid_x']][:, 5:]
                if self.label_smoothing:
                    tcls = torch.zeros_like(pcls)
                    tcls[range(len(t['cls_target'])), t['cls_target']] = 1.0 - self.label_smoothing
                    tcls += self.label_smoothing / self.num_classes
                else:
                    tcls = F.one_hot(t['cls_target'], self.num_classes)
                lcls += self.BCEcls(pcls, tcls.float())
            
            # Objectness loss
            tobj = torch.zeros_like(pred[..., 4])
            if len(t['batch_idx']) > 0:
                tobj[t['batch_idx'], t['anchor_idx'], t['grid_y'], t['grid_x']] = 1.0
            lobj += self.BCEobj(pred[..., 4], tobj) * self.balance[i]
        
        # Combine losses
        lbox *= self.box_weight
        lobj *= self.obj_weight
        lcls *= self.cls_weight
        
        loss = lbox + lobj + lcls
        
        return loss, {
            'box_loss': lbox.detach(),
            'obj_loss': lobj.detach(),
            'cls_loss': lcls.detach()
        }
    
    def _build_targets(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Build targets for one scale."""
        na, grid_h, grid_w = pred.shape[1:4]
        nt = len(targets)
        
        batch_idx, anchor_idx, grid_y, grid_x = [], [], [], []
        box_target, cls_target = [], []
        
        if nt:
            # Calculate anchor indices
            gain = torch.ones(7, device=pred.device)
            gain[2:6] = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=pred.device)
            
            t = targets * gain  # Scale to gridspace
            
            if na > 1:  # For multiple anchors
                anchor_vec = self.anchors[layer_idx]
                gwh = t[:, 4:6].clone()
                # Select best anchor
                wh_ratio = gwh[:, None] / anchor_vec[None]
                j = torch.max(torch.min(wh_ratio, 1/wh_ratio).min(2)[0], 1)[1]
            else:
                j = torch.zeros(nt, device=pred.device).long()
            
            # Append data
            b, c = t[:, 0].long(), t[:, 1].long()
            gxy = t[:, 2:4]
            gwh = t[:, 4:6]
            gij = gxy.long()
            
            # Append to lists
            batch_idx.append(b)
            anchor_idx.append(j)
            grid_y.append(gij[:, 1].clamp_(0, grid_h-1))
            grid_x.append(gij[:, 0].clamp_(0, grid_w-1))
            box_target.append(torch.cat((gxy - gij.float(), gwh), 1))
            cls_target.append(c)
        
        # Concat lists
        batch_idx = torch.cat(batch_idx) if len(batch_idx) else torch.tensor([], device=pred.device)
        anchor_idx = torch.cat(anchor_idx) if len(anchor_idx) else torch.tensor([], device=pred.device)
        grid_y = torch.cat(grid_y) if len(grid_y) else torch.tensor([], device=pred.device)
        grid_x = torch.cat(grid_x) if len(grid_x) else torch.tensor([], device=pred.device)
        box_target = torch.cat(box_target) if len(box_target) else torch.tensor([], device=pred.device)
        cls_target = torch.cat(cls_target) if len(cls_target) else torch.tensor([], device=pred.device)
        
        return {
            'batch_idx': batch_idx,
            'anchor_idx': anchor_idx,
            'grid_y': grid_y,
            'grid_x': grid_x,
            'box_target': box_target,
            'cls_target': cls_target
        }
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes."""
        # Convert box1 from xywh to xyxy
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        
        # Convert box2 from xywh to xyxy
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
        
        # Intersection area
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        return inter_area / (b1_area + b2_area - inter_area + 1e-16)