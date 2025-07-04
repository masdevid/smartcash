"""
File: smartcash/model/training/loss_manager.py
Deskripsi: Manager untuk perhitungan loss YOLO dengan dukungan multi-layer currency detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Any

class YOLOLoss(nn.Module):
    """YOLO loss implementation untuk currency detection"""
    
    def __init__(self, num_classes: int = 7, anchors: Optional[List] = None, 
                 box_weight: float = 0.05, obj_weight: float = 1.0, cls_weight: float = 0.5,
                 focal_loss: bool = False, label_smoothing: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.focal_loss = focal_loss
        self.label_smoothing = label_smoothing
        
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
        
        # Calculate loss untuk setiap scale
        for i, pred in enumerate(predictions):
            b, a, gj, gi = indices[i]  # batch, anchor, grid_y, grid_x
            tobj = torch.zeros_like(pred[..., 0])  # target objectness
            
            if len(b) > 0:
                # Predictions subset
                ps = pred[b, a, gj, gi]
                
                # Regression (box) loss
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = self._bbox_iou(pbox.T, tbox[i], x1y1x2y2=False)
                lbox += (1.0 - iou).mean()
                
                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)
                
                # Classification
                if self.num_classes > 1:
                    t = torch.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(len(tcls[i])), tcls[i]] = self.cp
                    lcls += self._classification_loss(ps[:, 5:], t)
            
            # Objectness loss
            obji = self.bce_obj(pred[..., 4], tobj)
            lobj += obji * self.balance[i]
        
        # Scale losses
        lbox *= self.box_weight
        lobj *= self.obj_weight
        lcls *= self.cls_weight
        
        total_loss = lbox + lobj + lcls
        
        return total_loss, {
            'box_loss': lbox,
            'obj_loss': lobj,
            'cls_loss': lcls,
            'total_loss': total_loss
        }
    
    def _build_targets(self, predictions: List[torch.Tensor], targets: torch.Tensor, 
                      img_size: int) -> Tuple[List, List, List, List]:
        """Build targets untuk setiap prediction scale"""
        na, nt = len(self.anchors[0]), targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        
        # Gain untuk convert ke grid coordinates
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
        
        g = 0.5  # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=targets.device).float() * g
        
        for i in range(len(predictions)):
            anchors = self.anchors[i].to(targets.device)
            gain[2:6] = torch.tensor(predictions[i].shape)[[3, 2, 3, 2]]
            
            # Match targets ke anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]
                j = torch.max(r, 1.0 / r).max(2)[0] < 4.0  # anchor ratio threshold
                t = t[j]
                
                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0
            
            # Define
            b, c = t[:, :2].long().T  # batch, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices
            
            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class
        
        return tcls, tbox, indices, anch
    
    def _bbox_iou(self, box1: torch.Tensor, box2: torch.Tensor, 
                  x1y1x2y2: bool = True, eps: float = 1e-7) -> torch.Tensor:
        """Calculate IoU between boxes"""
        if x1y1x2y2:
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        else:
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
        
        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps
        
        return inter / union
    
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
        # Main currency detection (7 classes)
        self.loss_functions['banknote'] = YOLOLoss(
            num_classes=7,
            box_weight=self.box_weight,
            obj_weight=self.obj_weight,
            cls_weight=self.cls_weight,
            focal_loss=self.focal_loss,
            label_smoothing=self.label_smoothing
        )
        
        # Additional layers jika diperlukan
        if self._is_multilayer_mode():
            self.loss_functions['nominal'] = YOLOLoss(num_classes=7, **self._get_loss_params())
            self.loss_functions['security'] = YOLOLoss(num_classes=3, **self._get_loss_params())
    
    def _is_multilayer_mode(self) -> bool:
        """Check if model menggunakan multilayer detection"""
        return self.config.get('model', {}).get('layer_mode') == 'multilayer'
    
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
        total_loss = torch.tensor(0.0, device=targets.device, requires_grad=True)
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
        
        # Class mapping untuk different layers
        layer_class_ranges = {
            'banknote': list(range(0, 7)),    # Classes 0-6
            'nominal': list(range(7, 14)),    # Classes 7-13  
            'security': list(range(14, 17))   # Classes 14-16
        }
        
        valid_classes = layer_class_ranges.get(layer_name, list(range(0, 7)))
        
        # Filter targets dengan class yang sesuai
        mask = torch.zeros(len(targets), dtype=torch.bool, device=targets.device)
        for i, target in enumerate(targets):
            class_id = int(target[1].item())
            if class_id in valid_classes:
                mask[i] = True
        
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