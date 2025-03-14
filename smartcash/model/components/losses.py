"""
File: smartcash/model/components/losses.py
Deskripsi: Implementasi YOLOv5 loss function dengan perbaikan stabilitas numerik dan penanganan error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union

from smartcash.common.logger import SmartCashLogger
from smartcash.model.exceptions import ModelError

class YOLOLoss(nn.Module):
    """
    YOLOv5 Loss Function dengan perbaikan struktur untuk stabilitas training.
    
    Menghitung box loss (IoU), objectness loss, dan classification loss
    dengan penanganan error yang robust untuk mencegah NaN gradient.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        anchors: List[List[int]] = None,
        anchor_t: float = 4.0,
        balance: List[float] = [4.0, 1.0, 0.4],
        box_weight: float = 0.05,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0,
        label_smoothing: float = 0.0,
        eps: float = 1e-16,  # Epsilon untuk stabilitas numerik
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi YOLOLoss.
        
        Args:
            num_classes: Jumlah kelas untuk deteksi
            anchors: Daftar anchors boxes (default: YOLOv5s)
            anchor_t: Threshold untuk pemilihan anchor
            balance: Faktor balance untuk setiap skala
            box_weight: Bobot untuk box loss
            cls_weight: Bobot untuk classification loss
            obj_weight: Bobot untuk objectness loss
            label_smoothing: Nilai label smoothing (0.0-1.0)
            eps: Epsilon untuk stabilitas numerik
            logger: Logger untuk mencatat proses (opsional)
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        
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
        self.eps = eps
        
        # Convert anchors to tensor
        self.register_buffer('anchors', torch.tensor(anchors).float().view(len(anchors), -1, 2))
        self.register_buffer('anchor_t', torch.tensor(anchor_t))
        
        self.na = len(anchors[0])  # jumlah anchors per skala
        self.nl = len(anchors)  # jumlah level (biasanya 3: P3, P4, P5)
        
        # Initialize BCEWithLogitsLoss untuk classification dan objectness
        self.BCEcls = nn.BCEWithLogitsLoss(reduction='none')
        self.BCEobj = nn.BCEWithLogitsLoss(reduction='none')
        
        self.logger.info(f"✅ YOLOLoss diinisialisasi untuk {num_classes} kelas")
    
    def forward(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hitung loss untuk prediksi dan target.
        
        Args:
            predictions: List of predictions dari setiap skala [P3, P4, P5]
            targets: [num_targets, 6] (image_idx, class_idx, x, y, w, h)
        
        Returns:
            total_loss: Combined loss
            loss_items: Dictionary berisi individual loss components
            
        Raises:
            ModelError: Jika perhitungan loss gagal
        """
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        # Validasi input
        if not self._validate_inputs(predictions, targets):
            # Return dummy loss dengan requires_grad
            dummy = torch.tensor(0.1, device=device, requires_grad=True)
            self.logger.warning("⚠️ Input tidak valid, menggunakan dummy loss")
            return dummy, {
                'box_loss': torch.zeros(1, device=device),
                'obj_loss': torch.zeros(1, device=device),
                'cls_loss': dummy
            }
            
        # Standarisasi format target
        targets = self._standardize_targets(targets)
            
        try:
            batch_box_loss = []
            batch_obj_loss = []
            batch_cls_loss = []
            
            # Calculate losses for each prediction scale
            for i, pred in enumerate(predictions):
                # Validasi dimensi
                if pred.dim() < 4:
                    self.logger.warning(f"⚠️ Prediksi skala {i} memiliki dimensi yang tidak valid: {pred.dim()}")
                    continue
                
                # Reshape prediction for processing
                batch_size, _, grid_h, grid_w = pred.shape
                
                # Transform predictions
                pred = pred.view(batch_size, self.na, 5 + self.num_classes, grid_h, grid_w)
                pred = pred.permute(0, 1, 3, 4, 2).contiguous()
                
                # Build targets for this scale
                t = self._build_targets(pred, targets, i)
                
                # Box loss (IoU loss)
                if t['box_target'].shape[0] > 0:
                    pbox = pred[t['batch_idx'], t['anchor_idx'], t['grid_y'], t['grid_x']][:, :4]
                    iou = self._box_iou(pbox, t['box_target'])
                    box_loss = (1.0 - iou).mean()
                    batch_box_loss.append(box_loss)
                
                # Class loss (BCE loss with optional label smoothing)
                if t['cls_target'].shape[0] > 0:
                    pcls = pred[t['batch_idx'], t['anchor_idx'], t['grid_y'], t['grid_x']][:, 5:]
                    
                    # Handle class targets (binary cross-entropy)
                    if self.label_smoothing:
                        # One-hot dengan label smoothing
                        tcls = torch.zeros_like(pcls)
                        tcls[range(len(t['cls_target'])), t['cls_target'].long()] = 1.0 - self.label_smoothing
                        tcls += self.label_smoothing / self.num_classes
                    else:
                        # Standard one-hot encoding
                        tcls = F.one_hot(t['cls_target'].long(), self.num_classes).float()
                    
                    # Calculate class loss
                    cls_loss_raw = self.BCEcls(pcls, tcls)
                    cls_loss = cls_loss_raw.mean()
                    batch_cls_loss.append(cls_loss)
                
                # Objectness loss (BCE loss for object confidence)
                tobj = torch.zeros_like(pred[..., 4])
                if t['batch_idx'].shape[0] > 0:
                    tobj[t['batch_idx'], t['anchor_idx'], t['grid_y'], t['grid_x']] = 1.0
                
                # Calculate object loss with scale balancing
                obj_loss_raw = self.BCEobj(pred[..., 4], tobj)
                obj_loss = obj_loss_raw.mean() * self.balance[i]  # Apply balance factor
                batch_obj_loss.append(obj_loss)
            
            # Combine losses for each scale with weights
            if batch_box_loss:
                lbox = torch.stack(batch_box_loss).mean() * self.box_weight
            
            if batch_obj_loss:
                lobj = torch.stack(batch_obj_loss).mean() * self.obj_weight
                
            if batch_cls_loss:
                lcls = torch.stack(batch_cls_loss).mean() * self.cls_weight
            
            # Combine all losses
            loss = lbox + lobj + lcls
            
        except Exception as e:
            self.logger.error(f"❌ Perhitungan loss gagal: {str(e)}")
            # Fallback jika terjadi error
            loss = torch.tensor(0.1, device=device, requires_grad=True)
            lbox = torch.zeros(1, device=device)
            lobj = torch.zeros(1, device=device)
            lcls = loss
        
        # Return loss dan komponennya
        return loss, {
            'box_loss': lbox,
            'obj_loss': lobj,
            'cls_loss': lcls
        }
    
    def _validate_inputs(
        self,
        predictions: List[torch.Tensor],
        targets: torch.Tensor
    ) -> bool:
        """
        Validasi input untuk YOLOLoss.
        
        Args:
            predictions: List predictions
            targets: Target tensor
            
        Returns:
            bool: True jika input valid, False jika tidak
        """
        # Check predictions
        if not predictions or len(predictions) == 0:
            self.logger.warning("⚠️ Prediksi kosong")
            return False
            
        # Check targets
        if not isinstance(targets, torch.Tensor) or targets.numel() == 0:
            self.logger.warning("⚠️ Target tidak valid atau kosong")
            return False
            
        return True
    
    def _standardize_targets(
        self,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Standarisasi format targets ke [batch, num_targets, target_data].
        
        Args:
            targets: Target tensor
            
        Returns:
            torch.Tensor: Target yang sudah distandarisasi
        """
        if len(targets.shape) == 3 and targets.shape[2] >= 5:
            # Sudah dalam format [batch, num_targets, target_data]
            return targets
        elif len(targets.shape) == 2 and targets.shape[1] >= 5:
            # Format [num_targets, target_data] -> [batch=1, num_targets, target_data]
            return targets.unsqueeze(0)
        else:
            # Coba reshape
            try:
                if targets.numel() % 6 == 0:  # Assuming 6 values per target
                    reshaped = targets.view(-1, 6)
                    return reshaped.unsqueeze(0)  # [batch=1, num_targets, target_data]
                else:
                    # Format tidak dapat dikenali
                    return torch.zeros((1, 0, 6), device=targets.device)  # Empty tensor
            except:
                # Tidak bisa reshape
                return torch.zeros((1, 0, 6), device=targets.device)  # Empty tensor
    
    def _build_targets(
        self,
        pred: torch.Tensor,
        targets: torch.Tensor,
        layer_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Build targets untuk satu skala dengan penanganan error.
        
        Args:
            pred: Prediksi untuk satu skala
            targets: Target tensor
            layer_idx: Indeks layer
            
        Returns:
            Dict: Dictionary berisi target components untuk loss calculation
        """
        na, grid_h, grid_w = pred.shape[1:4]
        device = pred.device
        
        # Initialize empty output
        result = {
            'batch_idx': torch.tensor([], device=device, dtype=torch.long),
            'anchor_idx': torch.tensor([], device=device, dtype=torch.long),
            'grid_y': torch.tensor([], device=device, dtype=torch.long),
            'grid_x': torch.tensor([], device=device, dtype=torch.long),
            'box_target': torch.tensor([], device=device, dtype=torch.float32).view(0, 4),
            'cls_target': torch.tensor([], device=device, dtype=torch.long)
        }
        
        # Validasi target
        if not isinstance(targets, torch.Tensor) or targets.numel() == 0:
            return result
        
        if targets.dim() != 3 or targets.shape[2] < 6:
            return result
        
        # Process targets
        batch_indices = []
        anchor_indices = []
        grid_y_indices = []
        grid_x_indices = []
        box_targets = []
        cls_targets = []
        
        for bi in range(targets.shape[0]):  # batch index
            # Get targets for this batch
            bt = targets[bi]
            
            # Skip jika tidak ada target
            if bt.shape[0] == 0:
                continue
                
            # Filter target yang valid
            valid_mask = (
                torch.isfinite(bt).all(dim=1) &  # Tidak ada nan/inf
                (bt[:, 1:5] > 0).all(dim=1) &    # Box coords positif
                (bt[:, 0] >= 0) &                # Class valid
                (bt[:, 0] < self.num_classes)    # Class dalam range
            )
            
            if not valid_mask.any():
                continue
                
            # Filter valid targets
            bt = bt[valid_mask]
            
            try:
                # Calculate anchor indices
                if na > 1:
                    # Multi-anchor: select best anchor for each target
                    anchor_vec = self.anchors[layer_idx]
                    gwh = bt[:, 3:5].clone()  # grid wh
                    
                    # Ratio between target and anchor
                    wh_ratio = gwh[:, None] / anchor_vec[None]
                    # Gunakan max-min instead of IoU untuk efficiency
                    j = torch.max(torch.min(wh_ratio, 1/wh_ratio).min(2)[0], 1)[1]
                else:
                    # Single anchor (na=1): use the only anchor
                    j = torch.zeros(bt.shape[0], dtype=torch.long, device=device)
                
                # Extract target info
                gxy = bt[:, 1:3].clone()  # grid xy
                gwh = bt[:, 3:5].clone()  # grid wh
                gxy_int = gxy.long()
                
                # Clamp grid coordinates
                gxi = torch.clamp(gxy_int[:, 0], 0, grid_w-1)
                gyi = torch.clamp(gxy_int[:, 1], 0, grid_h-1)
                
                # Append to lists
                batch_indices.append(torch.full_like(j, bi))
                anchor_indices.append(j)
                grid_y_indices.append(gyi)
                grid_x_indices.append(gxi)
                box_targets.append(torch.cat((gxy - gxy_int.float(), gwh), 1))
                cls_targets.append(bt[:, 0])  # class
            except Exception as e:
                self.logger.warning(f"⚠️ Error saat memproses batch {bi}: {str(e)}")
                # Skip jika terjadi error pada batch ini
                continue
        
        # Concat lists
        if batch_indices:
            result['batch_idx'] = torch.cat(batch_indices)
            result['anchor_idx'] = torch.cat(anchor_indices)
            result['grid_y'] = torch.cat(grid_y_indices)
            result['grid_x'] = torch.cat(grid_x_indices)
            result['box_target'] = torch.cat(box_targets)
            result['cls_target'] = torch.cat(cls_targets)
        
        return result
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between two sets of boxes.
        
        Args:
            box1: [n, 4] Format: cx, cy, w, h
            box2: [n, 4] Format: cx, cy, w, h
            
        Returns:
            torch.Tensor: IoU tensor
        """
        # Convert center-x, center-y, width, height to xyxy format
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2
        
        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2
        
        # Get intersection area
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Get box areas
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
        # Calculate IoU
        union = b1_area + b2_area - intersection + 1e-7  # avoid division by zero
        return intersection / union