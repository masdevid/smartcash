# File: src/models/train.py
# Author: Alfrida Sabar
# Deskripsi: Training loop dan loss function untuk SmartCash Detector dengan optimasi performa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from tqdm import tqdm
from utils.logging import ColoredLogger

class SmartCashLoss(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.nc = num_classes
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.ce = nn.CrossEntropyLoss()
        
        # Loss weights
        self.lambda_box = 0.05  
        self.lambda_obj = 1.0
        self.lambda_cls = 0.5
        self.lambda_small = 2.0  # Small object emphasis
        
    def forward(self, preds, targets):
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        
        # Process each prediction level
        for pi, pred in enumerate(preds):
            b, a, gj, gi = self._build_targets(pred, targets, pi)
            tobj = torch.zeros_like(pred[..., 0], device=device)
            
            n = b.shape[0]
            if n:
                # Extract predictions
                ps = pred[b, a, gj, gi]
                pbox = self._decode_boxes(ps[..., :4])
                
                # Box loss (IoU based)
                iou = self._box_iou(pbox, targets[..., :4])
                lbox += (1.0 - iou).mean()
                
                # Small object weighting
                box_area = targets[..., 2] * targets[..., 3]
                small_weights = torch.exp(-box_area * self.lambda_small)
                lbox *= small_weights.mean()
                
                # Objectness loss
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                lobj += self.bce(pred[..., 4], tobj)
                
                # Classification loss
                t = torch.full((n,), self.nc, device=device)
                t[:n] = targets[..., 5].long()
                lcls += self.ce(ps[:, 5:], t)
        
        # Weight losses
        loss = self.lambda_box * lbox + self.lambda_obj * lobj + self.lambda_cls * lcls
        return loss, {'box': lbox.item(), 'obj': lobj.item(), 'cls': lcls.item()}

    def _decode_boxes(self, boxes):
        """Convert network outputs to bounding boxes"""
        boxes[..., 0:2] = torch.sigmoid(boxes[..., 0:2])  # center xy
        boxes[..., 2:4] = torch.exp(boxes[..., 2:4])  # wh
        return boxes
        
    def _build_targets(self, pred, targets, level):
        """Match targets to anchors"""
        na = pred.shape[1]  # number of anchors
        nt = targets.shape[0]  # number of targets
        bs = pred.shape[0]  # batch size

        gain = torch.ones(7, device=pred.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=pred.device).float().view(na, 1).repeat(1, nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = torch.tensor([[0, 0],
                          [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                          # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                          ], device=pred.device).float() * g

        return bs, na, gain, off

    def _box_iou(self, box1, box2):
        """Calculate IoU between boxes"""
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1 = box1[:, 0] - box1[:, 2] / 2, box1[:, 1] - box1[:, 3] / 2
        b1_x2, b1_y2 = box1[:, 0] + box1[:, 2] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_y1 = box2[:, 0] - box2[:, 2] / 2, box2[:, 1] - box2[:, 3] / 2
        b2_x2, b2_y2 = box2[:, 0] + box2[:, 2] / 2, box2[:, 1] + box2[:, 3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
        union = (w1 * h1 + 1e-16) + w2 * h2 - inter

        iou = inter / union
        return iou

class Trainer:
    def __init__(self, model, train_loader, device='cuda'):
        self.logger = ColoredLogger('Trainer')
        self.model = model
        self.loader = train_loader
        self.device = device
        self.criterion = SmartCashLoss()
        self.scaler = amp.GradScaler()
        
    def train_epoch(self, optimizer):
        self.model.train()
        epoch_loss = {'box': 0, 'obj': 0, 'cls': 0}
        pbar = tqdm(self.loader, desc='Training')
        
        for batch_idx, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(self.device)
            targets = targets.to(self.device)
            
            # Mixed precision training
            with amp.autocast():
                preds = self.model(imgs)
                loss, loss_items = self.criterion(preds, targets)
            
            # Backward pass
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Update metrics
            for k, v in loss_items.items():
                epoch_loss[k] += v
                
            # Update progress bar
            pbar.set_postfix({k: f'{v:.3f}' for k, v in loss_items.items()})
        
        return {k: v / len(self.loader) for k, v in epoch_loss.items()}

    def train(self, epochs, val_loader=None, patience=5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.OneCycleLR(
            optimizer, max_lr=0.01, epochs=epochs, 
            steps_per_epoch=len(self.loader)
        )
        
        best_loss = float('inf')
        wait = 0
        
        for epoch in range(epochs):
            self.logger.info(f'🏃 Epoch {epoch+1}/{epochs}')
            
            # Training
            train_loss = self.train_epoch(optimizer)
            self.logger.info('📈 Train Loss:', metrics=train_loss)
            
            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.logger.info('📊 Val Loss:', metrics=val_loss)
                
                # Early stopping
                if val_loss['box'] < best_loss:
                    best_loss = val_loss['box']
                    wait = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                    }, 'best.pt')
                    self.logger.info('💾 Model saved!')
                else:
                    wait += 1
                    if wait >= patience:
                        self.logger.info('🛑 Early stopping triggered!')
                        break
            
            scheduler.step()
            
    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = {'box': 0, 'obj': 0, 'cls': 0}
        
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc='Validating'):
                imgs = imgs.to(self.device)
                targets = targets.to(self.device)
                
                with amp.autocast():
                    preds = self.model(imgs)
                    _, loss_items = self.criterion(preds, targets)
                
                for k, v in loss_items.items():
                    val_loss[k] += v
        
        return {k: v / len(val_loader) for k, v in val_loss.items()}