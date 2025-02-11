# File: src/utils/metrics.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi metrik evaluasi untuk object detection

import numpy as np
import torch
from collections import defaultdict
from utils.logging import ColoredLogger

class MeanAveragePrecision:
    def __init__(self, num_classes=7, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.reset()
        self.logger = ColoredLogger('mAP')
        
    def reset(self):
        self.predictions = defaultdict(list)
        self.targets = defaultdict(list)
        
    def update(self, preds, targets):
        for pred, target in zip(preds, targets):
            pred_cls = pred[:, 5].int()
            for cls in range(self.num_classes):
                cls_mask_pred = pred_cls == cls
                cls_preds = pred[cls_mask_pred]
                if len(cls_preds):
                    self.predictions[cls].append(cls_preds[:, :5])  # box + conf
                
                cls_mask_target = target[:, 5].int() == cls
                cls_targets = target[cls_mask_target]
                if len(cls_targets):
                    self.targets[cls].append(cls_targets[:, :4])  # only box
    
    def compute(self):
        aps = []
        self.logger.info("📊 Menghitung mAP...")
        
        for cls in range(self.num_classes):
            cls_preds = self.predictions[cls]
            cls_targets = self.targets[cls]
            
            if not cls_preds or not cls_targets:
                continue
                
            # Concatenate all predictions and targets
            cls_preds = torch.cat(cls_preds, dim=0)
            cls_targets = torch.cat(cls_targets, dim=0)
            
            # Sort predictions by confidence
            conf_sort = torch.argsort(cls_preds[:, 4], descending=True)
            cls_preds = cls_preds[conf_sort]
            
            # Calculate precision and recall
            tp = torch.zeros(len(cls_preds))
            fp = torch.zeros(len(cls_preds))
            num_targets = len(cls_targets)
            
            if num_targets > 0:
                detected = []
                for pred_idx, pred_box in enumerate(cls_preds[:, :4]):
                    best_iou = 0
                    best_target_idx = None
                    
                    for target_idx, target_box in enumerate(cls_targets):
                        if target_idx in detected:
                            continue
                            
                        iou = self._box_iou(pred_box, target_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_target_idx = target_idx
                    
                    if best_iou > self.iou_threshold:
                        tp[pred_idx] = 1
                        detected.append(best_target_idx)
                    else:
                        fp[pred_idx] = 1
            
            # Calculate precision and recall
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            recalls = tp_cumsum / (num_targets + 1e-6)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
            
            # Compute AP using 11-point interpolation
            ap = 0
            for t in torch.arange(0, 1.1, 0.1):
                if torch.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = torch.max(precisions[recalls >= t])
                ap = ap + p / 11.0
            
            aps.append(float(ap))
            self.logger.info(f"    Kelas {cls}: AP = {ap:.4f}")
        
        mAP = np.mean(aps)
        self.logger.info(f"    mAP: {mAP:.4f}")
        return mAP
        
    def _box_iou(self, box1, box2):
        """Calculate IoU between two boxes"""
        # Convert to x1y1x2y2 format
        b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
        b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
        b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
        b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
        
        # Intersection
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
                     
        # Union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area
        
        return inter_area / (union_area + 1e-6)