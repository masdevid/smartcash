"""
file_path: /Users/masdevid/Projects/smartcash/smartcash/model/training/metrics/map_calculator.py

Mean Average Precision (mAP) calculator for object detection.

Implements both fine-grained (17 classes) and merged (8 main classes) mAP calculations
as specified in loss.json.

Key specifications from loss.json:
- training_mAP: Computed separately for each fine class 0-16, then averaged
- merged_mAP: Predictions and ground truth mapped to main_id using class_mapping.full_map
  - 8 main classes (7 denominations + 1 validation type)
  - IoU threshold = 0.5
  - AP computed per main_id, then averaged to get mAP@0.5
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from collections import defaultdict

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class MAPCalculator:
    """
    Calculates mean Average Precision (mAP) for object detection.
    
    Implements both fine-grained (17 classes) and merged (8 main classes) mAP calculations
    as specified in loss.json.
    """
    
    # Default class mapping from fine-grained (0-16) to main classes (0-7)
    # This will be overridden by the mapping from loss.json if available
    DEFAULT_FINE_TO_MAIN = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6,  # Main denominations
        7: 0, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 13: 6,  # Nominal features
        14: 7, 15: 7, 16: 7  # Validation features (all mapped to class 7)
    }
    
    # Class names for logging
    FINE_CLASS_NAMES = [
        '1000_whole', '2000_whole', '5000_whole', '10000_whole', 
        '20000_whole', '50000_whole', '100000_whole',
        '1000_nominal', '2000_nominal', '5000_nominal', '10000_nominal',
        '20000_nominal', '50000_nominal', '100000_nominal',
        'special_sign', 'micro_text', 'security_thread'
    ]
    
    MAIN_CLASS_NAMES = [
        '1000', '2000', '5000', '10000', '20000', '50000', '100000',
        'validation_feature'
    ]
    
    # Number of main classes (7 denominations + 1 validation type)
    NUM_MAIN_CLASSES = 8
    
    def __init__(self, iou_threshold: float = 0.5, num_fine_classes: int = 17,
                 config_path: Optional[str] = None):
        """
        Initialize the mAP calculator.
        
        Args:
            iou_threshold: IoU threshold for considering a detection as TP (default: 0.5)
            num_fine_classes: Number of fine-grained classes (default: 17)
            config_path: Path to loss.json for loading class mapping
        """
        self.iou_threshold = iou_threshold
        self.num_fine_classes = num_fine_classes
        
        # Load class mapping from config or use default
        self.fine_to_main = self._load_class_mapping(config_path)
        
        # Storage for all detections and ground truths
        self.all_detections = []
        self.all_gts = []
    
    def _load_class_mapping(self, config_path: Optional[str] = None) -> Dict[int, int]:
        """
        Load class mapping from loss.json or use default.
        
        Args:
            config_path: Path to loss.json
            
        Returns:
            Dictionary mapping fine class IDs to main class IDs
        """
        if not config_path or not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}, using default class mapping")
            return self.DEFAULT_FINE_TO_MAIN
            
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get mapping from config or use default
            if 'class_mapping' in config and 'mapping_to_main' in config['class_mapping']:
                mapping = config['class_mapping']['mapping_to_main']
                # Convert string keys to integers
                return {int(k): (v if isinstance(v, int) else 7) for k, v in mapping.items()}
            
            logger.warning("No valid class mapping found in config, using default")
            return self.DEFAULT_FINE_TO_MAIN
            
        except Exception as e:
            logger.error(f"Error loading class mapping from {config_path}: {e}")
            return self.DEFAULT_FINE_TO_MAIN
        
    def reset(self):
        """Reset the calculator to start a new evaluation."""
        self.all_detections = []
        self.all_gts = []
    
    def update(self, 
              detections: List[torch.Tensor], 
              targets: torch.Tensor,
              image_shapes: Optional[List[Tuple[int, int]]] = None):
        """
        Update the calculator with new detections and ground truths.
        
        Args:
            detections: List of detection tensors, each of shape [num_detections, 6] 
                       (x1, y1, x2, y2, confidence, class_id)
            targets: Ground truth tensor of shape [num_objects, 6] 
                     (image_idx, class_id, x, y, w, h) - YOLO format
            image_shapes: List of (height, width) for each image in the batch
        """
        if image_shapes is None:
            # Default to a large image size if not provided
            image_shapes = [(1024, 1024)] * len(detections)
        
        # Process each image in the batch
        for img_idx, (img_dets, img_shape) in enumerate(zip(detections, image_shapes)):
            # Convert detections from tensor to numpy if needed
            if torch.is_tensor(img_dets):
                img_dets = img_dets.detach().cpu().numpy()
            
            # Filter out detections with low confidence
            if len(img_dets) > 0:
                # Ensure detections are in the correct format [x1, y1, x2, y2, conf, class_id]
                if img_dets.shape[1] > 6:  # If class probabilities are included
                    # Get class with highest score
                    class_scores = img_dets[:, 5:5+self.num_fine_classes]
                    class_ids = np.argmax(class_scores, axis=1)
                    confidences = np.max(class_scores, axis=1)
                    boxes = img_dets[:, :4]
                    img_dets = np.column_stack([boxes, confidences, class_ids])
                
                # Sort detections by confidence (descending)
                img_dets = img_dets[img_dets[:, 4].argsort()[::-1]]
            
            # Get ground truths for this image
            img_gts = targets[targets[:, 0] == img_idx] if len(targets) > 0 else []
            
            # Convert ground truths to numpy if needed
            if torch.is_tensor(img_gts):
                img_gts = img_gts.detach().cpu().numpy()
            
            # Convert YOLO format (cx, cy, w, h) to (x1, y1, x2, y2)
            gt_boxes = []
            for gt in img_gts:
                _, class_id, cx, cy, w, h = gt
                x1 = (cx - w/2) * img_shape[1]  # Scale to image width
                y1 = (cy - h/2) * img_shape[0]  # Scale to image height
                x2 = (cx + w/2) * img_shape[1]
                y2 = (cy + h/2) * img_shape[0]
                gt_boxes.append([x1, y1, x2, y2, class_id])
            
            self.all_detections.append(img_dets)
            self.all_gts.append(gt_boxes)
    
    def compute_ap(self, precisions: np.ndarray, recalls: np.ndarray) -> float:
        """
        Compute the average precision, given the precision-recall curve.
        
        Args:
            precisions: Precision values
            recalls: Recall values
            
        Returns:
            Average precision
        """
        # Append sentinel values to ensure the curve starts at (0, 1) and ends at (1, 0)
        precisions = np.concatenate(([0.], precisions, [0.]))
        recalls = np.concatenate(([0.], recalls, [1.]))
        
        # Smooth the precision-recall curve
        for i in range(len(precisions) - 1, 0, -1):
            precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
        
        # Find the indices where recall changes
        i = np.where(recalls[1:] != recalls[:-1])[0]
        
        # Compute the area under the precision-recall curve
        ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        return ap
    
    def compute_map(self, merge_classes: bool = False) -> Tuple[float, Dict[int, float]]:
        """
        Compute mean Average Precision (mAP) across all classes.
        
        Implements both training_mAP (17 fine-grained classes) and merged_mAP (8 main classes)
        as specified in loss.json.
        
        Args:
            merge_classes: If True, compute mAP for merged main classes (8 classes)
                         If False, compute mAP for fine-grained classes (17 classes)
            
        Returns:
            Tuple of (mAP, class_aps) where class_aps is a dictionary mapping
            class IDs to their AP values
        """
        num_classes = self.NUM_MAIN_CLASSES if merge_classes else self.num_fine_classes
        class_aps = {}
        
        # For each class, compute AP
        for class_id in range(num_classes):
            # Get all detections and ground truths for this class
            detections = []
            gts = []
            
            for img_dets, img_gts in zip(self.all_detections, self.all_gts):
                # Process detections
                img_class_dets = []
                for det in img_dets:
                    if len(det) == 0:
                        continue
                    
                    # Extract bbox, confidence, and class ID
                    x1, y1, x2, y2, conf, cls_id = det[:6]
                    
                    # Map to main class if needed
                    if merge_classes:
                        if cls_id not in self.fine_to_main:
                            logger.warning(f"Skipping detection with invalid class ID: {cls_id}")
                            continue  # Skip invalid class IDs
                            
                        # Map to main class and check if it matches current class_id
                        main_cls = self.fine_to_main[cls_id]
                        if main_cls != class_id:
                            continue  # Not the class we're looking for
                    else:
                        # For fine-grained mAP, filter by exact class match
                        if cls_id != class_id:
                            continue  # Not the class we're looking for
                    
                    # Add detection with confidence and bbox
                    img_class_dets.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id
                    })
                
                # Process ground truths for this image
                img_class_gts = []
                for gt in img_gts:
                    if len(gt) < 5:  # Need at least [x1, y1, x2, y2, class_id]
                        logger.warning(f"Skipping invalid ground truth with {len(gt)} elements")
                        continue
                    
                    gt_x1, gt_y1, gt_x2, gt_y2, gt_cls = gt[:5]
                    
                    # Map to main class if needed
                    if merge_classes:
                        if gt_cls not in self.fine_to_main:
                            logger.warning(f"Skipping ground truth with invalid class ID: {gt_cls}")
                            continue  # Skip invalid class IDs
                            
                        # Map to main class and check if it matches current class_id
                        main_cls = self.fine_to_main[gt_cls]
                        if main_cls != class_id:
                            continue  # Not the class we're looking for
                    else:
                        # For fine-grained mAP, filter by exact class match
                        if gt_cls != class_id:
                            continue  # Not the class we're looking for
                    
                    # Add ground truth bbox
                    img_class_gts.append({
                        'bbox': [gt_x1, gt_y1, gt_x2, gt_y2],
                        'class_id': gt_cls,
                        'detected': False  # Track if this GT has been matched with a detection
                    })
                
                detections.append(img_class_dets)
                gts.append(img_class_gts)
            
            # Flatten the lists
            all_dets = [det for img_dets in detections for det in img_dets]
            all_gts = [gt for img_gts in gts for gt in img_gts]
            
            # Sort detections by confidence (descending)
            all_dets.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Calculate precision and recall
            num_gts = len(all_gts)
            if num_gts == 0:
                # No ground truths for this class
                ap = 0.0
                logger.debug(f"No ground truths for class {class_id}, AP = 0.0")
            else:
                # Initialize TP and FP arrays
                tp = np.zeros(len(all_dets))
                fp = np.zeros(len(all_dets))
                
                # For each detection, check if it matches a ground truth
                for i, det in enumerate(all_dets):
                    # Find best matching ground truth
                    best_iou = self.iou_threshold
                    best_gt_idx = -1
                    
                    for j, gt in enumerate(all_gts):
                        iou = self.bbox_iou(det['bbox'], gt['bbox'])
                        if iou > best_iou and not gt.get('detected', False):
                            best_iou = iou
                            best_gt_idx = j
                    
                    if best_gt_idx >= 0:
                        # True positive (IOU > threshold and GT not yet matched)
                        tp[i] = 1
                        all_gts[best_gt_idx]['detected'] = True
                    else:
                        # False positive (no matching ground truth or IOU too low)
                        fp[i] = 1
                
                # Calculate precision and recall
                tp_cumsum = np.cumsum(tp)
                fp_cumsum = np.cumsum(fp)
                
                # Avoid division by zero
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
                recall = tp_cumsum / (num_gts + 1e-10)
                
                # Compute average precision
                ap = self.compute_ap(precision, recall)
            
            # Store AP for this class
            class_aps[class_id] = ap
            
            # Log AP for this class with appropriate name
            if merge_classes:
                class_name = f'main_class_{class_id}'
                if class_id < len(self.MAIN_CLASS_NAMES):
                    class_name = self.MAIN_CLASS_NAMES[class_id]
                logger.debug(f"AP for main class {class_name} ({class_id}): {ap:.4f}")
            else:
                class_name = f'class_{class_id}'
                if class_id < len(self.FINE_CLASS_NAMES):
                    class_name = self.FINE_CLASS_NAMES[class_id]
                logger.debug(f"AP for fine class {class_name} ({class_id}): {ap:.4f}")
        
        # Calculate mean AP (mAP) across all classes
        mean_ap = np.mean(list(class_aps.values())) if class_aps else 0.0
        
        # Log final mAP
        map_type = "merged mAP@0.5" if merge_classes else "fine mAP@0.5"
        logger.info(f"{map_type} = {mean_ap:.4f} (over {len(class_aps)} classes)")
        
        return mean_ap, class_aps
    
    @staticmethod
    def bbox_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            box1: [x1, y1, x2, y2] format
            box2: [x1, y1, x2, y2] format
            
        Returns:
            IoU value
        """
        # Determine the coordinates of the intersection rectangle
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Compute the area of intersection
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Compute the area of both bounding boxes
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # Compute the union area
        union = box1_area + box2_area - intersection
        
        # Compute IoU
        iou = intersection / (union + 1e-6)
        
        return iou
