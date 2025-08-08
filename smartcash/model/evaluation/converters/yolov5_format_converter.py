"""
YOLOv5 format converter for evaluation data.
Converts evaluation format to YOLOv5 training format for mAP calculation.
"""

import torch
from typing import Dict, List, Any, Optional, Tuple

from smartcash.common.logger import get_logger


class YOLOv5FormatConverter:
    """Convert evaluation format to YOLOv5 training format"""
    
    def __init__(self, num_classes: int = 17):
        self.logger = get_logger('yolov5_format_converter')
        self.num_classes = num_classes
    
    def convert_to_yolo_format(self, predictions: List[Dict], ground_truths: List[Dict]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert evaluation format to YOLOv5 training format"""
        try:
            # Training module expects predictions as [batch_size, max_detections, 6] 
            # where each detection is [x, y, w, h, conf, class] in YOLO format (not xyxy)
            batch_size = len(predictions)
            max_detections = 100  # Use reasonable max
            
            # Initialize prediction tensor
            pred_tensor = torch.zeros((batch_size, max_detections, 6), dtype=torch.float32)
            
            for batch_idx, pred in enumerate(predictions):
                det_count = 0
                for detection in pred.get('detections', []):
                    if det_count >= max_detections:
                        break
                        
                    if 'class_id' in detection and 'confidence' in detection and 'bbox' in detection:
                        bbox = detection['bbox']  # [x_center, y_center, width, height] normalized
                        if len(bbox) == 4:
                            # Keep hierarchical classes as-is (0-16) for multi-layer evaluation
                            class_id = int(detection['class_id'])
                            
                            # Keep normalized YOLO format [x, y, w, h] as expected by training module
                            pred_tensor[batch_idx, det_count, :] = torch.tensor([
                                float(bbox[0]),  # x_center (normalized)
                                float(bbox[1]),  # y_center (normalized)
                                float(bbox[2]),  # width (normalized)
                                float(bbox[3]),  # height (normalized)
                                float(detection['confidence']),  # confidence
                                float(class_id)  # class (hierarchical 0-16)
                            ])
                            det_count += 1
            
            # Convert targets to tensor format [N, 6] where each row is [batch_idx, class, x, y, w, h]
            target_list = []
            batch_idx = 0
            
            for gt in ground_truths:
                for annotation in gt.get('annotations', []):
                    if 'class_id' in annotation and 'bbox' in annotation:
                        bbox = annotation['bbox']  # [x_center, y_center, width, height] normalized
                        if len(bbox) == 4:
                            # Keep hierarchical classes as-is (0-16) for multi-layer evaluation
                            class_id = int(annotation['class_id'])
                            
                            target_list.append([
                                batch_idx,  # batch index
                                float(class_id),  # class (hierarchical 0-16)
                                float(bbox[0]),  # x_center (normalized)
                                float(bbox[1]),  # y_center (normalized)
                                float(bbox[2]),  # width (normalized)
                                float(bbox[3])   # height (normalized)
                            ])
                batch_idx += 1
            
            if pred_tensor.sum() == 0 or not target_list:
                self.logger.warning(f"Empty conversions: pred_tensor_sum={pred_tensor.sum()}, targets={len(target_list)}")
                return None, None
            
            # Convert targets to tensor
            targets_tensor = torch.tensor(target_list, dtype=torch.float32)
            
            self.logger.info(f"📊 Converted to YOLOv5 format: pred_shape={pred_tensor.shape}, target_shape={targets_tensor.shape}")
            
            # Debug: show actual tensor contents
            pred_count = (pred_tensor.sum(dim=-1) != 0).sum()
            target_count = targets_tensor.shape[0]
            self.logger.info(f"📊 Non-zero predictions: {pred_count}, Targets: {target_count}")
            
            # Show some sample predictions and targets
            if pred_count > 0:
                # Find first non-zero prediction
                for i in range(pred_tensor.shape[0]):
                    for j in range(pred_tensor.shape[1]):
                        if pred_tensor[i, j].sum() != 0:
                            self.logger.info(f"📊 Sample prediction [{i},{j}]: {pred_tensor[i, j].tolist()}")
                            break
                    if pred_tensor[i, j].sum() != 0:
                        break
            
            if target_count > 0:
                self.logger.info(f"📊 Sample target [0]: {targets_tensor[0].tolist()}")
            
            return pred_tensor, targets_tensor
            
        except Exception as e:
            self.logger.error(f"Error converting to YOLOv5 format: {e}")
            return None, None


def create_yolov5_format_converter(num_classes: int = 17) -> YOLOv5FormatConverter:
    """Factory function to create YOLOv5 format converter"""
    return YOLOv5FormatConverter(num_classes=num_classes)