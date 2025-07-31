#!/usr/bin/env python3
"""
mAP calculation for the unified training pipeline.

This module handles mean Average Precision calculation for object detection
metrics using YOLO prediction formats.
"""

import torch
from typing import Dict

from smartcash.common.logger import get_logger
from smartcash.model.training.metrics_tracker import APCalculator

logger = get_logger(__name__)


class MAPCalculator:
    """Handles mAP calculation for object detection metrics."""
    
    def __init__(self):
        """Initialize mAP calculator."""
        self.ap_calculator = APCalculator()
    
    def process_batch_for_map(self, layer_preds, targets, image_shape, device, batch_idx):
        """
        Process batch predictions for mAP calculation.
        
        Args:
            layer_preds: Layer predictions (list, tuple, or tensor)
            targets: Target tensor
            image_shape: Shape of input images
            device: Device for tensor operations
            batch_idx: Current batch index
        """
        # Handle list, tuple, and tensor formats for mAP computation
        mAP_processed = False
        
        if isinstance(layer_preds, (list, tuple)) and len(layer_preds) > 0:
            # List/tuple format (expected YOLOv5 multi-scale output)
            try:
                if batch_idx == 0:
                    logger.info(f"mAP Debug - batch {batch_idx}: Processing {type(layer_preds).__name__} format with {len(layer_preds)} scales")
                self._add_to_map_calculator(self.ap_calculator, layer_preds, targets, image_shape, device, batch_idx)
                mAP_processed = True
            except Exception as e:
                if batch_idx < 5:
                    logger.warning(f"mAP processing failed for batch {batch_idx}: {e}")
                    import traceback
                    logger.warning(f"mAP traceback: {traceback.format_exc()}")
                    
        elif isinstance(layer_preds, torch.Tensor) and layer_preds.numel() > 0:
            # Single tensor format - wrap in list for compatibility
            try:
                if batch_idx == 0:
                    logger.info(f"mAP Debug - batch {batch_idx}: Processing tensor format {layer_preds.shape}, wrapping in list")
                wrapped_preds = [layer_preds]
                self._add_to_map_calculator(self.ap_calculator, wrapped_preds, targets, image_shape, device, batch_idx)
                mAP_processed = True
            except Exception as e:
                if batch_idx < 5:
                    logger.warning(f"mAP processing (tensor) failed for batch {batch_idx}: {e}")
                    import traceback
                    logger.warning(f"mAP traceback: {traceback.format_exc()}")
        
        if batch_idx == 0 and not mAP_processed:
            logger.warning(f"mAP Debug - batch {batch_idx}: No mAP processing - layer_preds type: {type(layer_preds)}, shape/len: {getattr(layer_preds, 'shape', len(layer_preds) if hasattr(layer_preds, '__len__') else 'N/A')}")
    
    def compute_final_map(self) -> Dict[str, float]:
        """
        Compute final mAP metrics.
        
        Returns:
            Dictionary containing mAP metrics
        """
        map_metrics = {}
        try:
            # Check if AP calculator has any data
            num_predictions = len(self.ap_calculator.predictions)
            num_targets = len(self.ap_calculator.targets)
            logger.info(f"mAP Calculator Data: {num_predictions} predictions, {num_targets} targets")
            
            if num_predictions == 0 or num_targets == 0:
                logger.warning(f"⚠️ Insufficient data for mAP computation: {num_predictions} predictions, {num_targets} targets")
                map_metrics['val_map50'] = 0.0
                map_metrics['val_map50_95'] = 0.0
            else:
                # Focus on mAP@0.5 (mAP50) as primary metric
                map50, class_aps = self.ap_calculator.compute_map(iou_threshold=0.5)
                map_metrics['val_map50'] = float(map50)
                
                # Compute mAP@0.5:0.95 (can be slower, used for final evaluation)
                map50_95 = self.ap_calculator.compute_map50_95()
                map_metrics['val_map50_95'] = float(map50_95)
                
                logger.info(f"✅ Computed mAP metrics: mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f}")
                if class_aps:
                    logger.debug(f"Per-class APs: {class_aps}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error computing mAP (falling back to 0.0): {e}")
            import traceback
            logger.debug(f"mAP computation traceback: {traceback.format_exc()}")
            map_metrics['val_map50'] = 0.0
            map_metrics['val_map50_95'] = 0.0
        
        return map_metrics
    
    def _add_to_map_calculator(self, ap_calculator, layer_preds, targets, image_shape, device, batch_idx):
        """
        Extract bounding box predictions and add to mAP calculator.
        
        Args:
            ap_calculator: APCalculator instance
            layer_preds: List of YOLO predictions [scales]
            targets: Target tensor [num_targets, 6] format [batch_idx, class, x, y, w, h]
            image_shape: Shape of input images
            device: Device for tensors
            batch_idx: Current batch index for logging
        """
        try:
            # Enhanced safety checks for layer_preds
            if not layer_preds:
                if batch_idx < 3:
                    logger.warning(f"mAP - batch {batch_idx}: layer_preds is None or falsy")
                return
                
            if not hasattr(layer_preds, '__len__'):
                if batch_idx < 3:
                    logger.warning(f"mAP - batch {batch_idx}: layer_preds has no length attribute")
                return
                
            if len(layer_preds) == 0:
                if batch_idx < 3:
                    logger.warning(f"mAP - batch {batch_idx}: Empty layer_preds list")
                return
                
            # Process first scale for simplicity (can be extended to all scales)
            # Enhanced safety checks with proper bounds checking
            if not layer_preds or len(layer_preds) == 0:
                if batch_idx < 3:
                    logger.warning(f"mAP - batch {batch_idx}: Empty layer_preds list")
                return
                
            if layer_preds[0] is None:
                if batch_idx < 3:
                    logger.warning(f"mAP - batch {batch_idx}: First element in layer_preds is None")
                return
                
            scale_pred = layer_preds[0]
            if batch_idx < 2:  # Log format for first few batches
                logger.info(f"mAP Debug - batch {batch_idx}: layer_preds type {type(layer_preds)}, length {len(layer_preds) if hasattr(layer_preds, '__len__') else 'N/A'}")
                logger.info(f"mAP Debug - batch {batch_idx}: scale_pred type {type(scale_pred)}, shape {scale_pred.shape if hasattr(scale_pred, 'shape') else 'N/A'}")
            
            # Handle both formats: 4D [batch, anchors, grid_y, grid_x, features] and 3D [batch, detections, features]
            if isinstance(scale_pred, torch.Tensor):
                flat_pred = self._flatten_predictions(scale_pred, batch_idx)
                if flat_pred is None:
                    return
            else:
                if batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}: scale_pred is not a tensor: {type(scale_pred)}")
                return
            
            # Extract YOLO prediction components
            obj_conf = torch.sigmoid(flat_pred[..., 4])  # Objectness confidence
            class_logits = flat_pred[..., 5:]       # Class logits
            class_probs = torch.sigmoid(class_logits)  # Class probabilities
            
            # Process coordinates and filter predictions
            bbox_coords = self._process_coordinates(flat_pred, scale_pred, image_shape, device)
            filtered_data = self._filter_predictions(
                bbox_coords, obj_conf, class_probs, batch_idx
            )
            
            if filtered_data:
                # Process each image in batch
                self._process_batch_images(
                    filtered_data, targets, image_shape, device, batch_idx, ap_calculator
                )
            elif batch_idx < 3:
                logger.warning(f"mAP - batch {batch_idx}: _filter_predictions returned None or empty data")
                    
        except Exception as e:
            if batch_idx < 3:  # Only log first few failures
                logger.warning(f"mAP extraction failed for batch {batch_idx}: {e}")
                import traceback
                logger.warning(f"mAP extraction traceback: {traceback.format_exc()}")
    
    def _flatten_predictions(self, scale_pred, batch_idx):
        """Flatten prediction tensor to consistent format."""
        if scale_pred is None:
            if batch_idx < 2:
                logger.warning(f"mAP Debug - batch {batch_idx}: scale_pred is None")
            return None
            
        if scale_pred.dim() == 5:
            # YOLO format: [batch_size, num_anchors, grid_height, grid_width, features]
            batch_size, num_anchors, grid_h, grid_w, num_features = scale_pred.shape
            if num_features < 7:  # Need at least x,y,w,h,obj + 2 classes
                if batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}: Insufficient features {num_features}, need >= 7")
                return None
            # Flatten to [batch_size, num_detections, features] where num_detections = num_anchors * grid_h * grid_w
            flattened = scale_pred.view(batch_size, -1, num_features)
            if batch_idx < 2:
                logger.debug(f"mAP Debug - batch {batch_idx}: Flattened 5D YOLO format [batch={batch_size}, anchors={num_anchors}, grid={grid_h}x{grid_w}, features={num_features}] -> [batch={batch_size}, detections={flattened.shape[1]}, features={num_features}]")
            return flattened
        elif scale_pred.dim() == 4:
            # 4D format [batch, anchors, grid_y, grid_x, features]
            batch_size, _, _, _, num_features = scale_pred.shape
            if num_features < 7:  # Need at least x,y,w,h,obj + 2 classes
                if batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}: Insufficient features {num_features}, need >= 7")
                return None
            # Flatten spatial dimensions: [batch, num_anchors * grid_h * grid_w, features]
            return scale_pred.view(batch_size, -1, num_features)
            
        elif scale_pred.dim() == 3:
            # 3D format [batch, detections, features] - modern YOLOv5 output
            batch_size, num_detections, num_features = scale_pred.shape
            if num_features < 7:  # Need at least x,y,w,h,obj + 2 classes
                if batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}: Insufficient features {num_features}, need >= 7")
                return None
            if batch_idx < 2:
                logger.info(f"mAP Debug - batch {batch_idx}: Using 3D format [batch={batch_size}, detections={num_detections}, features={num_features}]")
            return scale_pred
        else:
            if batch_idx < 2:
                logger.warning(f"mAP Debug - batch {batch_idx}: Invalid tensor format - expected 3D, 4D, or 5D tensor, got {scale_pred.shape}")
            return None
    
    def _process_coordinates(self, flat_pred, scale_pred, image_shape, device):
        """Process coordinate predictions based on tensor format."""
        # Safety check for inputs
        if flat_pred is None or scale_pred is None or image_shape is None:
            logger.warning("Invalid inputs to _process_coordinates")
            return None
            
        img_h, img_w = image_shape[-2:]
        
        if scale_pred.dim() == 4:
            # 4D format: need to convert grid coordinates
            return self._process_4d_coordinates(flat_pred, scale_pred, img_h, img_w, device)
        else:
            # 3D format: coordinates are already processed by YOLOv5
            return self._process_3d_coordinates(flat_pred)
    
    def _process_4d_coordinates(self, flat_pred, scale_pred, img_h, img_w, device):
        """Process 4D coordinate format."""
        _, _, grid_h, grid_w, _ = scale_pred.shape
        
        xy = torch.sigmoid(flat_pred[..., :2])  # Center coordinates (0-1)
        wh = torch.exp(flat_pred[..., 2:4])     # Width/height (relative to anchors)
        
        grid_scale_x = img_w / grid_w
        grid_scale_y = img_h / grid_h
        
        # Create grid coordinates
        grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).view(1, 1, grid_h, grid_w)
        grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t().view(1, 1, grid_h, grid_w)
        grid_xy = torch.cat([grid_x, grid_y], dim=1).float()
        grid_xy = grid_xy.view(1, 2, -1).permute(0, 2, 1)  # [1, grid_cells, 2]
        
        # Convert center coordinates to absolute
        xy_abs = (xy + grid_xy) * torch.tensor([grid_scale_x, grid_scale_y], device=device)
        
        # Convert to corner coordinates [x1, y1, x2, y2]
        wh_abs = wh * torch.tensor([grid_scale_x, grid_scale_y], device=device) * 0.5  # Scale factor
        x1 = xy_abs[..., 0] - wh_abs[..., 0]
        y1 = xy_abs[..., 1] - wh_abs[..., 1]
        x2 = xy_abs[..., 0] + wh_abs[..., 0]
        y2 = xy_abs[..., 1] + wh_abs[..., 1]
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _process_3d_coordinates(self, flat_pred):
        """Process 3D coordinate format."""
        # Assume format: [x_center, y_center, width, height] in image coordinates
        xy_center = flat_pred[..., :2]  # Center coordinates (already in pixels)
        wh_size = flat_pred[..., 2:4]   # Width/height (already in pixels)
        
        # Convert to corner coordinates [x1, y1, x2, y2] 
        x1 = xy_center[..., 0] - wh_size[..., 0] * 0.5
        y1 = xy_center[..., 1] - wh_size[..., 1] * 0.5
        x2 = xy_center[..., 0] + wh_size[..., 0] * 0.5
        y2 = xy_center[..., 1] + wh_size[..., 1] * 0.5
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _filter_predictions(self, bbox_coords, obj_conf, class_probs, batch_idx):
        """Filter predictions by confidence threshold."""
        # Safety checks for input tensors
        if bbox_coords is None or obj_conf is None or class_probs is None:
            if batch_idx < 3:
                logger.warning(f"mAP - batch {batch_idx}: Invalid input tensors to _filter_predictions")
            return None
            
        # Filter by confidence threshold (reasonable threshold to avoid performance issues)
        conf_thresh = 0.1  # Balanced threshold to avoid processing too many predictions
        max_preds_per_img = 1000  # Limit predictions per image for performance
        final_scores = obj_conf * class_probs.max(dim=-1)[0]
        conf_mask = final_scores > conf_thresh
        
        if batch_idx < 2:
            total_detections = conf_mask.sum().item()
            max_obj_conf = obj_conf.max().item() if obj_conf.numel() > 0 else 0
            max_class_prob = class_probs.max().item() if class_probs.numel() > 0 else 0
            max_final_score = final_scores.max().item() if final_scores.numel() > 0 else 0
            logger.info(f"mAP Debug - batch {batch_idx}: Found {total_detections} detections above threshold {conf_thresh}")
            logger.info(f"  Max objectness: {max_obj_conf:.4f}, Max class prob: {max_class_prob:.4f}, Max final score: {max_final_score:.4f}")
            if total_detections == 0:
                logger.warning(f"  No detections found! All {final_scores.numel()} predictions below threshold")
        
        return {
            'bbox_coords': bbox_coords,
            'final_scores': final_scores,
            'class_probs': class_probs,
            'conf_mask': conf_mask,
            'max_preds_per_img': max_preds_per_img
        }
    
    def _process_batch_images(self, filtered_data, targets, image_shape, device, batch_idx, ap_calculator):
        """Process each image in the batch for mAP calculation."""
        # Safety check for filtered_data
        if not filtered_data:
            if batch_idx < 3:
                logger.warning(f"mAP - batch {batch_idx}: Empty filtered_data")
            return
            
        bbox_coords = filtered_data['bbox_coords']
        final_scores = filtered_data['final_scores']
        class_probs = filtered_data['class_probs']
        conf_mask = filtered_data['conf_mask']
        max_preds_per_img = filtered_data['max_preds_per_img']
        
        # Additional safety checks
        if bbox_coords is None or final_scores is None or class_probs is None or conf_mask is None:
            if batch_idx < 3:
                logger.warning(f"mAP - batch {batch_idx}: Invalid filtered_data components")
            return
        
        img_h, img_w = image_shape[-2:]
        batch_size = bbox_coords.shape[0] if hasattr(bbox_coords, 'shape') and len(bbox_coords.shape) > 0 else 0
        
        # Check if we have valid batch size
        if batch_size <= 0:
            if batch_idx < 3:
                logger.warning(f"mAP - batch {batch_idx}: Invalid batch_size: {batch_size}")
            return
        
        # Process each image in batch
        for img_idx in range(batch_size):
            img_mask = conf_mask[img_idx]
            if not img_mask.any():
                continue
            
            # Limit number of predictions per image for performance
            img_scores = final_scores[img_idx][img_mask]
            if len(img_scores) > max_preds_per_img:
                # Keep only top-scored predictions
                top_k_indices = torch.topk(img_scores, max_preds_per_img)[1]
                img_indices = torch.where(img_mask)[0][top_k_indices]
                img_mask = torch.zeros_like(img_mask)
                img_mask[img_indices] = True
            
            # Get valid predictions for this image
            valid_boxes = bbox_coords[img_idx][img_mask]
            valid_scores = final_scores[img_idx][img_mask]
            valid_classes = class_probs[img_idx][img_mask].argmax(dim=-1)
            
            # Get ground truth for this image
            img_targets = targets[targets[:, 0] == img_idx]  # Filter by batch index
            if len(img_targets) == 0:
                continue
            
            # Convert target format [class, x_center, y_center, width, height] to [x1, y1, x2, y2]
            gt_classes = img_targets[:, 1].long()
            gt_centers = img_targets[:, 2:4] * torch.tensor([img_w, img_h], device=device)
            gt_sizes = img_targets[:, 4:6] * torch.tensor([img_w, img_h], device=device)
            gt_x1 = gt_centers[:, 0] - gt_sizes[:, 0] * 0.5
            gt_y1 = gt_centers[:, 1] - gt_sizes[:, 1] * 0.5
            gt_x2 = gt_centers[:, 0] + gt_sizes[:, 0] * 0.5
            gt_y2 = gt_centers[:, 1] + gt_sizes[:, 1] * 0.5
            gt_boxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)
            
            # Add to AP calculator (reshape to expected format)
            if len(valid_boxes) > 0 and len(gt_boxes) > 0:
                if batch_idx < 2:
                    logger.info(f"mAP Debug - batch {batch_idx}, img {img_idx}: Adding {len(valid_boxes)} predictions (max {max_preds_per_img}) and {len(gt_boxes)} targets to AP calculator")
                    logger.info(f"  Pred scores range: {valid_scores.min():.4f} - {valid_scores.max():.4f}")
                    logger.info(f"  Pred classes: {valid_classes.unique().tolist()}")
                    logger.info(f"  GT classes: {gt_classes.unique().tolist()}")
                
                ap_calculator.add_batch(
                    pred_boxes=valid_boxes.unsqueeze(0),      # [1, num_preds, 4]
                    pred_scores=valid_scores.unsqueeze(0),    # [1, num_preds]
                    pred_classes=valid_classes.unsqueeze(0),  # [1, num_preds]
                    true_boxes=gt_boxes.unsqueeze(0),         # [1, num_targets, 4]
                    true_classes=gt_classes.unsqueeze(0),     # [1, num_targets]
                    image_ids=[img_idx]                       # [1]
                )
            elif batch_idx < 2:
                logger.warning(f"mAP Debug - batch {batch_idx}, img {img_idx}: Skipping - valid_boxes: {len(valid_boxes)}, gt_boxes: {len(gt_boxes)}")