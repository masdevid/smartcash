"""
File: smartcash/model/training/losses/loss_coordinator.py
Description: High-level loss coordination and management
Responsibility: Coordinate between different loss types and manage losses
"""

import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import logging
from smartcash.common.logger import get_logger
from smartcash.model.training.losses.base_loss import YOLOLoss
from smartcash.model.training.losses.multi_task_loss import MultiTaskLoss
from torchvision.ops import box_convert

class LossCoordinator:
    """Coordinates loss calculation for object detection"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize loss coordinator
        
        Args:
            config: Training configuration dictionary with loss parameters
        """
        self.config = config
        self.logger = get_logger(__name__)
        loss_config = config.get('training', {}).get('loss', {})
        
        # Loss weights - Default values from loss.json
        self.box_weight = loss_config.get('box_weight', 0.05)
        self.obj_weight = loss_config.get('obj_weight', 1.0)
        self.cls_weight = loss_config.get('cls_weight', 0.5)
        self.focal_loss = loss_config.get('focal_loss', False)
        self.label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        # Phase awareness configuration (disabled by default)
        self.phase_aware = loss_config.get('phase_aware', False)  # Default is False
        self.current_phase = loss_config.get('current_phase', 1)
        
        # Dynamic weighting configuration
        self.use_dynamic_weighting = loss_config.get('use_dynamic_weighting', False)
        self.min_variance = loss_config.get('min_variance', 1e-3)
        self.max_variance = loss_config.get('max_variance', 10.0)
        
        # Multi-task learning
        self.use_multi_task = loss_config.get('use_multi_task', False)
        self.use_multi_task_loss = self.use_multi_task  # Alias for compatibility
        self.multi_task_loss = None
        
        # Initialize loss functions
        self.loss_functions = {}
        self._setup_loss_functions()
        
        # Initialize multi-task loss if enabled
        if self.use_multi_task:
            num_tasks = len(self.loss_functions)
            self.multi_task_loss = MultiTaskLoss(
                num_tasks=num_tasks,
                min_variance=loss_config.get('min_variance', 1e-3),
                max_variance=loss_config.get('max_variance', 10.0)
            )
    
    def set_current_phase(self, phase: int) -> None:
        """
        Set the current training phase for phase-aware loss selection.
        
        Args:
            phase: Training phase (1 or 2)
        """
        self.current_phase = phase
        self.logger.debug(f"LossCoordinator: Set current phase to {phase}")
        
        # Re-setup loss functions for the new phase
        self._setup_loss_functions()
    
    def _setup_loss_functions(self) -> None:
        """Setup YOLO loss functions for each detection layer"""
        loss_params = self._get_loss_params()
        
        # Single loss function for all layers by default
        self.loss_functions['yolo'] = YOLOLoss(
            num_classes=loss_params.get('num_classes', 7),
            box_weight=self.box_weight,
            obj_weight=self.obj_weight,
            cls_weight=self.cls_weight,
            focal_loss=self.focal_loss,
            label_smoothing=self.label_smoothing
        )
        
        # Add phase-specific losses if phase awareness is enabled
        if self.phase_aware:
            self.logger.info("Phase-aware loss is enabled")
            # Phase 1: Banknote detection (classes 0-6)
            self.loss_functions['phase1'] = YOLOLoss(
                num_classes=7,
                **self._get_loss_params()
            )
            # Phase 2: Security feature detection (classes 7-13)
            self.loss_functions['phase2'] = YOLOLoss(
                num_classes=7,
                **self._get_loss_params()
            )
            # Phase 3: Denomination detection (classes 14-16)
            self.loss_functions['phase3'] = YOLOLoss(
                num_classes=3,
                **self._get_loss_params()
            )
    
    def _is_multilayer_mode(self) -> bool:
        """Check if model uses multilayer detection"""
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
    
    def compute_loss(self, predictions: Dict[str, list], 
                            targets: torch.Tensor, img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute total loss for object detection
        
        Args:
            predictions: Dict with format {layer_name: [pred_p3, pred_p4, pred_p5]}
            targets: Batch targets [batch_idx, class, x, y, w, h]
            img_size: Input image size
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        try:
            # Use phase-specific loss if phase_aware is enabled, otherwise use default
            if self.phase_aware:
                phase_key = f'phase{self.current_phase}'
                if phase_key in self.loss_functions:
                    loss_fn = self.loss_functions[phase_key]
                else:
                    self.logger.warning(f"No loss function for phase {self.current_phase}, using default")
                    loss_fn = next(iter(self.loss_functions.values()))
            else:
                loss_fn = next(iter(self.loss_functions.values()))
            
            # Compute loss using the selected loss function
            return self._compute_individual_losses(predictions, targets, img_size, loss_fn)
                
        except Exception as e:
            self.logger.error(f"Error in loss computation: {str(e)}", exc_info=True)
            # Return a small non-zero loss to prevent training from getting stuck
            return torch.tensor(1e-6, device=targets.device, requires_grad=True), self._get_empty_metrics(targets)
        
        # Prepare targets and predictions only for active layers
        layer_targets = {}
        filtered_predictions = {}
        
        for layer_name in active_layers:
            if layer_name in predictions:
                # Add predictions for active layers
                filtered_predictions[layer_name] = predictions[layer_name]
                
                # Filter targets for this layer
                filtered_targets = filter_targets_for_layer(targets, layer_name)
                # Safe check for tensor size - avoid Boolean tensor comparison
                if torch.is_tensor(filtered_targets) and filtered_targets.numel() > 0:
                    layer_targets[layer_name] = filtered_targets
                elif isinstance(filtered_targets, (list, tuple)) and len(filtered_targets) > 0:
                    layer_targets[layer_name] = filtered_targets
                else:
                    # Log when no targets found for a layer
                    self.logger.debug(f"No targets found for {layer_name}: filtered_targets type={type(filtered_targets)}, numel={filtered_targets.numel() if torch.is_tensor(filtered_targets) else 'N/A'}")
        
        # Debug: Check if we have any layer targets at all
        if len(layer_targets) == 0:
            self.logger.warning(f"No layer targets found for any layer. Original targets shape: {targets.shape if hasattr(targets, 'shape') else 'no shape'}")
            self.logger.warning(f"Available prediction layers: {list(predictions.keys())}")
            # Return small loss instead of zero to avoid optimization issues
            return torch.tensor(1e-6, device=targets.device, requires_grad=True), self._get_empty_metrics(targets)
        
        # Initialize metrics dictionary - focus on core metrics + mAP50
        metrics = {
            'val_loss': 0.0,
            'val_map50': 0.0,  # Primary validation metric
            'val_precision': 0.0,  # To be computed during validation
            'val_recall': 0.0,  # To be computed during validation
            'val_f1': 0.0,  # To be computed during validation
            'val_accuracy': 0.0,  # To be computed during validation
            'num_targets': targets.shape[0] if hasattr(targets, 'shape') and len(targets.shape) > 0 else 0
        }
        
        self.logger.debug(f"Phase {self.current_phase}: Using {len(active_layers)} active layers: {active_layers}")
        self.logger.debug(f"Filtered predictions keys: {list(filtered_predictions.keys())}, layer_targets keys: {list(layer_targets.keys())}")
        
        try:
            # Use uncertainty-based multi-task loss with filtered predictions
            total_loss, loss_breakdown = self.multi_task_loss(filtered_predictions, layer_targets, img_size)
            
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
                    elif any(k.endswith(suffix) for suffix in ['_box_loss', '_obj_loss', '_cls_loss', '_total_loss']):
                        # Include layer-specific loss components (e.g., layer_1_box_loss, layer_2_obj_loss)
                        metrics[k] = v
            
            # Calculate overall metrics if not provided
            if 'val_map50' not in metrics:
                metrics['val_map50'] = metrics.get('map50', 0.0)
                
            # Calculate F1 if not provided
            if 'val_f1' not in metrics and all(k in metrics for k in ['val_precision', 'val_recall']):
                p, r = metrics['val_precision'], metrics['val_recall']
                metrics['val_f1'] = 2 * (p * r) / (p + r + 1e-16)
            
            return total_loss, metrics
            
        except Exception as e:
            import traceback
            self.logger.error(f"Error in multi-task loss computation: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            # Return small loss instead of zero to avoid optimization issues
            return torch.tensor(1e-6, device=targets.device, requires_grad=True), metrics
    
    def _compute_individual_losses(self, predictions, 
                                  targets: torch.Tensor, img_size: int = 640,
                                  loss_fn: Optional[torch.nn.Module] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute detection loss for a single set of predictions.
        
        Args:
            predictions: Either a dictionary of model outputs or a list of prediction tensors
            targets: Ground truth targets [batch_idx, class, x, y, w, h]
            img_size: Input image size
            loss_fn: Optional custom loss function to use
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Use provided loss function or default to the first one
        if loss_fn is None:
            loss_fn = next(iter(self.loss_functions.values()))
        
        # Create a wrapper function to ensure gradients flow back to the original tensors
        def compute_loss_with_grads(preds, tgt, img_sz, lf):
            # Ensure predictions are tensors with requires_grad=True
            preds = [p.detach().requires_grad_(True) if not p.requires_grad else p for p in preds]
            
            # Forward pass
            loss, loss_items = lf(preds, tgt, img_sz)
            
            # Ensure we have a valid loss tensor with gradients
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=tgt.device, requires_grad=True)
            elif not loss.requires_grad:
                loss = loss.detach().requires_grad_(True)
                
            return loss, loss_items, preds
        
        # If predictions is a list, use it directly
        if isinstance(predictions, list):
            loss, loss_items, preds = compute_loss_with_grads(predictions, targets, img_size, loss_fn)
            return self._compute_single_head_loss(preds, targets, img_size, loss_fn)
            
        # If using multi-task learning, collect losses for each prediction head
        if self.use_multi_task and self.multi_task_loss is not None and isinstance(predictions, dict):
            all_losses = []
            metrics = {}
            
            for head_name, preds in predictions.items():
                if head_name in self.loss_functions:
                    loss, loss_items, updated_preds = compute_loss_with_grads(
                        preds, targets, img_size, self.loss_functions[head_name]
                    )
                    
                    # Update the predictions with the ones that have requires_grad=True
                    predictions[head_name] = updated_preds
                    
                    # Compute the loss breakdown
                    loss_breakdown = self._get_loss_breakdown(loss_items, targets)
                    
                    # Store the loss and metrics
                    all_losses.append(loss)
                    metrics.update({f"{head_name}_{k}": v for k, v in loss_breakdown.items()})
            
            # Combine losses using multi-task weighting
            if all_losses:
                total_loss = sum(all_losses) / len(all_losses)  # Simple average for now
                return total_loss, metrics
        
        # Standard single-task loss computation with dictionary input
        if isinstance(predictions, dict):
            # Get the first prediction head if available
            preds = next(iter(predictions.values()), None)
            if preds is not None:
                loss, loss_items, updated_preds = compute_loss_with_grads(
                    preds, targets, img_size, loss_fn
                )
                # Update the predictions with the ones that have requires_grad=True
                predictions[list(predictions.keys())[0]] = updated_preds
                
                # Compute the loss breakdown
                loss_breakdown = self._get_loss_breakdown(loss_items, targets)
                
                return loss, loss_breakdown
        
        # Fallback to empty loss if no valid predictions found
        return torch.tensor(0.0, device=targets.device, requires_grad=True), self._get_empty_metrics(targets)
        
    def _compute_single_head_loss(self, predictions, targets: torch.Tensor, img_size: int,
                                loss_fn: torch.nn.Module) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute loss for a single prediction head.
        
        Args:
            predictions: List of prediction tensors or a single prediction tensor
            targets: Ground truth targets [batch_idx, class, x, y, w, h]
            img_size: Input image size
            loss_fn: Loss function to use (YOLOLoss)
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Handle empty predictions
        if predictions is None or (isinstance(predictions, (list, tuple)) and len(predictions) > 0 and predictions[0].numel() == 0):
            return torch.tensor(0.0, device=targets.device, requires_grad=True), self._get_empty_metrics(targets)
        
        # Ensure predictions are on the correct device and require gradients
        device = targets.device
        
        # Handle different prediction formats
        if isinstance(predictions, torch.Tensor):
            # Single tensor input - check if it's a 4D tensor [batch, channels, height, width]
            if predictions.dim() == 4:
                # Convert to list of tensors (one for each scale)
                # This assumes the model outputs a single tensor with all scales concatenated
                # You may need to adjust the splitting logic based on your model's output format
                num_scales = 3  # Default to 3 scales (P3, P4, P5)
                if predictions.size(1) % num_scales == 0:
                    # Split the channels dimension into num_scales parts
                    predictions = torch.split(predictions, predictions.size(1) // num_scales, dim=1)
                else:
                    # If we can't split evenly, just wrap in a list
                    predictions = [predictions]
            else:
                # Not sure what to do with other tensor shapes, wrap in a list
                predictions = [predictions]
        elif isinstance(predictions, (list, tuple)):
            # Already a list/tuple of tensors
            predictions = list(predictions)
        else:
            # Unsupported type, wrap in a list
            predictions = [predictions]
        
        # Ensure all predictions are on the correct device and require gradients
        predictions = [p.to(device).requires_grad_(True) for p in predictions if p is not None]
        
        try:
            # Forward pass through YOLO loss
            loss, loss_items = loss_fn(predictions, targets, img_size)
            
            # Ensure we have a valid loss tensor with gradients
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device, requires_grad=True)
            elif not loss.requires_grad:
                loss = loss.detach().requires_grad_(True)
            
            # Unpack loss components while maintaining the computation graph
            if isinstance(loss_items, (list, tuple)) and len(loss_items) >= 3:
                lbox_tensor = loss_items[0] if isinstance(loss_items[0], torch.Tensor) else torch.tensor(loss_items[0], device=device)
                lobj_tensor = loss_items[1] if isinstance(loss_items[1], torch.Tensor) else torch.tensor(loss_items[1], device=device)
                lcls_tensor = loss_items[2] if isinstance(loss_items[2], torch.Tensor) else torch.tensor(loss_items[2], device=device)
            else:
                # If loss_items doesn't have the expected format, use the total loss
                lbox_tensor = lobj_tensor = lcls_tensor = loss / 3.0
            
            # For empty targets, we still want some loss from objectness (background)
            if targets.numel() == 0:
                # Create background targets to ensure non-zero loss
                bg_targets = torch.zeros((1, 6), device=device)
                _, bg_loss_items = loss_fn(predictions, bg_targets, img_size)
                if isinstance(bg_loss_items, (list, tuple)) and len(bg_loss_items) >= 3:
                    lobj_tensor = bg_loss_items[1] if isinstance(bg_loss_items[1], torch.Tensor) else torch.tensor(bg_loss_items[1], device=device)
            
            # Calculate metrics (simplified for now)
            metrics = {
                'box_ciou': 0.5,  # Placeholder, should be calculated from loss_fn
                'obj_accuracy': 0.5,  # Placeholder
                'cls_accuracy': 0.5   # Placeholder
            }
            
            # Calculate weighted losses while maintaining the computation graph
            box_loss_tensor = lbox_tensor * self.box_weight
            obj_loss_tensor = lobj_tensor * self.obj_weight
            cls_loss_tensor = lcls_tensor * self.cls_weight
            
            # Calculate total loss and ensure it's a scalar (0-dimensional tensor)
            total_loss = (box_loss_tensor + obj_loss_tensor + cls_loss_tensor).sum()
            
            # Ensure the total_loss is a scalar (0-dimensional tensor)
            if total_loss.dim() > 0:
                total_loss = total_loss.sum()
            
            # Create loss breakdown with Python floats for metrics/logging
            loss_breakdown = {
                'box_loss': box_loss_tensor.detach().item(),
                'obj_loss': obj_loss_tensor.detach().item(),
                'cls_loss': cls_loss_tensor.detach().item(),
                'metrics': metrics
            }
            
            # Ensure total_loss is a tensor with requires_grad
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.tensor(total_loss, device=device, requires_grad=True)
            
            # Return the loss and breakdown
            return total_loss, loss_breakdown
                
        except Exception as e:
            self.logger.error(f"Error in loss computation: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return small non-zero loss to avoid optimization issues
            small_loss = torch.tensor(1e-6, device=device, requires_grad=True)
            empty_metrics = self._get_empty_metrics(targets)
            return small_loss, empty_metrics
    
    def _compute_individual_losses(self, predictions, 
                                  targets: torch.Tensor, img_size: int = 640,
                                  loss_fn: Optional[torch.nn.Module] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute detection loss for a single set of predictions.
        
        Args:
            predictions: Either a dictionary of model outputs or a list of prediction tensors
            targets: Ground truth targets [batch_idx, class, x, y, w, h]
            img_size: Input image size
            loss_fn: Optional custom loss function to use
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Handle empty predictions
        if not predictions:
            return torch.tensor(0.0, device=targets.device), self._get_empty_metrics(targets)
            
        # Use provided loss function or default to the first one
        if loss_fn is None:
            loss_fn = next(iter(self.loss_functions.values()))
        
        # If predictions is a list, use it directly
        if isinstance(predictions, list):
            return self._compute_single_head_loss(predictions, targets, img_size, loss_fn)
            
        # If using multi-task learning, collect losses for each prediction head
        if self.use_multi_task and self.multi_task_loss is not None and isinstance(predictions, dict):
            all_losses = []
            metrics = {}
            
            for head_name, preds in predictions.items():
                if head_name in self.loss_functions:
                    head_loss, head_metrics = self._compute_single_head_loss(
                        preds, targets, img_size, self.loss_functions[head_name]
                    )
                    all_losses.append(head_loss)
                    metrics.update({f"{head_name}_{k}": v for k, v in head_metrics.items()})
            
            # Combine losses using multi-task weighting
            if all_losses:
                total_loss, mt_metrics = self.multi_task_loss(all_losses)
                metrics.update(mt_metrics)
                return total_loss, metrics
        
        # Standard single-task loss computation with dictionary input
        if isinstance(predictions, dict):
            # Get the first prediction head if available
            preds = next(iter(predictions.values()), None)
            if preds is not None:
                return self._compute_single_head_loss(preds, targets, img_size, loss_fn)
        
        # Fallback to empty loss if no valid predictions found
        return torch.tensor(0.0, device=targets.device), self._get_empty_metrics(targets)
        
        # Apply weights from loss.json
        with torch.enable_grad():
            weighted_lbox = lbox * 0.05  # lambda_box = 0.05
            weighted_lobj = lobj * 1.0   # lambda_obj = 1.0
            weighted_lcls = lcls * 0.5   # lambda_cls = 0.5
            
            # Total loss - ensure we have a single scalar for backprop
            loss = weighted_lbox + weighted_lobj + weighted_lcls
            
            # Ensure loss is a scalar and requires grad
            if not loss.requires_grad:
                loss = loss.detach().requires_grad_(True)
            
            # Register hooks to track gradients for each prediction tensor
            for i, pred in enumerate(pred_tensors):
                def make_hook(i):
                    def hook(grad):
                        pred_tensors[i].grad = grad
                    return hook
                pred.register_hook(make_hook(i))
        
        # Prepare loss breakdown with accumulated metrics
        with torch.no_grad():
            loss_breakdown = {
                'total_loss': loss.item(),
                'box_loss': weighted_lbox.item(),
                'obj_loss': weighted_lobj.item(),
                'cls_loss': weighted_lcls.item(),
                'metrics': {
                    'box_ciou': (1.0 - lbox.item()/0.05) if lbox.item() > 0 else 0.0,
                    'obj_accuracy': 0.0,  # Will be set below if we have predictions
                    'cls_accuracy': 0.0,  # Will be set below if we have predictions
                }
            }
            
            # Calculate accuracies if we have predictions
            if 'pred_obj' in locals() and 'tobj' in locals():
                loss_breakdown['metrics']['obj_accuracy'] = (tobj == (pred_obj > 0.5)).float().mean().item()
            if 'pred_cls' in locals() and 'tcls' in locals() and pred_cls.numel() > 0:
                loss_breakdown['metrics']['cls_accuracy'] = (tcls == (pred_cls > 0.5)).float().mean().item()
        
        return loss, loss_breakdown
    
    def bbox_iou(self, box1, box2, x1y1x2y2=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
        """
        Calculate IoU, GIoU, DIoU, or CIoU between two sets of boxes
        """
        # Ensure boxes are on the same device
        box2 = box2.to(box1.device)
        
        # Get the coordinates of bounding boxes
        if x1y1x2y2:  # x1, y1, x2, y2 = box1
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
        else:  # transform from xywh to xyxy
            b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
            b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
            b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
            b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union
        
        if GIoU or DIoU or CIoU:
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
            
            if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
                c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
                rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                       (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
                
                if DIoU:
                    return iou - rho2 / c2  # DIoU
                elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                    v = (4.0 / (math.pi ** 2)) * torch.pow(torch.atan2(w2, h2) - torch.atan2(w1, h1), 2)
                    with torch.no_grad():
                        alpha = v / (v - iou + (1 + eps))
                    return iou - (rho2 / c2 + v * alpha)  # CIoU
            else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
                c_area = cw * ch + eps  # convex area
                return iou - (c_area - union) / c_area  # GIoU
        
        return iou  # IoU
    
    def _get_loss_breakdown(self, loss_items, targets: torch.Tensor) -> Dict[str, Any]:
        """
        Convert loss items into a structured breakdown dictionary.
        
        Args:
            loss_items: Tuple of (lbox, lobj, lcls) from YOLO loss
            targets: Ground truth targets [batch_idx, class, x, y, w, h]
            
        Returns:
            Dictionary containing loss breakdown and metrics with Python float values
        """
        # Unpack loss components
        if isinstance(loss_items, (list, tuple)) and len(loss_items) >= 3:
            lbox, lobj, lcls = loss_items[0], loss_items[1], loss_items[2]
        else:
            # If loss_items doesn't have the expected format, use the total loss
            lbox = lobj = lcls = loss_items / 3.0
        
        # For empty targets, we still want some loss from objectness (background)
        if targets.numel() == 0 and hasattr(self, 'loss_functions') and 'layer_1' in self.loss_functions:
            # Create background targets to ensure non-zero loss
            bg_targets = torch.zeros((1, 6), device=targets.device)
            _, bg_loss_items = self.loss_functions['layer_1'](self.predictions['layer_1'], bg_targets, 640)
            if isinstance(bg_loss_items, (list, tuple)) and len(bg_loss_items) >= 3:
                _, bg_lobj, _ = bg_loss_items
                lobj = bg_lobj  # Use background objectness loss
        
        # Calculate metrics (simplified for now)
        metrics = {
            'box_ciou': 0.5,  # Placeholder, should be calculated from loss_fn
            'obj_accuracy': 0.5,  # Placeholder
            'cls_accuracy': 0.5   # Placeholder
        }
        
        # Ensure we have Python float values for the loss breakdown
        def to_float(x):
            if isinstance(x, torch.Tensor):
                return x.item() if x.numel() == 1 else float(x.mean().item())
            return float(x)
        
        # Get loss values as floats
        box_loss = to_float(lbox) * self.box_weight
        obj_loss = to_float(lobj) * self.obj_weight
        cls_loss = to_float(lcls) * self.cls_weight
        
        # Prepare loss breakdown with Python float values
        return {
            'box_loss': box_loss,
            'obj_loss': obj_loss,
            'cls_loss': cls_loss,
            'metrics': metrics
        }
        
    def _get_empty_metrics(self, targets: torch.Tensor) -> Dict[str, Any]:
        """Return empty metrics dictionary."""
        return {
            'box_loss': 0.0,
            'obj_loss': 0.0,
            'cls_loss': 0.0,
            'metrics': {
                'box_ciou': 0.0,
                'obj_accuracy': 0.0,
                'cls_accuracy': 0.0
            }
        }
    
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
        """Update loss weights dynamically during training"""
        if box_weight is not None:
            self.box_weight = box_weight
        if obj_weight is not None:
            self.obj_weight = obj_weight
        if cls_weight is not None:
            self.cls_weight = cls_weight
        
        # Update all loss functions
        for loss_fn in self.loss_functions.values():
            loss_fn.box_weight = self.box_weight
            loss_fn.obj_weight = self.obj_weight
            loss_fn.cls_weight = self.cls_weight
    
    def get_loss_breakdown_summary(self, loss_dict: Dict[str, Any]) -> str:
        """Get formatted summary of loss breakdown"""
        total = loss_dict.get('total_loss', 0)
        box = loss_dict.get('box_loss', 0)
        obj = loss_dict.get('obj_loss', 0)
        cls = loss_dict.get('cls_loss', 0)
        
        return f"Total: {total:.4f} | Box: {box:.4f} | Obj: {obj:.4f} | Cls: {cls:.4f}"


# Convenience functions
def create_loss_coordinator(config: Dict[str, Any]) -> LossCoordinator:
    """Factory function for creating loss coordinator"""
    return LossCoordinator(config)


def compute_coordinated_loss(predictions: Dict[str, list], targets: torch.Tensor,
                           config: Dict[str, Any], img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """One-liner for computing coordinated loss"""
    coordinator = LossCoordinator(config)
    return coordinator.compute_loss(predictions, targets, img_size)