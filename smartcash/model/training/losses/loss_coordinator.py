"""
File: smartcash/model/training/losses/loss_coordinator.py
Description: High-level loss coordination and management
Responsibility: Coordinate between different loss types, manage multi-task losses, and handle phase-aware configurations
"""

import torch
from typing import Dict, Tuple, Any, Optional

from smartcash.common.logger import get_logger
from .base_loss import YOLOLoss
from .target_builder import filter_targets_for_layer


class LossCoordinator:
    """Coordinates loss calculation for single or multilayer detection"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize loss coordinator
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        loss_config = config.get('training', {}).get('loss', {})
        
        # Loss weights
        self.box_weight = loss_config.get('box_weight', 0.05)
        self.obj_weight = loss_config.get('obj_weight', 4.0)
        self.cls_weight = loss_config.get('cls_weight', 0.5)
        self.focal_loss = loss_config.get('focal_loss', False)
        self.label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        # Phase awareness for proper loss selection
        self.current_phase = config.get('current_phase', 1)  # Read from config, default to Phase 1
        
        # Dynamic weighting parameters
        self.use_dynamic_weighting = loss_config.get('dynamic_weighting', True)
        self.min_variance = loss_config.get('min_variance', 1e-3)
        self.max_variance = loss_config.get('max_variance', 10.0)
        
        # Initialize loss functions for different layer modes
        self.loss_functions = {}
        self.use_multi_task_loss = False
        self._setup_loss_functions()
    
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
        """Setup loss functions based on model configuration"""
        # Check if we should use uncertainty-based multi-task loss
        loss_type = self.config.get('training', {}).get('loss', {}).get('type', 'uncertainty_multi_task')
        
        # Use multi-task loss for ALL phases with dynamic weighting
        should_use_multitask = (loss_type == 'uncertainty_multi_task' and self._is_multilayer_mode())
        
        if should_use_multitask:
            # Use MODEL_ARC.md compliant uncertainty-based multi-task loss with phase-specific configuration
            from smartcash.model.training.multi_task_loss import UncertaintyMultiTaskLoss
            
            # Phase-specific layer configuration according to phase-loss.json
            if self.current_phase == 1:
                # Phase 1: Only layer_1 active (weight=1.0), layer_2/layer_3 inactive (weight=0.0)
                layer_config = {
                    'layer_1': {'description': 'Full banknote detection', 'num_classes': 7}
                }
                self.logger.debug(f"Phase {self.current_phase}: Using uncertainty multi-task loss (1 layer)")
            else:
                # Phase 2: All layers active with uncertainty-based weighting
                layer_config = {
                    'layer_1': {'description': 'Full banknote detection', 'num_classes': 7},
                    'layer_2': {'description': 'Denomination-specific features', 'num_classes': 7}, 
                    'layer_3': {'description': 'Common features', 'num_classes': 3}
                }
                self.logger.debug(f"Phase {self.current_phase}: Using uncertainty multi-task loss (3 layers)")
            
            loss_config = {
                'box_weight': self.box_weight,
                'obj_weight': self.obj_weight,
                'cls_weight': self.cls_weight,
                'focal_loss': self.focal_loss,
                'label_smoothing': self.label_smoothing,
                'dynamic_weighting': self.use_dynamic_weighting,
                'min_variance': self.min_variance,
                'max_variance': self.max_variance
            }
            
            # Create UncertaintyMultiTaskLoss directly with phase-specific layer configuration
            self.multi_task_loss = UncertaintyMultiTaskLoss(
                layer_config=layer_config,
                loss_config=loss_config
            )
            self.use_multi_task_loss = True
        else:
            # Use individual YOLO losses for backward compatibility
            self.use_multi_task_loss = False
            self._setup_individual_losses()
            self.logger.debug(f"Phase {self.current_phase}: Using individual YOLO losses (fallback)")
    
    def _setup_individual_losses(self) -> None:
        """Setup individual YOLO loss functions for each layer"""
        # MODEL_ARC.md compliant layer names - Phase-specific classes
        self.loss_functions['layer_1'] = YOLOLoss(
            num_classes=7,  # Layer 1 classes only (0-6) for Phase 1
            box_weight=self.box_weight,
            obj_weight=self.obj_weight,
            cls_weight=self.cls_weight,
            focal_loss=self.focal_loss,
            label_smoothing=self.label_smoothing
        )
        
        # Additional layers if required
        if self._is_multilayer_mode():
            self.loss_functions['layer_2'] = YOLOLoss(num_classes=7, **self._get_loss_params())   # Layer 2: 7 classes (7-13)
            self.loss_functions['layer_3'] = YOLOLoss(num_classes=3, **self._get_loss_params())   # Layer 3: 3 classes (14-16)
        
        # Legacy support for backward compatibility
        self.loss_functions['banknote'] = self.loss_functions['layer_1']
        if self._is_multilayer_mode():
            self.loss_functions['nominal'] = self.loss_functions['layer_2']
            self.loss_functions['security'] = self.loss_functions['layer_3']
    
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
        Compute total loss for single or multilayer detection
        
        Args:
            predictions: Dict with format {layer_name: [pred_p3, pred_p4, pred_p5]}
            targets: Batch targets [batch_idx, class, x, y, w, h]
            img_size: Input image size
            
        Returns:
            total_loss: Combined loss
            loss_breakdown: Detailed loss components
        """
        # Use MODEL_ARC.md compliant uncertainty-based multi-task loss if available
        if hasattr(self, 'use_multi_task_loss') and self.use_multi_task_loss:
            self.logger.debug(f"Using multi-task loss computation for {len(predictions)} layers")
            return self._compute_multi_task_loss(predictions, targets, img_size)
        else:
            self.logger.debug(f"Using individual loss computation for {len(predictions)} layers")
            return self._compute_individual_losses(predictions, targets, img_size)
    
    def _compute_multi_task_loss(self, predictions: Dict[str, list], 
                                targets: torch.Tensor, img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss using MODEL_ARC.md compliant uncertainty-based multi-task loss"""
        # Get active layers based on current phase (according to phase-loss.json)
        if self.current_phase == 1:
            active_layers = ['layer_1']  # Phase 1: Only layer_1 (weight=1.0)
        else:
            active_layers = ['layer_1', 'layer_2', 'layer_3']  # Phase 2: All layers with uncertainty weighting
        
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
    
    def _compute_individual_losses(self, predictions: Dict[str, list], 
                                  targets: torch.Tensor, img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss using individual YOLO losses (backward compatibility)"""
        device = targets.device if hasattr(targets, 'shape') and targets.numel() > 0 else torch.device('cpu')
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_breakdown = {}
        
        # Debug: Check if we have targets
        if not hasattr(targets, 'shape') or targets.numel() == 0:
            self.logger.warning(f"Individual loss computation: No targets available. Targets type: {type(targets)}")
            return torch.tensor(1e-6, device=device, requires_grad=True), loss_breakdown
        
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
            layer_losses = []
            active_layers = 0
            
            for layer_name, layer_preds in predictions.items():
                if layer_name in self.loss_functions:
                    # Filter targets for this layer
                    layer_targets = filter_targets_for_layer(targets, layer_name)
                    
                    # Safe check for tensor size - avoid Boolean tensor comparison
                    has_targets = False
                    if torch.is_tensor(layer_targets) and layer_targets.numel() > 0:
                        has_targets = True
                    elif isinstance(layer_targets, (list, tuple)) and len(layer_targets) > 0:
                        has_targets = True
                    
                    if has_targets:
                        loss_fn = self.loss_functions[layer_name]
                        layer_loss, layer_components = loss_fn(layer_preds, layer_targets, img_size)
                        
                        # Add layer prefix to component names
                        prefixed_components = {f"{layer_name}_{k}": v for k, v in layer_components.items()}
                        loss_breakdown.update(prefixed_components)
                        
                        layer_losses.append(layer_loss)
                        active_layers += 1
            
            # Average the losses instead of summing to prevent high values in multi-layer mode
            if layer_losses:
                if active_layers > 1:
                    # Multi-layer: average the losses
                    total_loss = torch.stack(layer_losses).mean()
                    pass  # Removed verbose logging
                else:
                    # Single layer: use the loss directly
                    total_loss = layer_losses[0]
        
        # Add overall metrics
        loss_breakdown['total_loss'] = total_loss
        loss_breakdown['num_targets'] = targets.shape[0] if hasattr(targets, 'shape') and len(targets.shape) > 0 else 0
        
        return total_loss, loss_breakdown
    
    def _get_empty_metrics(self, targets: torch.Tensor) -> Dict[str, Any]:
        """Get empty metrics dictionary"""
        return {
            'val_loss': 0.0,
            'val_map50': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0,
            'num_targets': targets.shape[0] if hasattr(targets, 'shape') and len(targets.shape) > 0 else 0
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