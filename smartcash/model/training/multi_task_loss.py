"""
File: smartcash/model/training/multi_task_loss.py
Description: Uncertainty-based multi-task loss implementation based on Kendall et al. (Google DeepMind)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import SmartCashLogger
from smartcash.model.training.loss_manager import YOLOLoss
from smartcash.model.training.utils.tensor_format_converter import convert_for_yolo_loss


class UncertaintyMultiTaskLoss(nn.Module):
    """
    TASK.md: Uncertainty-based multi-task loss implementation with learnable sigma parameters.
    
    Formula per TASK.md specification:
    loss_total = (1/sigma1**2) * (box_loss1 + obj_loss1 + cls_loss1) + log(sigma1) +
                 (1/sigma2**2) * (box_loss2 + obj_loss2 + cls_loss2) + log(sigma2) +
                 (1/sigma3**2) * (box_loss3 + obj_loss3 + cls_loss3) + log(sigma3)
    
    Where:
    - σ parameters are learnable using torch.nn.Parameter
    - Uses log trick to prevent NaN (store log(σ) instead of σ)
    - Each layer loss includes: box_loss + obj_loss + cls_loss
    """
    
    def __init__(self, layer_config: Dict[str, Any], loss_config: Dict[str, Any] = None,
                 logger: Optional[SmartCashLogger] = None):
        """
        Initialize uncertainty-based multi-task loss
        
        Args:
            layer_config: Configuration for each detection layer
            loss_config: Loss function configuration
            logger: Logger instance
        """
        super().__init__()
        self.logger = logger or SmartCashLogger(__name__)
        self.layer_config = layer_config
        self.loss_config = loss_config or {}
        
        # Extract layer information
        self.layer_names = list(layer_config.keys())
        self.num_layers = len(self.layer_names)
        
        # TASK.md: Initialize learnable uncertainty parameters (log sigma, not log variance)
        self.log_vars = nn.ParameterDict()
        for layer_name in self.layer_names:
            # Initialize log(σ) = 0.0, so σ = 1.0 initially (TASK.md: Use log trick to prevent NaN)
            self.log_vars[layer_name] = nn.Parameter(torch.tensor(0.0))
        
        # Initialize individual YOLO loss functions for each layer
        self.layer_losses = nn.ModuleDict()
        for layer_name, config in layer_config.items():
            num_classes = config.get('num_classes', 7)
            self.layer_losses[layer_name] = YOLOLoss(
                num_classes=num_classes,
                box_weight=self.loss_config.get('box_weight', 0.05),   # From loss.json
                obj_weight=self.loss_config.get('obj_weight', 1.0),  # From loss.json
                cls_weight=self.loss_config.get('cls_weight', 0.5),
                focal_loss=self.loss_config.get('focal_loss', False),
                label_smoothing=self.loss_config.get('label_smoothing', 0.0)
            )
        
        # Loss weighting parameters
        self.use_dynamic_weighting = self.loss_config.get('dynamic_weighting', True)
        self.min_variance = self.loss_config.get('min_variance', 1e-3)
        self.max_variance = self.loss_config.get('max_variance', 10.0)
        
        self.logger.info(f"✅ Uncertainty-based multi-task loss initialized for {self.num_layers} layers")
        for layer_name in self.layer_names:
            self.logger.info(f"   • {layer_name}: {layer_config[layer_name]['num_classes']} classes")

        # ThreadPool for parallel loss calculation
        self.loss_executor = ThreadPoolExecutor(max_workers=self.num_layers, thread_name_prefix='multi_task_loss')
    
    def forward(self, predictions: Dict[str, List[torch.Tensor]], 
                targets: Dict[str, torch.Tensor], img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute uncertainty-weighted multi-task loss
        
        Args:
            predictions: Dict of layer predictions {layer_name: [pred_p3, pred_p4, pred_p5]}
            targets: Dict of layer targets {layer_name: targets_tensor}
            img_size: Input image size
            
        Returns:
            total_loss: Weighted total loss
            loss_breakdown: Detailed loss components
        """
        self.logger.debug(f"Multi-task loss forward: predictions={list(predictions.keys())}, targets={list(targets.keys())}")
        # Determine device from predictions or fallback to CPU
        device = torch.device('cpu')  # Default device
        if predictions:
            for pred_list in predictions.values():
                if pred_list is not None and isinstance(pred_list, list) and len(pred_list) > 0:
                    if isinstance(pred_list[0], torch.Tensor):
                        device = pred_list[0].device
                        break
        
        # Initialize containers for losses and metrics
        layer_losses: Dict[str, torch.Tensor] = {}
        loss_breakdown: Dict[str, Any] = {}
        uncertainties: Dict[str, float] = {}
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Parallelize loss calculation for each layer
        future_to_layer = {
            self.loss_executor.submit(
                self._compute_layer_loss,
                layer_name,
                predictions.get(layer_name),
                targets.get(layer_name),
                img_size,
                device
            ): layer_name
            for layer_name in self.layer_names
        }

        for future in as_completed(future_to_layer):
            layer_name = future_to_layer[future]
            try:
                l_loss, l_breakdown = future.result()
                layer_losses[layer_name] = l_loss
                loss_breakdown.update(l_breakdown)
            except Exception as e:
                self.logger.error(f"Error computing loss for layer {layer_name} in parallel: {e}")
                layer_losses[layer_name] = torch.tensor(0.0, device=device, requires_grad=True)
                loss_breakdown[f"{layer_name}_total_loss"] = torch.tensor(0.0, device=device)

        # Apply uncertainty-based weighting
        if self.use_dynamic_weighting:
            weighted_losses = {}
            regularization_terms = {}
            
            for layer_name in self.layer_names:
                if layer_name in layer_losses:
                    # Get learnable log-sigma parameter (TASK.md: Use log trick to prevent NaN)
                    log_sigma = self.log_vars[layer_name]
                    
                    # Clamp log_sigma to reasonable range to prevent extreme values
                    log_sigma_clamped = log_sigma.clamp(-5.0, 5.0)  # sigma between e^-5 and e^5
                    sigma_squared = torch.exp(2 * log_sigma_clamped)  # σ² = exp(2 * log(σ))
                    
                    # TASK.md: loss_total = (1/sigma**2) * (box_loss + obj_loss + cls_loss) + log(sigma)
                    layer_loss = layer_losses[layer_name]
                    precision = 1.0 / sigma_squared  # 1/σ²
                    weighted_loss = precision * layer_loss
                    regularization = log_sigma_clamped  # log(σ)
                    
                    weighted_losses[layer_name] = weighted_loss
                    regularization_terms[layer_name] = regularization
                    uncertainties[layer_name] = sigma_squared.item()
                    
                    # Add to total loss
                    total_loss = total_loss + weighted_loss + regularization
                    
                    # Store weighted components
                    loss_breakdown[f"{layer_name}_weighted_loss"] = weighted_loss
                    loss_breakdown[f"{layer_name}_regularization"] = regularization
                    loss_breakdown[f"{layer_name}_uncertainty"] = sigma_squared
        else:
            # Use equal weighting
            for layer_name, layer_loss in layer_losses.items():
                total_loss = total_loss + layer_loss
                uncertainties[layer_name] = 1.0
        
        # Store summary metrics
        loss_breakdown.update({
            'total_loss': total_loss,
            'num_layers': len([l for l in layer_losses.values() if l.item() > 0]),
            'layer_losses': layer_losses,
            'uncertainties': uncertainties,
            'use_uncertainty_weighting': self.use_dynamic_weighting
        })
        
        return total_loss, loss_breakdown

    def _compute_layer_loss(self, layer_name: str, layer_preds: Optional[List[torch.Tensor]], 
                            layer_targets: Optional[torch.Tensor], img_size: int, device: torch.device) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute loss for a single layer."""
        loss_breakdown = {}
        if layer_preds and layer_targets is not None:
            # Ensure layer_preds is a list of tensors on the right device
            if not isinstance(layer_preds, (list, tuple)):
                layer_preds = [layer_preds]
            
            processed_preds = []
            for p in layer_preds:
                if isinstance(p, torch.Tensor):
                    processed_preds.append(p.to(device))
                elif isinstance(p, (list, tuple)):
                    nested_tensors = []
                    for nested_p in p:
                        if isinstance(nested_p, torch.Tensor):
                            nested_tensors.append(nested_p.to(device))
                        else:
                            self.logger.warning(f"Skipping non-tensor element in nested predictions: {type(nested_p)}")
                            continue
                    if nested_tensors:
                        processed_preds.extend(nested_tensors)
                else:
                    self.logger.warning(f"Skipping non-tensor prediction element: {type(p)}")
                    continue
            
            layer_preds = processed_preds
            
            if not layer_preds:
                self.logger.warning(f"No valid tensor predictions found for layer {layer_name}")
                return torch.tensor(0.0, device=device, requires_grad=True), {f"{layer_name}_total_loss": torch.tensor(0.0, device=device)}

            if isinstance(layer_targets, (list, tuple)) and len(layer_targets) > 0:
                layer_targets = layer_targets[0]
            if isinstance(layer_targets, torch.Tensor):
                layer_targets = layer_targets.to(device)

            has_valid_targets = False
            if layer_targets is not None:
                if torch.is_tensor(layer_targets):
                    has_valid_targets = layer_targets.numel() > 0
                elif isinstance(layer_targets, (list, tuple)):
                    has_valid_targets = len(layer_targets) > 0
                else:
                    has_valid_targets = True

            if has_valid_targets:
                loss_fn = self.layer_losses[layer_name]
                try:
                    converted_preds = convert_for_yolo_loss(layer_preds, img_size)
                    layer_loss, layer_components = loss_fn(converted_preds, layer_targets, img_size)
                    for comp_name, comp_value in layer_components.items():
                        loss_breakdown[f"{layer_name}_{comp_name}"] = comp_value
                    return layer_loss, loss_breakdown
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error computing loss for layer {layer_name}: {str(e)}")
                    self.logger.error(f"Traceback: {traceback.format_exc()}")
                    return torch.tensor(0.0, device=device, requires_grad=True), {f"{layer_name}_total_loss": torch.tensor(0.0, device=device)}
        
        return torch.tensor(0.0, device=device, requires_grad=True), {f"{layer_name}_total_loss": torch.tensor(0.0, device=device)}

    def get_uncertainty_weights(self) -> Dict[str, float]:
        """Get current uncertainty weights for each layer (TASK.md: weights = 1/σ²)"""
        weights = {}
        for layer_name in self.layer_names:
            log_sigma = self.log_vars[layer_name]
            log_sigma_clamped = log_sigma.clamp(-5.0, 5.0)
            sigma_squared = torch.exp(2 * log_sigma_clamped)
            weights[layer_name] = (1.0 / sigma_squared).item()
        return weights
    
    def get_uncertainty_values(self) -> Dict[str, float]:
        """Get current uncertainty (σ²) values for each layer"""
        uncertainties = {}
        for layer_name in self.layer_names:
            log_sigma = self.log_vars[layer_name]
            log_sigma_clamped = log_sigma.clamp(-5.0, 5.0)
            sigma_squared = torch.exp(2 * log_sigma_clamped)
            uncertainties[layer_name] = sigma_squared.item()
        return uncertainties
    
    def update_loss_config(self, new_config: Dict[str, Any]) -> None:
        """Update loss configuration for all layers"""
        self.loss_config.update(new_config)
        
        # Update individual layer loss functions
        for layer_name in self.layer_names:
            if hasattr(self.layer_losses[layer_name], 'box_weight'):
                self.layer_losses[layer_name].box_weight = new_config.get('box_weight', 0.5)
            if hasattr(self.layer_losses[layer_name], 'obj_weight'):
                self.layer_losses[layer_name].obj_weight = new_config.get('obj_weight', 1.5)
            if hasattr(self.layer_losses[layer_name], 'cls_weight'):
                self.layer_losses[layer_name].cls_weight = new_config.get('cls_weight', 0.5)
    
    def get_loss_summary(self, loss_dict: Dict[str, Any]) -> str:
        """Get formatted summary of multi-task loss"""
        total = loss_dict.get('total_loss', 0)
        num_layers = loss_dict.get('num_layers', 0)
        uncertainties = loss_dict.get('uncertainties', {})
        
        summary = f"Total: {total:.4f} | Layers: {num_layers}"
        
        # Add uncertainty information
        if uncertainties:
            unc_str = " | Uncertainties: "
            unc_parts = [f"{name}: {val:.3f}" for name, val in uncertainties.items()]
            summary += unc_str + ", ".join(unc_parts)
        
        return summary

    def cleanup(self):
        """Shutdown the thread pool executor gracefully."""
        self.loss_executor.shutdown(wait=True)

    def __del__(self):
        self.cleanup()


class AdaptiveMultiTaskLoss(UncertaintyMultiTaskLoss):
    """
    Extended multi-task loss with adaptive weighting based on layer performance
    """
    
    def __init__(self, layer_config: Dict[str, Any], loss_config: Dict[str, Any] = None,
                 performance_window: int = 100, logger: Optional[SmartCashLogger] = None):
        super().__init__(layer_config, loss_config, logger)
        
        self.performance_window = performance_window
        self.loss_history = {layer: [] for layer in self.layer_names}
        self.step_count = 0
        
        # Adaptive weighting parameters
        self.adaptation_rate = loss_config.get('adaptation_rate', 0.01) if loss_config else 0.01
        self.performance_threshold = loss_config.get('performance_threshold', 0.1) if loss_config else 0.1
    
    def forward(self, predictions: Dict[str, List[torch.Tensor]], 
                targets: Dict[str, torch.Tensor], img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass with adaptive weighting"""
        # Get base uncertainty-weighted loss
        total_loss, loss_breakdown = super().forward(predictions, targets, img_size)
        
        # Update loss history for adaptation
        self._update_loss_history(loss_breakdown)
        
        # Apply adaptive adjustments if enough history
        if self.step_count > self.performance_window:
            adaptation_factors = self._compute_adaptation_factors()
            total_loss = self._apply_adaptive_weighting(total_loss, adaptation_factors)
            loss_breakdown['adaptation_factors'] = adaptation_factors
        
        self.step_count += 1
        return total_loss, loss_breakdown
    
    def _update_loss_history(self, loss_breakdown: Dict[str, Any]) -> None:
        """Update loss history for each layer"""
        for layer_name in self.layer_names:
            layer_loss_key = f"{layer_name}_total_loss"
            if layer_loss_key in loss_breakdown:
                loss_value = loss_breakdown[layer_loss_key].item()
                self.loss_history[layer_name].append(loss_value)
                
                # Keep only recent history
                if len(self.loss_history[layer_name]) > self.performance_window:
                    self.loss_history[layer_name] = self.loss_history[layer_name][-self.performance_window:]
    
    def _compute_adaptation_factors(self) -> Dict[str, float]:
        """Compute adaptive weighting factors based on layer performance"""
        factors = {}
        
        for layer_name in self.layer_names:
            if len(self.loss_history[layer_name]) >= self.performance_window:
                recent_losses = self.loss_history[layer_name][-self.performance_window//2:]
                early_losses = self.loss_history[layer_name][:self.performance_window//2]
                
                recent_avg = sum(recent_losses) / len(recent_losses) if recent_losses else 1.0
                early_avg = sum(early_losses) / len(early_losses) if early_losses else 1.0
                
                # Compute improvement rate
                improvement = (early_avg - recent_avg) / (early_avg + 1e-8)
                
                # Layers with slower improvement get higher weights
                if improvement < self.performance_threshold:
                    factors[layer_name] = 1.0 + self.adaptation_rate
                else:
                    factors[layer_name] = max(0.5, 1.0 - self.adaptation_rate)
            else:
                factors[layer_name] = 1.0
        
        return factors
    
    def _apply_adaptive_weighting(self, total_loss: torch.Tensor, 
                                factors: Dict[str, float]) -> torch.Tensor:
        """Apply adaptive weighting factors"""
        # This is a simplified approach - in practice, you'd want to
        # decompose the total loss back to individual components
        avg_factor = sum(factors.values()) / len(factors)
        return total_loss * avg_factor


# Factory functions
def create_uncertainty_loss(layer_config: Dict[str, Any], loss_config: Dict[str, Any] = None,
                           **kwargs) -> UncertaintyMultiTaskLoss:
    """Factory function to create uncertainty-based multi-task loss"""
    return UncertaintyMultiTaskLoss(layer_config, loss_config, **kwargs)


def create_adaptive_loss(layer_config: Dict[str, Any], loss_config: Dict[str, Any] = None,
                        **kwargs) -> AdaptiveMultiTaskLoss:
    """Factory function to create adaptive multi-task loss"""
    return AdaptiveMultiTaskLoss(layer_config, loss_config, **kwargs)


def create_banknote_multi_task_loss(use_adaptive: bool = False, **kwargs) -> UncertaintyMultiTaskLoss:
    """Create multi-task loss specifically for banknote detection"""
    layer_config = {
        'layer_1': {
            'description': 'Full banknote detection',
            'num_classes': 7
        },
        'layer_2': {
            'description': 'Nominal-defining features', 
            'num_classes': 7
        },
        'layer_3': {
            'description': 'Common features',
            'num_classes': 3
        }
    }
    
    if use_adaptive:
        return AdaptiveMultiTaskLoss(layer_config, **kwargs)
    else:
        return UncertaintyMultiTaskLoss(layer_config, **kwargs)