"""
Training Compatibility Wrapper for YOLOv5 Integration
Makes YOLOv5 models compatible with existing SmartCash training pipeline
"""

import torch
import torch.nn as nn
from smartcash.common.logger import SmartCashLogger

# Import YOLOv5 components
try:
    from pathlib import Path
    import sys
    yolov5_path = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
    if str(yolov5_path) not in sys.path:
        sys.path.append(str(yolov5_path))
    
    from models.common import C3, SPPF
    from models.yolo import Detect
    YOLOV5_AVAILABLE = True
except ImportError:
    # Define dummy classes for type checking
    class C3: pass
    class SPPF: pass
    class Detect: pass
    YOLOV5_AVAILABLE = False


class SmartCashTrainingCompatibilityWrapper(nn.Module):
    """
    Wrapper to make YOLOv5 integrated models compatible with existing training pipeline
    """
    
    def __init__(self, yolov5_model, logger=None):
        """
        Initialize compatibility wrapper
        
        Args:
            yolov5_model: YOLOv5 integrated model
            logger: Logger instance
        """
        super().__init__()
        self.yolov5_model = yolov5_model
        self.logger = logger or SmartCashLogger(__name__)
        
        # Initialize phase tracking
        self.current_phase = 1  # Default to phase 1
        
        # Store multi-layer configuration if available
        self.multi_layer_config = getattr(yolov5_model, 'multi_layer_config', None)
        
        # Extract components for compatibility
        self.backbone = self._extract_backbone()
        self.neck = self._extract_neck()
        self.head = self._extract_head()
        
        self.logger.info("🔄 Created training compatibility wrapper")
        
        # Propagate phase to all model components
        self._propagate_phase_to_components()
    
    def __setattr__(self, name, value):
        """Override setattr to propagate phase changes"""
        super().__setattr__(name, value)
        if name == 'current_phase' and hasattr(self, 'yolov5_model'):
            self._propagate_phase_to_components()
    
    def _extract_backbone(self):
        """Extract backbone component"""
        if not YOLOV5_AVAILABLE:
            return None
            
        # For YOLOv5, the backbone is the first part of the model
        # up to the last C3/SPPF layer
        backbone_layers = []
        for module in self.yolov5_model.model.model:
            if isinstance(module, (C3, SPPF)):
                backbone_layers.append(module)
                break
            backbone_layers.append(module)
        
        return nn.Sequential(*backbone_layers) if backbone_layers else None
    
    def _extract_neck(self):
        """Extract neck component (FPN in YOLOv5)"""
        if not YOLOV5_AVAILABLE:
            return None
            
        # The neck is the FPN part of YOLOv5
        neck_layers = []
        in_backbone = True
        
        for module in self.yolov5_model.model.model:
            # Skip backbone layers
            if in_backbone:
                if isinstance(module, (C3, SPPF)):
                    in_backbone = False
                continue
                
            # Stop at detection head
            if isinstance(module, Detect):
                break
                
            neck_layers.append(module)
            
        return nn.Sequential(*neck_layers) if neck_layers else None
    
    def _extract_head(self):
        """Extract detection head"""
        if not YOLOV5_AVAILABLE:
            return None
            
        # Find the Detect layer in the model
        for module in self.yolov5_model.model.model:
            if isinstance(module, Detect):
                return module
        return None
    
    def forward(self, x):
        """Forward pass compatible with training pipeline"""
        return self.yolov5_model(x)
    
    def predict(self, x, conf_threshold=0.25, nms_threshold=0.45):
        """
        Prediction with post-processing
        
        Args:
            x: Input tensor
            conf_threshold: Confidence threshold for predictions
            nms_threshold: NMS threshold for post-processing
            
        Returns:
            Model predictions
        """
        self.eval()
        with torch.no_grad():
            # TODO: Implement proper confidence and NMS thresholding
            # For now, return raw model output
            _ = conf_threshold, nms_threshold  # Acknowledge parameters for future use
            return self.yolov5_model(x)
    
    def _propagate_phase_to_components(self):
        """Propagate current_phase to all model components"""
        try:
            # Set phase on YOLOv5 model (force create the attribute)
            if hasattr(self, 'yolov5_model') and self.yolov5_model is not None:
                self.yolov5_model.current_phase = self.current_phase
            
            # Set phase on nested model components
            if hasattr(self, 'yolov5_model') and hasattr(self.yolov5_model, 'model'):
                nested_model = self.yolov5_model.model
                nested_model.current_phase = self.current_phase
                
                # Set phase on detection head if it exists (comprehensive search)
                if hasattr(nested_model, 'model') and hasattr(nested_model.model, '__iter__'):
                    try:
                        # Search through all layers for detection heads
                        for layer in nested_model.model:
                            # Set phase on any layer that might be a detection head
                            layer.current_phase = self.current_phase
                            
                            # Also check for multi-layer heads specifically
                            if hasattr(layer, 'heads') or 'MultiLayer' in str(type(layer)):
                                layer.current_phase = self.current_phase
                                self.logger.debug(f"Set phase {self.current_phase} on detection head: {type(layer)}")
                    except Exception as layer_e:
                        self.logger.debug(f"Could not set phase on model layers: {layer_e}")
            
            # Set phase on extracted components
            if hasattr(self, 'head') and self.head:
                self.head.current_phase = self.current_phase
                self.logger.debug(f"Set phase {self.current_phase} on extracted head: {type(self.head)}")
            if hasattr(self, 'backbone') and self.backbone:
                self.backbone.current_phase = self.current_phase
            if hasattr(self, 'neck') and self.neck:
                self.neck.current_phase = self.current_phase
            
            # Additional comprehensive search for any detection heads in the model hierarchy
            self._deep_phase_propagation(self.yolov5_model)
                
        except Exception as e:
            self.logger.debug(f"Phase propagation encountered an issue: {e}")
    
    def _deep_phase_propagation(self, module, visited=None):
        """Deep search and propagation of current_phase to all nested modules"""
        if visited is None:
            visited = set()
            
        # Avoid infinite recursion
        if id(module) in visited:
            return
        visited.add(id(module))
        
        try:
            # Set phase on current module
            if hasattr(module, '__dict__'):
                module.current_phase = self.current_phase
            
            # Recursively search child modules
            if hasattr(module, '_modules') and module._modules:
                for child in module._modules.values():
                    if child is not None:
                        self._deep_phase_propagation(child, visited)
            
            # Also search other attributes that might contain modules
            if hasattr(module, '__dict__'):
                for attr_name, attr_value in module.__dict__.items():
                    if (hasattr(attr_value, '_modules') or 
                        hasattr(attr_value, 'forward') or
                        'Module' in str(type(attr_value))):
                        self._deep_phase_propagation(attr_value, visited)
                        
        except Exception as e:
            self.logger.debug(f"Deep phase propagation failed on {type(module)}: {e}")
    
    def get_model_summary(self):
        """Get model summary compatible with existing interface"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': 'SmartCash-YOLOv5 Integrated',
            'backbone': 'YOLOv5 Compatible',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }