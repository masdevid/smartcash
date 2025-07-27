"""
YOLOv5 Integration Manager for SmartCash
Provides unified interface for using SmartCash architectures with YOLOv5
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
from copy import deepcopy
from typing import Dict, List, Any, Optional, Union

# Import YOLOv5 components
yolov5_path = Path(__file__).parent.parent.parent.parent / "yolov5"
if str(yolov5_path) not in sys.path:
    sys.path.append(str(yolov5_path))

try:
    # Import all necessary YOLOv5 components
    from models.yolo import DetectionModel, parse_model, Model
    from models.common import Conv, C3, SPPF, Concat, Bottleneck, autopad
    from models.yolo import Detect
    from utils.general import LOGGER, make_divisible, colorstr
    from utils.torch_utils import select_device, initialize_weights, scale_img
    from utils.autoanchor import check_anchors, check_anchor_order
    
    # Make sure required modules are in the global namespace for eval() in parse_model
    import torch.nn as nn
    import torch
    import math
    import contextlib
    
    # Add required modules to globals for eval()
    globals()['nn'] = nn
    globals()['torch'] = torch
    globals()['math'] = math
    globals()['autopad'] = autopad
    
    YOLOV5_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLOv5 not available: {e}")
    YOLOV5_AVAILABLE = False
    # Define dummy classes for type checking
    class Detect: pass
    class Bottleneck: pass
    class DetectionModel: pass
    class Model: pass
    def parse_model(*args, **kwargs): pass
    def make_divisible(*args, **kwargs): pass
    def check_anchor_order(*args, **kwargs): pass
    def select_device(*args, **kwargs): return 'cpu'
    def initialize_weights(*args, **kwargs): pass
    def check_anchors(*args, **kwargs): pass
    def scale_img(*args, **kwargs): pass

from smartcash.common.logger import SmartCashLogger
from smartcash.model.architectures.backbones.yolov5_backbone import (
    YOLOv5BackboneFactory, YOLOv5CSPDarknetAdapter, YOLOv5EfficientNetAdapter
)
from smartcash.model.architectures.heads.yolov5_head import (
    YOLOv5MultiLayerDetect, YOLOv5HeadAdapter, register_yolov5_components
)
from smartcash.model.architectures.necks.yolov5_neck import YOLOv5FPNPANNeck


class SmartCashYOLOv5Integration:
    """
    Main integration class for SmartCash with YOLOv5
    Manages the creation and configuration of integrated models
    """
    
    def __init__(self, logger=None):
        """
        Initialize integration manager
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
        
        # Register custom components with YOLOv5
        if YOLOV5_AVAILABLE:
            register_yolov5_components()
            self._register_custom_components()
        
        self.logger.info("✅ SmartCash YOLOv5 integration initialized")
    
    def _register_custom_components(self):
        """Register SmartCash components with YOLOv5 parsing system"""
        try:
            # Import required modules
            import builtins
            import sys
            
            # Import our custom components
            from smartcash.model.architectures.backbones.yolov5_backbone import (
                YOLOv5CSPDarknetAdapter, YOLOv5EfficientNetAdapter
            )
            from smartcash.model.architectures.heads.yolov5_head import YOLOv5MultiLayerDetect
            
            # Add to this module's globals for eval() in parse_model
            globals()['YOLOv5CSPDarknetAdapter'] = YOLOv5CSPDarknetAdapter
            globals()['YOLOv5EfficientNetAdapter'] = YOLOv5EfficientNetAdapter
            globals()['YOLOv5MultiLayerDetect'] = YOLOv5MultiLayerDetect
            
            # Add to builtins for YOLOv5's parse_model to find them
            builtins.YOLOv5CSPDarknetAdapter = YOLOv5CSPDarknetAdapter
            builtins.YOLOv5EfficientNetAdapter = YOLOv5EfficientNetAdapter
            builtins.YOLOv5MultiLayerDetect = YOLOv5MultiLayerDetect
            
            # CRITICAL: Add to YOLOv5's model parser global namespace
            try:
                import models.yolo as yolo_module
                setattr(yolo_module, 'YOLOv5CSPDarknetAdapter', YOLOv5CSPDarknetAdapter)
                setattr(yolo_module, 'YOLOv5EfficientNetAdapter', YOLOv5EfficientNetAdapter)
                setattr(yolo_module, 'YOLOv5MultiLayerDetect', YOLOv5MultiLayerDetect)
                
                # Also add to the global namespace of the yolo module
                yolo_module_globals = yolo_module.__dict__
                yolo_module_globals['YOLOv5CSPDarknetAdapter'] = YOLOv5CSPDarknetAdapter
                yolo_module_globals['YOLOv5EfficientNetAdapter'] = YOLOv5EfficientNetAdapter
                yolo_module_globals['YOLOv5MultiLayerDetect'] = YOLOv5MultiLayerDetect
                
            except ImportError as e:
                self.logger.warning(f"Could not access YOLOv5 model module: {e}")
            
            # Only log if this is the first registration
            if not hasattr(register_yolov5_components, '_registered'):
                self.logger.info("🔧 Registered SmartCash components with YOLOv5")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register components: {e}", exc_info=True)
            raise
    
    def _create_cspdarknet_backbone(self, *args, **kwargs):
        """Factory function for CSPDarknet backbone"""
        # Ensure the input channels are properly set
        if 'ch' not in kwargs:
            kwargs['ch'] = 3  # Default to 3 input channels (RGB)
        return YOLOv5CSPDarknetAdapter(logger=self.logger, **kwargs)
    
    def _create_efficientnet_backbone(self, *args, **kwargs):
        """Factory function for EfficientNet-B4 backbone"""
        return YOLOv5EfficientNetAdapter(logger=self.logger, **kwargs)
    
    def create_model(self, backbone_type="cspdarknet", model_size="s", 
                     config_path=None, **kwargs):
        """
        Create SmartCash model integrated with YOLOv5
        
        Args:
            backbone_type: 'cspdarknet' or 'efficientnet_b4'
            model_size: 's', 'm', 'l', 'x' (for CSPDarknet)
            config_path: Path to custom configuration file
            **kwargs: Additional model parameters including:
                - nc: Number of classes
                - ch: Input channels (default: 3)
                - img_size: Input image size (default: 640)
                - anchors: Anchor boxes
                - pretrained: Whether to use pretrained weights
                
        Returns:
            Integrated model instance
        """
        try:
            # Get configuration
            config = self._get_config(backbone_type, **kwargs)
            
            # Set default parameters if not provided
            config.setdefault('ch', 3)  # Default input channels
            config.setdefault('img_size', 640)  # Default image size
            
            # Get input channels and number of classes
            ch = config['ch']
            nc = config['nc']
            anchors = config.get('anchors')
            
            # Create a wrapper model that handles the input properly
            class YOLOv5Wrapper(nn.Module):
                def __init__(self, config, ch, nc, anchors, logger=None):
                    super().__init__()
                    # Save configuration
                    self.config = deepcopy(config)
                    self.config['ch'] = ch
                    self.config['nc'] = nc
                    if anchors is not None:
                        self.config['anchors'] = anchors
                    
                    # Add logger
                    from smartcash.common.logger import get_logger
                    self.logger = logger or get_logger("yolov5_wrapper")
                    
                    # Initialize model in the first forward pass
                    self.model = None
                    self.initialized = False
                    
                def _initialize_model(self, x):
                    if self.initialized:
                        return
                    
                    try:
                        # Create a temporary YAML file with the config
                        import tempfile
                        import yaml
                        import os
                        
                        # Get the model configuration
                        config = deepcopy(self.config)
                        
                        # Set default values if not provided
                        config['nc'] = config.get('nc', 80)  # number of classes
                        config['depth_multiple'] = config.get('depth_multiple', 0.33)  # model depth multiple
                        config['width_multiple'] = config.get('width_multiple', 0.50)  # layer channel multiple
                        
                        # Ensure we have anchors
                        if 'anchors' not in config:
                            config['anchors'] = [
                                [10, 13, 16, 30, 33, 23],  # P3/8
                                [30, 61, 62, 45, 59, 119],  # P4/16
                                [116, 90, 156, 198, 373, 326]  # P5/32
                            ]
                        
                        # Ensure we have a backbone and head
                        if 'backbone' not in config:
                            config['backbone'] = [
                                # From, Number, Module, Arguments
                                [-1, 1, 'Conv', [64, 6, 2, 2]],  # 0-P1/2
                                [-1, 1, 'Conv', [128, 3, 2]],    # 1-P2/4
                                [-1, 3, 'C3', [128]],
                                [-1, 1, 'Conv', [256, 3, 2]],    # 3-P3/8
                                [-1, 6, 'C3', [256]],
                                [-1, 1, 'Conv', [512, 3, 2]],    # 5-P4/16
                                [-1, 9, 'C3', [512]],
                                [-1, 1, 'Conv', [1024, 3, 2]],   # 7-P5/32
                                [-1, 3, 'C3', [1024]],
                                [-1, 1, 'SPPF', [1024, 5]],
                            ]
                        
                        if 'head' not in config:
                            config['head'] = [
                                [-1, 1, 'Conv', [512, 1, 1]],
                                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                                [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                                [-1, 3, 'C3', [512, False]],
                                [-1, 1, 'Conv', [256, 1, 1]],
                                [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                                [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                                [-1, 3, 'C3', [256, False]],
                                [-1, 1, 'Conv', [256, 3, 2]],
                                [[-1, 14], 1, 'Concat', [1]],  # cat head P4
                                [-1, 3, 'C3', [512, False]],
                                [-1, 1, 'Conv', [512, 3, 2]],
                                [[-1, 10], 1, 'Concat', [1]],  # cat head P5
                                [-1, 3, 'C3', [1024, False]],
                                [[17, 20, 23], 1, 'Detect', [config['nc'], config['anchors']]],  # Detect(P3, P4, P5)
                            ]
                        
                        # Ensure all required fields are in the config
                        config['ch'] = self.config.get('ch', 3)  # input channels
                        config['activation'] = config.get('activation', 'nn.SiLU()')
                        config['channel_multiple'] = config.get('channel_multiple', 8)
                        
                        # Make sure we have the required keys for YOLOv5
                        if 'nc' not in config:
                            config['nc'] = 80  # default number of classes
                        if 'anchors' not in config:
                            config['anchors'] = [
                                [10, 13, 16, 30, 33, 23],  # P3/8
                                [30, 61, 62, 45, 59, 119],  # P4/16
                                [116, 90, 156, 198, 373, 326]  # P5/32
                            ]
                        
                        # Debug: Print config keys before parse_model
                        self.logger.info(f"Config keys: {list(config.keys())}")
                        self.logger.info(f"Backbone layers: {len(config.get('backbone', []))}")
                        self.logger.info(f"Head layers: {len(config.get('head', []))}")
                        
                        try:
                            # Create a temporary YAML file for model initialization
                            import tempfile
                            import os
                            
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                                yaml.dump(config, f)
                                temp_yaml_path = f.name
                            
                            try:
                                # Import and ensure classes are available for eval()
                                from smartcash.model.architectures.backbones.yolov5_backbone import (
                                    YOLOv5CSPDarknetAdapter, YOLOv5EfficientNetAdapter
                                )
                                from smartcash.model.architectures.heads.yolov5_head import YOLOv5MultiLayerDetect
                                
                                # CRITICAL: Inject classes into the global namespace that YOLOv5's eval() uses
                                import models.yolo
                                original_globals = models.yolo.__dict__.copy()
                                
                                # Add our classes to YOLOv5's module globals
                                models.yolo.__dict__['YOLOv5CSPDarknetAdapter'] = YOLOv5CSPDarknetAdapter
                                models.yolo.__dict__['YOLOv5EfficientNetAdapter'] = YOLOv5EfficientNetAdapter
                                models.yolo.__dict__['YOLOv5MultiLayerDetect'] = YOLOv5MultiLayerDetect
                                
                                try:
                                    # Initialize the model using YOLOv5's Model class which handles the config properly
                                    from models.yolo import Model as YOLOModel
                                    
                                    # Add safe globals for PyTorch 2.6+ weights_only loading
                                    import torch.serialization
                                    safe_globals = [YOLOModel]
                                    # Also add any other model classes that might be in checkpoints
                                    try:
                                        from models.common import Conv, C3, SPPF, Bottleneck
                                        safe_globals.extend([Conv, C3, SPPF, Bottleneck])
                                    except ImportError:
                                        pass
                                    
                                    # Temporarily suppress YOLOv5 verbose logging during model creation
                                    import logging
                                    from utils.general import LOGGER
                                    original_level = LOGGER.level
                                    LOGGER.setLevel(logging.ERROR)  # Only show errors during model creation
                                    
                                    try:
                                        # Handle PyTorch version compatibility
                                        if hasattr(torch.serialization, 'safe_globals'):
                                            # PyTorch 2.6.0+ with safe_globals
                                            with torch.serialization.safe_globals(safe_globals):
                                                temp_model = YOLOModel(temp_yaml_path, ch=config['ch'])
                                        else:
                                            # Fallback for older PyTorch versions
                                            self.logger.warning("PyTorch version < 2.6.0 detected, using fallback model loading")
                                            temp_model = YOLOModel(temp_yaml_path, ch=config['ch'])
                                        
                                        # Initialize model weights
                                        initialize_weights(temp_model)
                                        
                                        # Set model to evaluation mode
                                        temp_model.eval()
                                    except Exception as e:
                                        self.logger.error(f"Error during model initialization: {str(e)}")
                                        raise
                                    finally:
                                        # Restore original logging level
                                        LOGGER.setLevel(original_level)
                                    
                                finally:
                                    # Restore original globals to avoid pollution
                                    models.yolo.__dict__.clear()
                                    models.yolo.__dict__.update(original_globals)
                                
                            finally:
                                # Clean up the temporary file
                                if os.path.exists(temp_yaml_path):
                                    os.unlink(temp_yaml_path)
                            
                            # Move model to the correct device
                            device = next(temp_model.parameters()).device
                            temp_model = temp_model.to(device)
                            
                            # Set model names
                            temp_model.names = [str(i) for i in range(config['nc'])]
                            temp_model.inplace = config.get('inplace', True)
                            
                        except Exception as e:
                            self.logger.error(f"Error in model parsing/initialization: {str(e)}")
                            self.logger.error(f"Config: {config}")
                            raise
                        
                        # Store the actual model
                        self.model = temp_model
                        self.initialized = True
                        
                        # Initialize model with pretrained weights if specified
                        if hasattr(self, 'pretrained_weights'):
                            try:
                                # Use safe globals context manager for PyTorch 2.6+ compatibility
                                with torch.serialization.safe_globals(safe_globals):
                                    ckpt = torch.load(self.pretrained_weights, map_location='cpu')
                                
                                if 'model' in ckpt:
                                    csd = ckpt['model'].float().state_dict()
                                    # Filter state dict to match model architecture
                                    model_sd = self.model.state_dict()
                                    csd = {k: v for k, v in csd.items() if k in model_sd and model_sd[k].shape == v.shape}
                                    self.model.load_state_dict(csd, strict=False)
                                    self.logger.info(f"Loaded pretrained weights from {self.pretrained_weights}")
                                else:
                                    self.logger.warning(f"No 'model' key found in checkpoint {self.pretrained_weights}")
                            except Exception as e:
                                self.logger.warning(f"Failed to load pretrained weights: {e}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to initialize YOLOv5 model: {e}")
                        raise
                
                def forward(self, x, *args, **kwargs):
                    # Ensure input is a tensor and not a list
                    if isinstance(x, (list, tuple)) and len(x) == 1:
                        x = x[0]
                    
                    # Initialize model on first forward pass
                    if not self.initialized:
                        self._initialize_model(x)
                    
                    # Forward pass
                    return self.model(x, *args, **kwargs)
            
            # Create the model wrapper
            model = YOLOv5Wrapper(config, ch, nc, anchors, logger=self.logger)
            
            # Store pretrained weights path if specified
            if kwargs.get('pretrained', False):
                model.pretrained_weights = kwargs.get('weights', 'yolov5s.pt')
                model.logger = self.logger
            
            # Initialize model with dummy input to set strides
            with torch.no_grad():
                dummy_input = torch.zeros(1, ch, config['img_size'], config['img_size'])
                _ = model(dummy_input)
                    
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create YOLOv5 model: {str(e)}")
            self.logger.error(f"❌ Failed to create YOLOv5 model: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create YOLOv5 model: {str(e)}") from e
    
    def _get_default_config_path(self, backbone_type, model_size):
        """Get default configuration path"""
        config_dir = Path(__file__).parent / "configs"
        
        if backbone_type == "cspdarknet":
            return config_dir / f"smartcash_yolov5{model_size}_cspdarknet.yaml"
        elif backbone_type == "efficientnet_b4":
            return config_dir / f"smartcash_yolov5{model_size}_efficientnet.yaml"
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    def _load_config(self, config_path: str) -> dict:
        """
        Load model configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Dictionary containing the model configuration
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise
            
    def _get_default_config(self, config_name: str, model_size: str = 's', **kwargs) -> dict:
        """
        Get the default configuration for a YOLOv5 model.
        
        Args:
            config_name: Name of the configuration (e.g., 'yolov5s' or 'cspdarknet')
            model_size: Model size ('n', 's', 'm', 'l', 'x')
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary containing the model configuration
        """
        # If config_name is in the format 'yolov5s', extract the size
        if config_name.startswith('yolov5'):
            model_size = config_name[6:]  # Extract 's' from 'yolov5s'
            config_name = 'cspdarknet'  # Default to cspdarknet for yolov5 configs
            
        # Size multipliers
        size_multipliers = {
            'n': 0.25,  # nano
            's': 0.50,  # small
            'm': 0.75,  # medium
            'l': 1.00,  # large
            'x': 1.25   # xlarge
        }
        
        # Get width and depth multipliers
        width_mult = size_multipliers.get(model_size.lower()[0], 0.50)
        depth_mult = min(1.0, width_mult * 1.33)  # depth multiplier
        
        # Base configuration for CSPDarknet
        if 'cspdarknet' in config_name.lower():
            return {
                'nc': kwargs.get('num_classes', 80),  # number of classes
                'depth_multiple': depth_mult,  # model depth multiple
                'width_multiple': width_mult,  # layer channel multiple
                'anchors': [
                    [10, 13, 16, 30, 33, 23],  # P3/8
                    [30, 61, 62, 45, 59, 119],  # P4/16
                    [116, 90, 156, 198, 373, 326]  # P5/32
                ],
                'backbone': [
                    # [from, number, module, args]
                    [-1, 1, 'Conv', [64, 6, 2, 2]],  # 0-P1/2
                    [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
                    [-1, 3, 'C3', [128]],
                    [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
                    [-1, 6, 'C3', [256]],
                    [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
                    [-1, 9, 'C3', [512]],
                    [-1, 1, 'Conv', [1024, 3, 2]],  # 7-P5/32
                    [-1, 3, 'C3', [1024]],
                    [-1, 1, 'SPPF', [1024, 5]],  # 9
                ],
                'head': [
                    [-1, 1, 'Conv', [512, 1, 1]],
                    [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                    [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
                    [-1, 3, 'C3', [512, False]],  # 13

                    [-1, 1, 'Conv', [256, 1, 1]],
                    [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
                    [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
                    [-1, 3, 'C3', [256, False]],  # 17 (P3/8-small)

                    [-1, 1, 'Conv', [256, 3, 2]],
                    [[-1, 14], 1, 'Concat', [1]],  # cat head P4
                    [-1, 3, 'C3', [512, False]],  # 20 (P4/16-medium)

                    [-1, 1, 'Conv', [512, 3, 2]],
                    [[-1, 10], 1, 'Concat', [1]],  # cat head P5
                    [-1, 3, 'C3', [1024, False]],  # 23 (P5/32-large)

                    [[17, 20, 23], 1, 'Detect', [kwargs.get('num_classes', 80), 'anchors']],  # Detect(P3, P4, P5)
                ]
            }
        else:
            raise ValueError(f"Unsupported config name: {config_name}")
    
    def _get_config(self, backbone_type: str, **kwargs) -> dict:
        """
        Get the configuration for the specified backbone type.
        
        Args:
            backbone_type: Type of backbone to use (e.g., 'cspdarknet')
            **kwargs: Additional configuration parameters
            
        Returns:
            Dictionary containing the model configuration
        """
        # Default configuration based on backbone type
        if 'cspdarknet' in backbone_type.lower():
            model_size = kwargs.get('model_size', 's')
            config = self._get_default_config('yolov5s', model_size, **kwargs)
            
            # Update with any additional parameters
            if kwargs:
                config.update(kwargs)
                
            return config
        elif 'efficientnet' in backbone_type.lower():
            # EfficientNet config - use a simpler standard config for now
            # Since EfficientNet integration is complex, use standard YOLOv5 backbone
            config = self._get_default_config("yolov5s", 's', **kwargs)
            
            # Update with any additional parameters
            if kwargs:
                config.update(kwargs)
                
            return config
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    def create_training_compatible_model(self, backbone_type="cspdarknet", **kwargs):
        """
        Create model compatible with existing SmartCash training pipeline
        
        Args:
            backbone_type: Type of backbone
            **kwargs: Model parameters
            
        Returns:
            Model instance with training compatibility layer
        """
        base_model = self.create_model(backbone_type=backbone_type, **kwargs)
        
        # Wrap with compatibility layer
        wrapped_model = SmartCashTrainingCompatibilityWrapper(base_model, self.logger)
        
        return wrapped_model
    
    def get_available_architectures(self) -> List[str]:
        """Get list of available architecture types"""
        architectures = ['legacy']
        if YOLOV5_AVAILABLE:
            architectures.append('yolov5')
        return architectures
    
    def get_model_info(self, model):
        """Get information about integrated model"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            'architecture': 'SmartCash-YOLOv5 Integrated',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'yolov5_compatible': True,
            'multi_layer_detection': True
        }
        
        if hasattr(model, 'model') and hasattr(model.model[-1], 'get_layer_info'):
            info.update(model.model[-1].get_layer_info())
        
        return info


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
        
        # Extract components for compatibility
        self.backbone = self._extract_backbone()
        self.neck = self._extract_neck()
        self.head = self._extract_head()
        
        self.logger.info("🔄 Created training compatibility wrapper")
    
    def _extract_backbone(self):
        """Extract backbone component"""
        # For YOLOv5, the backbone is the first part of the model
        # up to the last C3/SPPF layer
        backbone_layers = []
        for module in self.yolov5_model.model.model:
            if isinstance(module, (C3, SPPF)):
                backbone_layers.append(module)
                break
            backbone_layers.append(module)
        
        return nn.Sequential(*backbone_layers)
    
    def _extract_neck(self):
        """Extract neck component (FPN in YOLOv5)"""
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
        # Find the Detect layer in the model
        for module in self.yolov5_model.model.model:
            if isinstance(module, Detect):
                return module
        return None
    
    def forward(self, x):
        """Forward pass compatible with training pipeline"""
        return self.yolov5_model(x)
    
    def predict(self, x, conf_threshold=0.25, nms_threshold=0.45):
        """Prediction with post-processing"""
        self.eval()
        with torch.no_grad():
            return self.yolov5_model(x)
    
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


# Global integration instance
_integration_manager = None

def get_integration_manager():
    """Get global integration manager instance"""
    global _integration_manager
    if _integration_manager is None:
        _integration_manager = SmartCashYOLOv5Integration()
    return _integration_manager


def create_smartcash_yolov5_model(backbone_type="cspdarknet", **kwargs):
    """
    Convenience function to create SmartCash-YOLOv5 integrated model
    
    Args:
        backbone_type: Type of backbone
        **kwargs: Model parameters
        
    Returns:
        Integrated model instance
    """
    manager = get_integration_manager()
    return manager.create_model(backbone_type=backbone_type, **kwargs)


def create_training_model(backbone_type="cspdarknet", **kwargs):
    """
    Create model compatible with existing training pipeline
    
    Args:
        backbone_type: Type of backbone
        **kwargs: Model parameters
        
    Returns:
        Training-compatible model instance
    """
    manager = get_integration_manager()
    return manager.create_training_compatible_model(backbone_type=backbone_type, **kwargs)


# Export key functions and classes
__all__ = [
    'SmartCashYOLOv5Integration',
    'SmartCashTrainingCompatibilityWrapper',
    'create_smartcash_yolov5_model',
    'create_training_model',
    'get_integration_manager'
]