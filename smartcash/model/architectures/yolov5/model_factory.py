"""
Model Factory for YOLOv5 Integration
Handles model creation, initialization, and wrapper logic
"""

import torch
import torch.nn as nn
import tempfile
import os
import yaml
import logging
import sys
from pathlib import Path
from copy import deepcopy
from smartcash.common.logger import SmartCashLogger

# Import YOLOv5 components
try:
    yolov5_path = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
    if str(yolov5_path) not in sys.path:
        sys.path.append(str(yolov5_path))
    
    from models.yolo import Model as YOLOModel
    from utils.torch_utils import initialize_weights
    from utils.general import LOGGER
    import models.yolo
    
    YOLOV5_AVAILABLE = True
except ImportError as e:
    print(f"Warning: YOLOv5 not available: {e}")
    YOLOV5_AVAILABLE = False
    class YOLOModel: pass
    def initialize_weights(*args, **kwargs): pass
    LOGGER = None
    models = None

from smartcash.model.architectures.backbones.yolov5_backbone import YOLOv5Backbone
from smartcash.model.architectures.heads.yolov5_head import YOLOv5Head


class YOLOv5ModelFactory:
    """
    Factory for creating YOLOv5-based models
    """
    
    def __init__(self, logger=None):
        """
        Initialize model factory
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
    
    def create_yolov5_wrapper(self, config, ch, nc, anchors):
        """
        Create YOLOv5 wrapper model
        
        Args:
            config: Model configuration
            ch: Input channels
            nc: Number of classes
            anchors: Anchor boxes
            
        Returns:
            YOLOv5 wrapper model
        """
        return YOLOv5Wrapper(config, ch, nc, anchors, logger=self.logger)
        
    def create_model(self, backbone: str = 'yolov5s', num_classes: int = 1, 
                    img_size: int = 640, pretrained: bool = True, **kwargs):
        """
        Create a YOLOv5 model with the specified configuration.
        
        Args:
            backbone: Backbone type (e.g., 'yolov5s')
            num_classes: Number of output classes
            img_size: Input image size
            pretrained: Whether to load pretrained weights
            **kwargs: Additional model configuration
            
        Returns:
            YOLOv5 model instance
        """
        try:
            # Default anchor boxes for YOLOv5 (taken from YOLOv5s)
            anchors = [
                [10,13, 16,30, 33,23],  # P3/8
                [30,61, 62,45, 59,119],  # P4/16
                [116,90, 156,198, 373,326]  # P5/32
            ]
            
            # Create a basic YOLOv5 model configuration
            config = {
                'nc': num_classes,
                'depth_multiple': 0.33,  # model depth multiple
                'width_multiple': 0.50,  # layer channel multiple
                'anchors': anchors,
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

                    [[17, 20, 23], 1, 'Detect', [num_classes, anchors]],  # Detect(P3, P4, P5)
                ],
                'pretrained': pretrained,
                'img_size': img_size,
                **kwargs
            }
            
            # Create a wrapper for the model
            model = YOLOv5Wrapper(
                config=config,
                ch=3,  # RGB input channels
                nc=num_classes,
                anchors=anchors,
                logger=self.logger
            )
            
            # Initialize the model with a dummy input
            dummy_input = torch.randn(1, 3, img_size, img_size)
            model(dummy_input)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create YOLOv5 model: {e}", exc_info=True)
            raise
    
    def register_custom_components(self):
        """Register SmartCash components with YOLOv5 parsing system"""
        if not YOLOV5_AVAILABLE:
            return
            
        try:
            # Import required modules
            import builtins
            import sys
            
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
                setattr(models.yolo, 'YOLOv5CSPDarknetAdapter', YOLOv5CSPDarknetAdapter)
                setattr(models.yolo, 'YOLOv5EfficientNetAdapter', YOLOv5EfficientNetAdapter)
                setattr(models.yolo, 'YOLOv5MultiLayerDetect', YOLOv5MultiLayerDetect)
                
                # Also add to the global namespace of the yolo module
                yolo_module_globals = models.yolo.__dict__
                yolo_module_globals['YOLOv5CSPDarknetAdapter'] = YOLOv5CSPDarknetAdapter
                yolo_module_globals['YOLOv5EfficientNetAdapter'] = YOLOv5EfficientNetAdapter
                yolo_module_globals['YOLOv5MultiLayerDetect'] = YOLOv5MultiLayerDetect
                
            except ImportError as e:
                self.logger.warning(f"Could not access YOLOv5 model module: {e}")
            
            self.logger.debug("🔧 Registered SmartCash components with YOLOv5")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to register components: {e}", exc_info=True)
            raise


class YOLOv5Wrapper(nn.Module):
    """
    Wrapper model that handles YOLOv5 initialization and input processing
    """
    
    def __init__(self, config, ch, nc, anchors, logger=None):
        """
        Initialize YOLOv5 wrapper
        
        Args:
            config: Model configuration
            ch: Input channels
            nc: Number of classes
            anchors: Anchor boxes
            logger: Logger instance
        """
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
        """
        Initialize the YOLOv5 model
        
        Args:
            x: Input tensor for shape inference
        """
        if self.initialized or not YOLOV5_AVAILABLE:
            return
        
        try:
            # Get the model configuration
            config = deepcopy(self.config)
            
            # Log configuration info
            self.logger.info(f"Config keys: {list(config.keys())}")
            self.logger.info(f"Backbone layers: {len(config.get('backbone', []))}")
            self.logger.info(f"Head layers: {len(config.get('head', []))}")
            
            temp_model = self._create_yolo_model(config)
            
            # Move model to the correct device
            device = next(temp_model.parameters()).device
            temp_model = temp_model.to(device)
            
            # Set model names
            temp_model.names = [str(i) for i in range(config['nc'])]
            temp_model.inplace = config.get('inplace', True)
            
            # Store the actual model
            self.model = temp_model
            
            # Store multi-layer configuration on the model for later access
            if 'multi_layer_nc' in config:
                self.model.multi_layer_config = config['multi_layer_nc']
                self.logger.info(f"Stored multi-layer config on model: {config['multi_layer_nc']}")
            
            self.initialized = True
            
            # Initialize model with pretrained weights if specified
            self._load_pretrained_weights()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLOv5 model: {e}")
            raise
    
    def _create_yolo_model(self, config):
        """
        Create YOLOv5 model from configuration
        
        Args:
            config: Model configuration
            
        Returns:
            YOLOv5 model instance
        """
        try:
            # Create a temporary YAML file for model initialization
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config, f)
                temp_yaml_path = f.name
            
            try:
                # CRITICAL: Inject classes into the global namespace that YOLOv5's eval() uses
                original_globals = models.yolo.__dict__.copy()
                
                # Add our classes to YOLOv5's module globals
                models.yolo.__dict__['YOLOv5Backbone'] = YOLOv5Backbone
                models.yolo.__dict__['YOLOv5Head'] = YOLOv5Head
                
                try:
                    # Initialize the model using YOLOv5's Model class which handles the config properly
                    
                    # Disable PyTorch 2.6+ weights_only loading for YOLOv5 compatibility
                    # YOLOv5 checkpoints contain complex model structures that are safe to load
                    original_load = torch.load
                    
                    def safe_yolo_load(f, map_location=None, pickle_module=None, **kwargs):
                        # Force weights_only=False for YOLOv5 model loading
                        return original_load(f, map_location=map_location, 
                                           pickle_module=pickle_module, weights_only=False)
                    
                    # Temporarily replace torch.load during model creation
                    torch.load = safe_yolo_load
                    
                    # Temporarily suppress YOLOv5 verbose logging during model creation
                    original_level = LOGGER.level if LOGGER else logging.INFO
                    if LOGGER:
                        LOGGER.setLevel(logging.ERROR)  # Only show errors during model creation
                    
                    try:
                        # Create YOLOv5 model with disabled weights_only loading
                        # Check if we should use our custom SmartCash model for multi-layer support
                        if 'multi_layer_nc' in config:
                            # Multi-layer configuration detected, but for now use standard YOLOModel
                            # Use the modern YOLOv5Backbone implementation
                            self.logger.info("Using modern YOLOv5Backbone implementation")
                            backbone = 'yolov5s'  # Default backbone, can be configured
                            num_classes = config.get('nc', 1)
                            temp_model = YOLOv5Backbone(
                                backbone=backbone,
                                num_classes=num_classes,
                                pretrained=True,
                                device='cuda' if torch.cuda.is_available() else 'cpu'
                            )
                        else:
                            # Standard single-layer configuration
                            temp_model = YOLOModel(temp_yaml_path, ch=config['ch'])
                        
                        # Initialize model weights
                        initialize_weights(temp_model)
                        
                        # Set model to evaluation mode
                        temp_model.eval()
                    except Exception as e:
                        self.logger.error(f"Error during model initialization: {str(e)}")
                        raise
                    finally:
                        # Restore original torch.load and logging level
                        torch.load = original_load
                        if LOGGER:
                            LOGGER.setLevel(original_level)
                    
                finally:
                    # Restore original globals to avoid pollution
                    models.yolo.__dict__.clear()
                    models.yolo.__dict__.update(original_globals)
                
            finally:
                # Clean up the temporary file
                if os.path.exists(temp_yaml_path):
                    os.unlink(temp_yaml_path)
            
            return temp_model
            
        except Exception as e:
            self.logger.error(f"Error in model parsing/initialization: {str(e)}")
            self.logger.error(f"Config: {config}")
            raise
    
    def _load_pretrained_weights(self):
        """Load pretrained weights if specified"""
        if hasattr(self, 'pretrained_weights'):
            try:
                # Load YOLOv5 checkpoint with weights_only=False for compatibility
                ckpt = torch.load(self.pretrained_weights, map_location='cpu', weights_only=False)
                
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
    
    def forward(self, x, *args, **kwargs):
        """
        Forward pass
        
        Args:
            x: Input tensor
            *args: Additional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Model output
        """
        # Ensure input is a tensor and not a list
        if isinstance(x, (list, tuple)) and len(x) == 1:
            x = x[0]
        
        # Initialize model on first forward pass
        if not self.initialized:
            self._initialize_model(x)
        
        # Forward pass
        return self.model(x, *args, **kwargs)