"""
Configuration Management for YOLOv5 Integration
Handles loading, validation, and default configurations
"""

import yaml
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any
from smartcash.common.logger import SmartCashLogger


class YOLOv5ConfigManager:
    """
    Manages configuration for YOLOv5 models
    """
    
    def __init__(self, logger=None):
        """
        Initialize configuration manager
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config_dir = Path(__file__).parent.parent / "configs"
    
    def load_config(self, config_path: str) -> dict:
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
    
    def get_default_config_path(self, backbone_type, model_size):
        """Get default configuration path"""
        if backbone_type == "cspdarknet":
            return self.config_dir / "models" / "yolov5" / "cspdarknet" / f"smartcash_yolov5{model_size}_cspdarknet.yaml"
        elif backbone_type == "efficientnet":
            return self.config_dir / "models" / "yolov5" / "efficientnet" / f"smartcash_yolov5{model_size}_efficientnet.yaml"
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    def get_default_config(self, config_name: str, model_size: str = 's', **kwargs) -> dict:
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
    
    def get_config(self, backbone_type: str, **kwargs) -> dict:
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
            model_size = kwargs.pop('model_size', 's')  # Remove from kwargs to avoid duplicate
            config = self.get_default_config('yolov5s', model_size, **kwargs)
            
            # Update with any additional parameters
            if kwargs:
                config.update(kwargs)
                
            return config
        elif 'efficientnet' in backbone_type.lower():
            # EfficientNet config - use a simpler standard config for now
            # Since EfficientNet integration is complex, use standard YOLOv5 backbone
            kwargs.pop('model_size', None)  # Remove model_size from kwargs to avoid duplicate
            config = self.get_default_config("yolov5s", 's', **kwargs)
            
            # Update with any additional parameters
            if kwargs:
                config.update(kwargs)
                
            return config
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
    
    def convert_multiclass_dicts(self, obj):
        """
        Recursively convert any multi-layer class dicts to integers
        
        Args:
            obj: Object to convert
            
        Returns:
            Converted object with multi-layer dicts converted to total classes
        """
        if isinstance(obj, dict):
            # Check if this looks like a multi-layer class dict
            if all(k.startswith('layer_') for k in obj.keys()) and all(isinstance(v, int) for v in obj.values()):
                total_classes = sum(obj.values())
                self.logger.debug(f"Converting multi-layer dict {obj} -> {total_classes}")
                return total_classes  # Convert to total classes
            else:
                # Recursively process nested dicts
                return {k: self.convert_multiclass_dicts(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_multiclass_dicts(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_multiclass_dicts(item) for item in obj)
        else:
            return obj
    
    def prepare_config_for_yolov5(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare configuration for YOLOv5 compatibility
        
        Args:
            config: Original configuration
            
        Returns:
            YOLOv5-compatible configuration
        """
        config = deepcopy(config)
        
        # Set default parameters if not provided
        config.setdefault('ch', 3)  # Default input channels
        config.setdefault('img_size', 640)  # Default image size
        
        # Get input channels and number of classes
        nc = config['nc']
        
        # Handle multi-layer num_classes configuration early
        if isinstance(nc, dict):
            # Multi-layer configuration: convert dict to total classes for YOLOv5
            # Store original multi-layer config for later use
            config['multi_layer_nc'] = nc
            # Calculate total classes for YOLOv5 compatibility  
            nc = sum(nc.values()) if nc else 80
            config['nc'] = nc  # Update config to use integer
            self.logger.info(f"Multi-layer config detected: {config['multi_layer_nc']} -> total classes: {nc}")
        
        # Comprehensive multi-layer configuration cleanup
        # Ensure ALL dict references are converted to integers for YOLOv5 compatibility
        config = self.convert_multiclass_dicts(config)
        
        # Set default nc value if not provided
        config.setdefault('nc', 80)
        
        # Log the final nc value
        self.logger.info(f"Final nc value for YOLOv5: {config['nc']}")
        
        # Set default values if not provided
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
                [[17, 20, 23], 1, 'Detect', [config['nc'], config['anchors']]],  # Detect(P3, P4, P5) - nc is now integer
            ]
        
        # Ensure all required fields are in the config
        config['ch'] = config.get('ch', 3)  # input channels
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
        
        return config