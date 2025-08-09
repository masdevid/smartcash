"""
SmartCash YOLOv5 Integration Manager
Main entry point for YOLOv5 integration functionality
"""

from typing import List
from smartcash.common.logger import SmartCashLogger
from .memory_manager import YOLOv5MemoryManager
from .config_manager import YOLOv5ConfigManager
from .pretrained_weights import YOLOv5PretrainedWeights
from .model_factory import model_factory
from .training_compatibility import SmartCashTrainingCompatibilityWrapper

# Import YOLOv5 registration
try:
    from pathlib import Path
    import sys
    yolov5_path = Path(__file__).parent.parent.parent.parent.parent / "yolov5"
    if str(yolov5_path) not in sys.path:
        sys.path.append(str(yolov5_path))
    
    from smartcash.model.architectures.heads.yolov5_head import register_yolov5_components
    YOLOV5_AVAILABLE = True
except ImportError:
    def register_yolov5_components(): pass
    YOLOV5_AVAILABLE = False


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
        
        # Initialize component managers
        self.memory_manager = YOLOv5MemoryManager(logger=self.logger)
        self.config_manager = YOLOv5ConfigManager(logger=self.logger)
        self.weights_manager = YOLOv5PretrainedWeights(logger=self.logger)
        self.model_factory = model_factory
        
        # Perform memory cleanup before initialization
        self.memory_manager.initial_cleanup()
        
        # Register custom components with YOLOv5
        if YOLOV5_AVAILABLE:
            register_yolov5_components()
            
        
        self.logger.info("✅ SmartCash YOLOv5 integration initialized")
    
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
            if config_path:
                config = self.config_manager.load_config(config_path)
            else:
                # Pass model_size to config manager
                kwargs['model_size'] = model_size
                config = self.config_manager.get_config(backbone_type, **kwargs)
            
            # Prepare config for YOLOv5 compatibility
            config = self.config_manager.prepare_config_for_yolov5(config)
            
            # Get input channels and number of classes
            ch = config['ch']
            nc = config['nc']
            anchors = config.get('anchors')
            
            # Create the model wrapper
            model = self.model_factory.create_model(
                backbone=backbone_type,
                num_classes=config['nc'],
                img_size=config['img_size'],
                pretrained=kwargs.get('pretrained', False),
                ch=config['ch'],
                anchors=config.get('anchors')
            )
            
            # Store pretrained weights path if specified
            if kwargs.get('pretrained', False):
                # Get pretrained weights path, downloading to /data/pretrained if needed
                weights_name = kwargs.get('weights', 'yolov5s.pt')
                model.pretrained_weights = self.weights_manager.get_weights_path(weights_name)
                model.logger = self.logger
            
            # Initialize model with dummy input to set strides
            import torch
            with torch.no_grad():
                dummy_input = torch.zeros(1, ch, config['img_size'], config['img_size'])
                _ = model(dummy_input)
            
            # Cleanup memory after model creation
            self.memory_manager.cleanup_after_model_creation()
                    
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to create YOLOv5 model: {str(e)}")
            self.logger.error(f"❌ Failed to create YOLOv5 model: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to create YOLOv5 model: {str(e)}") from e
    
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