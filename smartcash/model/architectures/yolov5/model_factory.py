"""
Modern Model Factory for YOLOv5 Integration
Simplified factory for creating optimized SmartCash models
"""

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class YOLOv5ModelFactory:
    """
    Modern factory for creating optimized YOLOv5-based SmartCash models
    """
    
    def __init__(self):
        """Initialize model factory"""
        self.logger = logger
        self.logger.info("✅ YOLOv5 model factory initialized")
    
    def create_model(self, backbone: str = 'yolov5s', num_classes: int = 17, 
                    img_size: int = 640, pretrained: bool = True, **kwargs):
        """
        Create optimized YOLOv5-based SmartCash model
        
        Args:
            backbone: Backbone type ('cspdarknet', 'efficientnet_b4', 'yolov5s', etc.)
            num_classes: Number of output classes (default: 17 for SmartCash)
            img_size: Input image size
            pretrained: Whether to use pretrained weights
            **kwargs: Additional configuration
            
        Returns:
            SmartCash YOLOv5 model instance
        """
        try:
            # Use the modern SmartCash YOLOv5 architecture
            from smartcash.model.architectures.model import SmartCashYOLOv5Model
            
            # Create optimized model
            model = SmartCashYOLOv5Model(
                backbone=backbone,
                num_classes=num_classes,
                img_size=img_size,
                pretrained=pretrained,
                device='auto'  # Auto-detect device
            )
            
            self.logger.info(f"✅ Created {backbone} model with {num_classes} classes")
            return model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create model: {e}")
            raise
    
    def get_model_info(self, backbone: str) -> dict:
        """
        Get information about a specific model backbone
        
        Args:
            backbone: Backbone name
            
        Returns:
            Dictionary with model information
        """
        from smartcash.model.architectures.model import SmartCashYOLOv5Model
        
        model_info = SmartCashYOLOv5Model.get_backbone_info()
        return model_info.get(backbone, {
            'params': 'Unknown',
            'speed': 'Unknown', 
            'use_case': 'Custom backbone'
        })
    
    def get_supported_backbones(self) -> list:
        """
        Get list of supported backbone architectures
        
        Returns:
            List of supported backbone names
        """
        from smartcash.model.architectures.model import SmartCashYOLOv5Model
        return SmartCashYOLOv5Model.get_supported_backbones()





# Export factory instance for easy importing
model_factory = YOLOv5ModelFactory()