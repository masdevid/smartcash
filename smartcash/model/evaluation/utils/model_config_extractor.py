"""
Model configuration extractor for evaluation.
Extracts model configuration from checkpoint metadata.
"""

from typing import Dict, Any

from smartcash.common.logger import get_logger


class ModelConfigExtractor:
    """Extract model configuration from checkpoint metadata"""
    
    def __init__(self):
        self.logger = get_logger('model_config_extractor')
    
    def extract_model_config_from_checkpoint(self, checkpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 Extract model configuration from checkpoint metadata"""
        
        # Extract backbone information with cleanup
        backbone = checkpoint_info.get('backbone', 'cspdarknet')
        # Clean up backbone names that have extra suffixes
        if backbone.startswith('cspdarknet'):
            backbone = 'cspdarknet'
        elif backbone.startswith('efficientnet'):
            backbone = 'efficientnet_b4'
        
        layer_mode = checkpoint_info.get('layer_mode', 'multi')
        num_classes = 17  # Updated default for current training pipeline
        
        # Try to extract from various locations in checkpoint
        config = checkpoint_info.get('config', {})
        if 'model' in config and isinstance(config['model'], dict):
            num_classes = config['model'].get('num_classes', 17)
        elif 'num_classes' in config:
            num_classes = config['num_classes']
        elif 'training_config' in checkpoint_info and isinstance(checkpoint_info['training_config'], dict):
            num_classes = checkpoint_info['training_config'].get('num_classes', 17)
        
        # Build comprehensive config for API
        model_config = {
            'device': {'type': 'auto'},
            'model': {
                'backbone': backbone,
                'num_classes': num_classes,
                'img_size': 640,
                'layer_mode': layer_mode,
                'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
                'feature_optimization': {'enabled': True}
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 1e-3
            }
        }
        
        self.logger.debug(f"🔧 Extracted model config: backbone={backbone}, num_classes={num_classes}, layer_mode={layer_mode}")
        return model_config


def create_model_config_extractor() -> ModelConfigExtractor:
    """Factory function to create model config extractor"""
    return ModelConfigExtractor()