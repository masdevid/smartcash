"""
Simplified Training Config Handler - No Shared Config Dependency
Just handles config validation and backbone integration.
"""

from typing import Dict, Any
from smartcash.ui.logger import get_module_logger
from .train_defaults import get_default_train_config, get_layer_mode_configs, get_optimization_types
from ..constants import generate_model_name


class SimpleTrainConfigHandler:
    """
    Simplified configuration handler for training module.
    
    Features:
    - 🚀 Training configuration validation
    - 🔧 Backbone configuration integration
    - 📋 Configuration merging and defaults
    - 🛡️ Training parameter validation
    """
    
    def __init__(self):
        """Initialize simple training configuration handler."""
        self.logger = get_module_logger("smartcash.ui.model.train.configs")
        self.layer_configs = get_layer_mode_configs()
        self.optimization_types = get_optimization_types()
        self._config = get_default_train_config()
        
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config.copy()
    
    def update_config(self, config: Dict[str, Any]) -> None:
        """Update configuration."""
        self._config.update(config)
    
    def reset_config(self) -> None:
        """Reset to default configuration."""
        self._config = get_default_train_config()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            training_config = config.get('training', {})
            
            # Validate required training parameters
            required_params = ['epochs', 'batch_size', 'learning_rate', 'layer_mode']
            for param in required_params:
                if param not in training_config:
                    self.logger.warning(f"Missing required training parameter: {param}")
                    return False
            
            # Validate parameter ranges
            epochs = training_config.get('epochs', 0)
            if not isinstance(epochs, int) or epochs <= 0:
                self.logger.warning(f"Invalid epochs value: {epochs}")
                return False
            
            batch_size = training_config.get('batch_size', 0)
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.logger.warning(f"Invalid batch_size value: {batch_size}")
                return False
            
            learning_rate = training_config.get('learning_rate', 0)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                self.logger.warning(f"Invalid learning_rate value: {learning_rate}")
                return False
            
            # Validate layer mode
            layer_mode = training_config.get('layer_mode')
            if layer_mode not in self.layer_configs:
                self.logger.warning(f"Invalid layer_mode: {layer_mode}")
                return False
            
            # Validate optimization type
            optimization_type = training_config.get('optimization_type', 'default')
            if optimization_type not in self.optimization_types:
                self.logger.warning(f"Invalid optimization_type: {optimization_type}")
                return False
            
            self.logger.debug("✅ Training configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def integrate_backbone_config(self, train_config: Dict[str, Any], backbone_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate backbone configuration with training configuration.
        
        Args:
            train_config: Current training configuration
            backbone_config: Backbone configuration to integrate
            
        Returns:
            Merged configuration
        """
        try:
            merged_config = train_config.copy()
            
            # Ensure backbone_integration section exists
            if 'backbone_integration' not in merged_config:
                merged_config['backbone_integration'] = {}
            
            # Extract relevant backbone parameters
            backbone_section = backbone_config.get('backbone', {})
            
            # Map backbone parameters to training config
            backbone_integration = {
                'backbone_type': backbone_section.get('model_type', 'efficientnet_b4'),
                'pretrained': backbone_section.get('pretrained', True),
                'input_size': backbone_section.get('input_size', 640),
                'num_classes': backbone_section.get('num_classes', 7),
                'feature_optimization': backbone_section.get('feature_optimization', True),
                'mixed_precision': backbone_section.get('mixed_precision', True),
                'integrated_at': self._get_timestamp()
            }
            
            merged_config['backbone_integration'].update(backbone_integration)
            
            # Update training config with backbone-influenced settings
            training_section = merged_config.get('training', {})
            
            # Set mixed precision from backbone if not explicitly set
            if 'mixed_precision' not in training_section:
                training_section['mixed_precision'] = backbone_integration['mixed_precision']
            
            # Set input size in training config
            training_section['input_size'] = backbone_integration['input_size']
            training_section['num_classes'] = backbone_integration['num_classes']
            
            merged_config['training'] = training_section
            
            self.logger.info(f"✅ Backbone configuration integrated: {backbone_integration['backbone_type']}")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Failed to integrate backbone config: {e}")
            return train_config
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_config(self) -> bool:
        """
        Save current configuration.
        
        Returns:
            True if saved successfully
        """
        try:
            # For simplicity, just log that save was requested
            # Could implement actual file saving if needed
            self.logger.info("💾 Training configuration save requested")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")
            return False
    
    def get_model_name_preview(self) -> str:
        """
        Get preview of the model name that will be generated.
        
        Returns:
            Model name string
        """
        try:
            backbone_integration = self._config.get('backbone_integration', {})
            training_config = self._config.get('training', {})
            
            backbone_type = backbone_integration.get('backbone_type', 'efficientnet_b4')
            layer_mode = training_config.get('layer_mode', 'single')
            optimization_type = training_config.get('optimization_type', 'default')
            
            return generate_model_name(backbone_type, layer_mode, optimization_type)
            
        except Exception as e:
            self.logger.error(f"Failed to generate model name preview: {e}")
            return "unknown_model"
    
    def refresh_backbone_config(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refresh backbone configuration from backbone module.
        
        Args:
            current_config: Current training configuration
            
        Returns:
            Updated configuration with latest backbone config
        """
        try:
            from smartcash.ui.core.ui_module import SharedMethodRegistry
            
            # Try to get latest backbone configuration
            get_backbone_config = SharedMethodRegistry.get_method('backbone.get_config')
            if get_backbone_config:
                latest_backbone_config = get_backbone_config()
                
                if latest_backbone_config:
                    # Integrate the latest backbone config
                    updated_config = self.integrate_backbone_config(
                        current_config, latest_backbone_config
                    )
                    self.logger.info("🔄 Backbone configuration refreshed successfully")
                    return updated_config
                else:
                    self.logger.warning("No backbone configuration available to refresh")
            else:
                self.logger.warning("Backbone module not available for config refresh")
            
            return current_config
            
        except Exception as e:
            self.logger.error(f"Failed to refresh backbone config: {e}")
            return current_config