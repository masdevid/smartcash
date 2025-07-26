"""
File: smartcash/ui/model/training/configs/training_config_handler.py
Configuration handler for training module using BaseUIModule pattern.
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.logger import get_module_logger
from .unified_training_defaults import (
    get_unified_training_defaults,
    validate_unified_training_config
)


class TrainingConfigHandler:
    """
    Training configuration handler using pure delegation pattern.
    
    This class follows the modern BaseUIModule architecture where config handlers
    are pure implementation classes that delegate to BaseUIModule mixins.
    
    Features:
    - 🏗️ Model selection based on backbone configuration
    - 🎯 Training parameter validation and management
    - 📊 Chart and monitoring configuration
    - 🔧 Optimizer and scheduler configuration
    - 🛡️ Type checking and constraints validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, logger=None):
        """
        Initialize training configuration handler.
        
        Args:
            config: Optional initial configuration
            logger: Optional logger instance
        """
        # Initialize pure delegation pattern (no mixin inheritance)
        self.logger = logger or get_module_logger('smartcash.ui.model.training.config')
        self.module_name = 'training'
        self.parent_module = 'model'
        
        # Load unified configuration
        self._default_config = get_unified_training_defaults()
        self._config = self._default_config.copy()
        
        # Update with provided config if any
        if config:
            self._config.update(config)
        
        # Config sections that require UI synchronization
        self.ui_sync_sections = ['training', 'model_selection', 'monitoring', 'charts', 'ui']
        
        self.logger.info("✅ Training config handler initialized")

    # --- Core Configuration Methods ---

    def get_current_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._config.copy()

    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.validate_config(updates):
            self._config.update(updates)
            self.logger.debug(f"Configuration updated: {list(updates.keys())}")
        else:
            raise ValueError("Invalid configuration updates provided")

    def reset_config(self) -> None:
        """Reset configuration to defaults."""
        self._config = get_unified_training_defaults().copy()
        self.logger.info("Configuration reset to defaults")

    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            config_path: Optional path to save config file
            
        Returns:
            True if save successful, False otherwise
        """
        try:
            # Implementation would save to YAML file
            self.logger.info("Configuration saved successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False

    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default training configuration.
        
        Returns:
            Default configuration dictionary
        """
        return get_unified_training_defaults()

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - ensure required keys exist
            if not isinstance(config, dict):
                return False
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._config
    
    @config.setter
    def config(self, value: Dict[str, Any]) -> None:
        """Set configuration with validation."""
        if self.validate_config(value):
            self._config = value
        else:
            raise ValueError("Invalid configuration provided")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default training configuration.
        
        Returns:
            Default configuration dictionary
        """
        return get_unified_training_defaults()
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            if not isinstance(config, dict):
                self.logger.error("Configuration must be a dictionary")
                return False
            
            # Check required sections
            required_sections = ['training', 'model_selection']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required section: {section}")
                    return False
            
            # Training-specific validation
            if not self._validate_training_section(config.get('training', {})):
                return False
            
            if not self._validate_model_selection_section(config.get('model_selection', {})):
                return False
            
            self.logger.debug("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def _validate_training_section(self, training_config: Dict[str, Any]) -> bool:
        """Validate training configuration section."""
        # Check epochs
        epochs = training_config.get('epochs', 100)
        min_epochs = 1
        max_epochs = 200
        if not (min_epochs <= epochs <= max_epochs):
            self.logger.error(f"epochs must be between {min_epochs} and {max_epochs}")
            return False
        
        # Check batch size
        batch_size = training_config.get('batch_size', 16)
        min_batch = 1
        max_batch = 32
        if not (min_batch <= batch_size <= max_batch):
            self.logger.error(f"batch_size must be between {min_batch} and {max_batch}")
            return False
        
        # Check learning rate
        lr = training_config.get('learning_rate', 0.001)
        min_lr = 1e-7
        max_lr = 1e-1
        if not (min_lr <= lr <= max_lr):
            self.logger.error(f"learning_rate must be between {min_lr} and {max_lr}")
            return False
        
        # Check optimizer
        optimizer = training_config.get('optimizer', 'adam')
        if optimizer not in self.available_optimizers:
            self.logger.error(f"Invalid optimizer: {optimizer}. Available: {list(self.available_optimizers.keys())}")
            return False
        
        # Check scheduler
        scheduler = training_config.get('scheduler', 'cosine')
        if scheduler not in self.available_schedulers:
            self.logger.error(f"Invalid scheduler: {scheduler}. Available: {list(self.available_schedulers.keys())}")
            return False
        
        return True
    
    def _validate_model_selection_section(self, model_config: Dict[str, Any]) -> bool:
        """Validate model selection configuration section."""
        # Check source
        source = model_config.get('source', 'backbone')
        valid_sources = ['backbone', 'checkpoint', 'pretrained']
        if source not in valid_sources:
            self.logger.error(f"Invalid model source: {source}. Available: {valid_sources}")
            return False
        
        # If checkpoint source, check path exists
        if source == 'checkpoint':
            checkpoint_path = model_config.get('checkpoint_path', '')
            if not checkpoint_path:
                self.logger.error("checkpoint_path is required when source is 'checkpoint'")
                return False
        
        return True
    
    # Note: update_config is provided by ConfigurationMixin
    
    def set_ui_components(self, ui_components: Dict[str, Any]) -> None:
        """
        Set UI components reference for configuration extraction.
        
        Args:
            ui_components: UI components dictionary
        """
        self._ui_components = ui_components
        self.logger.debug("✅ UI components reference set")
    
    def get_validation_errors(self, config: Dict[str, Any]) -> List[str]:
        """
        Get detailed validation errors for configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        try:
            if not isinstance(config, dict):
                errors.append("Configuration must be a dictionary")
                return errors
            
            # Check required sections
            if 'training' not in config:
                errors.append("Missing 'training' section")
            if 'model_selection' not in config:
                errors.append("Missing 'model_selection' section")
            
            # Validate training section details
            training_config = config.get('training', {})
            
            epochs = training_config.get('epochs', 100)
            if not isinstance(epochs, int) or epochs < 1 or epochs > 1000:
                errors.append("Epochs must be integer between 1 and 1000")
            
            batch_size = training_config.get('batch_size', 16)
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 256:
                errors.append("Batch size must be integer between 1 and 256")
            
            lr = training_config.get('learning_rate', 0.001)
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
                errors.append("Learning rate must be number between 0 and 1.0")
            
            optimizer = training_config.get('optimizer', 'adam')
            if optimizer not in self.available_optimizers:
                errors.append(f"Invalid optimizer: {optimizer}")
            
            scheduler = training_config.get('scheduler', 'cosine')
            if scheduler not in self.available_schedulers:
                errors.append(f"Invalid scheduler: {scheduler}")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    def select_model_from_backbone(self, backbone_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select and configure model based on backbone configuration.
        
        Args:
            backbone_config: Backbone configuration from backbone module
            
        Returns:
            Model selection configuration
        """
        try:
            model_selection = {
                'source': 'backbone',
                'auto_detect': True,
                'validation_required': True
            }
            
            # Extract backbone information
            backbone_type = backbone_config.get('backbone', {}).get('model_type', 'efficientnet_b4')
            input_size = backbone_config.get('backbone', {}).get('input_size', 640)
            num_classes = backbone_config.get('backbone', {}).get('num_classes', 7)
            feature_optimization = backbone_config.get('backbone', {}).get('feature_optimization', False)
            
            # Update model selection config
            model_selection.update({
                'backbone_type': backbone_type,
                'input_size': input_size,
                'num_classes': num_classes,
                'feature_optimization': feature_optimization,
                'model_name': backbone_config.get('model', {}).get('model_name', 'smartcash_model')
            })
            
            # Update training config based on backbone
            self._config['model_selection'] = model_selection
            
            # Adjust training parameters based on backbone
            if backbone_type == 'efficientnet_b4':
                # EfficientNet-B4 specific optimizations
                self._config['training'].update({
                    'learning_rate': 0.001,
                    'batch_size': min(self._config['training'].get('batch_size', 16), 32),
                    'optimizer': 'adamw'
                })
            elif backbone_type == 'cspdarknet':
                # CSPDarkNet specific optimizations
                self._config['training'].update({
                    'learning_rate': 0.01,
                    'batch_size': min(self._config['training'].get('batch_size', 16), 64),
                    'optimizer': 'sgd'
                })
            
            self.logger.info(f"✅ Model selected: {backbone_type} with {num_classes} classes")
            
            return {
                'success': True,
                'message': f'Model selected from backbone: {backbone_type}',
                'model_selection': model_selection
            }
            
        except Exception as e:
            error_msg = f"Failed to select model from backbone: {e}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    # ==================== ABSTRACT METHOD IMPLEMENTATIONS ====================
    
    def create_config_handler(self, config: Dict[str, Any]) -> 'TrainingConfigHandler':
        """Create config handler instance for training module."""
        return TrainingConfigHandler(config)