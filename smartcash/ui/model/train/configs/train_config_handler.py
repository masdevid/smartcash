"""
File: smartcash/ui/model/train/configs/train_config_handler.py
Configuration handler for train module following UIModule pattern.
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.core.handlers.config_handler import ConfigHandler
from smartcash.ui.logger import get_module_logger
from .train_defaults import get_default_train_config, get_layer_mode_configs, get_optimization_types
from ..constants import VALIDATION_CONFIG, LayerMode, OptimizationType, generate_model_name


class TrainConfigHandler(ConfigHandler):
    """
    Configuration handler for training module.
    
    Features:
    - 🚀 Training configuration validation
    - 🔧 Backbone configuration integration
    - 📋 Single/multilayer configuration management
    - 🎯 UI component sync support
    - 🛡️ Training parameter validation
    - 📊 Chart configuration management
    """
    
    def __init__(self):
        """Initialize training configuration handler."""
        super().__init__(
            module_name='train',
            parent_module='model'
        )
        self.logger = get_module_logger("smartcash.ui.model.train.configs")
        self.layer_configs = get_layer_mode_configs()
        self.optimization_types = get_optimization_types()
        
        # Config sections that require UI synchronization
        self.ui_sync_sections = ['training', 'optimizer', 'scheduler', 'monitoring', 'ui']
        
        self.logger.debug("✅ TrainConfigHandler initialized")
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default training configuration.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_train_config()
    
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
            required_sections = ['training', 'optimizer', 'scheduler']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required section: {section}")
                    return False
            
            # Validate training section
            training_config = config.get('training', {})
            if not self._validate_training_section(training_config):
                return False
            
            # Validate optimizer section
            optimizer_config = config.get('optimizer', {})
            if not self._validate_optimizer_section(optimizer_config):
                return False
            
            # Validate scheduler section
            scheduler_config = config.get('scheduler', {})
            if not self._validate_scheduler_section(scheduler_config):
                return False
            
            self.logger.debug("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return False
    
    def _validate_training_section(self, training_config: Dict[str, Any]) -> bool:
        """Validate training configuration section."""
        # Check layer mode
        layer_mode = training_config.get('layer_mode')
        if not layer_mode:
            self.logger.error("layer_mode is required in training section")
            return False
        
        if layer_mode not in [mode.value for mode in LayerMode]:
            self.logger.error(f"Invalid layer_mode: {layer_mode}")
            return False
        
        # Check epochs
        epochs = training_config.get('epochs', 100)
        epoch_limits = VALIDATION_CONFIG.get('epochs', {'min': 1, 'max': 1000})
        if not (epoch_limits['min'] <= epochs <= epoch_limits['max']):
            self.logger.error(f"Epochs must be between {epoch_limits['min']} and {epoch_limits['max']}")
            return False
        
        # Check batch size
        batch_size = training_config.get('batch_size', 16)
        batch_limits = VALIDATION_CONFIG.get('batch_size', {'min': 1, 'max': 256})
        if not (batch_limits['min'] <= batch_size <= batch_limits['max']):
            self.logger.error(f"Batch size must be between {batch_limits['min']} and {batch_limits['max']}")
            return False
        
        # Check learning rate
        learning_rate = training_config.get('learning_rate', 0.001)
        lr_limits = VALIDATION_CONFIG.get('learning_rate', {'min': 1e-6, 'max': 1.0})
        if not (lr_limits['min'] <= learning_rate <= lr_limits['max']):
            self.logger.error(f"Learning rate must be between {lr_limits['min']} and {lr_limits['max']}")
            return False
        
        # Check optimization type
        optimization_type = training_config.get('optimization_type', 'default')
        if optimization_type not in [opt.value for opt in OptimizationType]:
            self.logger.error(f"Invalid optimization_type: {optimization_type}")
            return False
        
        return True
    
    def _validate_optimizer_section(self, optimizer_config: Dict[str, Any]) -> bool:
        """Validate optimizer configuration section."""
        optimizer_type = optimizer_config.get('type', 'adam')
        valid_optimizers = ['adam', 'sgd', 'adamw']
        
        if optimizer_type not in valid_optimizers:
            self.logger.error(f"Invalid optimizer type: {optimizer_type}. Valid: {valid_optimizers}")
            return False
        
        return True
    
    def _validate_scheduler_section(self, scheduler_config: Dict[str, Any]) -> bool:
        """Validate scheduler configuration section."""
        scheduler_type = scheduler_config.get('type', 'cosine')
        valid_schedulers = ['cosine', 'step', 'exponential', 'plateau']
        
        if scheduler_type not in valid_schedulers:
            self.logger.error(f"Invalid scheduler type: {scheduler_type}. Valid: {valid_schedulers}")
            return False
        
        return True
    
    def integrate_backbone_config(self, train_config: Dict[str, Any], backbone_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate backbone configuration into training configuration.
        
        Args:
            train_config: Current training configuration
            backbone_config: Backbone configuration from backbone module
            
        Returns:
            Integrated configuration
        """
        try:
            integrated_config = train_config.copy()
            
            # Extract backbone information
            backbone_info = backbone_config.get('backbone', {})
            model_info = backbone_config.get('model', {})
            
            # Update training config with backbone information
            integrated_config.setdefault('backbone_integration', {})
            integrated_config['backbone_integration'].update({
                'backbone_type': backbone_info.get('model_type', 'efficientnet_b4'),
                'pretrained': backbone_info.get('pretrained', True),
                'feature_optimization': backbone_info.get('feature_optimization', True),
                'input_size': backbone_info.get('input_size', 640),
                'num_classes': backbone_info.get('num_classes', 7),
                'backbone_config': backbone_info,
                'model_config': model_info
            })
            
            # Generate model name based on configuration
            backbone_type = backbone_info.get('model_type', 'efficientnet_b4')
            layer_mode = integrated_config.get('training', {}).get('layer_mode', 'single')
            optimization_type = integrated_config.get('training', {}).get('optimization_type', 'default')
            
            model_name = generate_model_name(backbone_type, layer_mode, optimization_type)
            integrated_config.setdefault('model_storage', {})
            integrated_config['model_storage']['model_name'] = model_name
            
            self.logger.info(f"🔗 Integrated backbone config, model name: {model_name}")
            return integrated_config
            
        except Exception as e:
            self.logger.error(f"Failed to integrate backbone config: {e}")
            return train_config
    
    def sync_to_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
        """
        Synchronize configuration to UI components.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration to sync
            
        Returns:
            True if sync successful, False otherwise
        """
        try:
            if not ui_components or not config:
                self.logger.warning("Missing UI components or config for sync")
                return False
            
            # Update form container if available
            form_container = ui_components.get('form_container')
            if form_container and hasattr(form_container, 'update_from_config'):
                form_container.update_from_config(config)
                self.logger.debug("✅ Form container updated from config")
            
            # Update chart containers with live data configuration
            loss_chart = ui_components.get('loss_chart')
            if loss_chart and hasattr(loss_chart, 'update_config'):
                chart_config = config.get('monitoring', {})
                loss_chart.update_config(chart_config)
                self.logger.debug("✅ Loss chart updated from config")
            
            map_chart = ui_components.get('map_chart')
            if map_chart and hasattr(map_chart, 'update_config'):
                chart_config = config.get('monitoring', {})
                map_chart.update_config(chart_config)
                self.logger.debug("✅ mAP chart updated from config")
            
            return True
            
        except Exception as e:
            self.logger.error(f"UI sync error: {e}")
            return False
    
    def sync_from_ui(self, ui_components: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Synchronize configuration from UI components.
        
        Args:
            ui_components: UI components dictionary
            
        Returns:
            Configuration dictionary if sync successful, None otherwise
        """
        try:
            if not ui_components:
                self.logger.warning("No UI components available for sync")
                return None
            
            # Get current config from form container
            form_container = ui_components.get('form_container')
            if form_container and hasattr(form_container, 'get_form_values'):
                form_values = form_container.get_form_values()
                
                # Convert form values to config structure
                config = self._form_values_to_config(form_values)
                
                self.logger.debug("✅ Configuration synced from UI")
                return config
            
            self.logger.warning("Form container not available for sync")
            return None
            
        except Exception as e:
            self.logger.error(f"UI to config sync error: {e}")
            return None
    
    def _form_values_to_config(self, form_values: Dict[str, Any]) -> Dict[str, Any]:
        """Convert form values to configuration structure."""
        layer_mode = form_values.get('layer_mode', 'single')
        epochs = form_values.get('epochs', 100)
        batch_size = form_values.get('batch_size', 16)
        learning_rate = form_values.get('learning_rate', 0.001)
        optimization_type = form_values.get('optimization_type', 'default')
        mixed_precision = form_values.get('mixed_precision', True)
        early_stopping = form_values.get('early_stopping_enabled', True)
        
        return {
            'training': {
                'layer_mode': layer_mode,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'validation_interval': form_values.get('validation_interval', 1),
                'save_interval': form_values.get('save_interval', 5),
                'optimization_type': optimization_type,
                'mixed_precision': mixed_precision,
                'gradient_accumulation': form_values.get('gradient_accumulation', 1),
                'early_stopping': {
                    'enabled': early_stopping,
                    'patience': form_values.get('early_stopping_patience', 15),
                    'metric': 'val_map50',
                    'mode': 'max'
                }
            },
            'optimizer': {
                'type': form_values.get('optimizer_type', 'adam'),
                'weight_decay': form_values.get('weight_decay', 0.0005),
                'momentum': form_values.get('momentum', 0.9)
            },
            'scheduler': {
                'type': form_values.get('scheduler_type', 'cosine'),
                'warmup_epochs': form_values.get('warmup_epochs', 5),
                'min_lr': form_values.get('min_lr', 0.00001)
            },
            'monitoring': {
                'live_charts_enabled': form_values.get('live_charts_enabled', True),
                'progress_updates_enabled': form_values.get('progress_updates_enabled', True),
                'chart_update_interval': form_values.get('chart_update_interval', 1000)
            },
            'ui': {
                'show_advanced_options': form_values.get('show_advanced_options', False),
                'dual_charts_layout': form_values.get('dual_charts_layout', 'horizontal')
            }
        }
    
    def get_layer_mode_info(self, layer_mode: str) -> Dict[str, Any]:
        """
        Get layer mode information.
        
        Args:
            layer_mode: Layer mode string
            
        Returns:
            Layer mode information dictionary
        """
        return self.layer_configs.get(layer_mode, {})
    
    def get_optimization_type_info(self, optimization_type: str) -> Dict[str, Any]:
        """
        Get optimization type information.
        
        Args:
            optimization_type: Optimization type string
            
        Returns:
            Optimization type information dictionary
        """
        return self.optimization_types.get(optimization_type, {})
    
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
            if 'optimizer' not in config:
                errors.append("Missing 'optimizer' section")
            if 'scheduler' not in config:
                errors.append("Missing 'scheduler' section")
            
            # Validate training section details
            training_config = config.get('training', {})
            
            layer_mode = training_config.get('layer_mode')
            if not layer_mode:
                errors.append("Layer mode is required")
            elif layer_mode not in [mode.value for mode in LayerMode]:
                errors.append(f"Invalid layer mode: {layer_mode}")
            
            epochs = training_config.get('epochs', 100)
            if not isinstance(epochs, int) or epochs < 1 or epochs > 1000:
                errors.append("Epochs must be integer between 1 and 1000")
            
            batch_size = training_config.get('batch_size', 16)
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 256:
                errors.append("Batch size must be integer between 1 and 256")
            
            learning_rate = training_config.get('learning_rate', 0.001)
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0 or learning_rate > 1.0:
                errors.append("Learning rate must be number between 0 and 1.0")
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors