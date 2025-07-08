"""
File: smartcash/ui/model/backbone/configs/backbone_config_handler.py
Description: Configuration handler for backbone module following dependency pattern
"""

from typing import Dict, Any, Optional
import copy

try:
    from smartcash.common.logger import SmartCashLogger
except ImportError:
    # Fallback for testing
    class SmartCashLogger:
        def __init__(self, name=None): pass
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")

from .backbone_defaults import (
    get_default_backbone_config,
    get_available_backbones,
    get_detection_layers_config,
    get_layer_modes_config
)

class BackboneConfigHandler:
    """Configuration handler for backbone module."""
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """Initialize configuration handler.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or SmartCashLogger('BackboneConfigHandler')
        self._config = get_default_backbone_config()
        self._available_backbones = get_available_backbones()
        self._detection_layers = get_detection_layers_config()
        self._layer_modes = get_layer_modes_config()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return copy.deepcopy(self._config)
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """Update configuration.
        
        Args:
            config: New configuration to apply
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            # Validate configuration
            if not self.validate_config(config):
                return False
            
            # Deep merge configuration
            self._config = self._deep_merge(self._config, config)
            self.logger.info("✅ Configuration updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update configuration: {e}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            model_config = config.get('model', {})
            
            # Validate backbone
            backbone = model_config.get('backbone')
            if backbone and backbone not in self._available_backbones:
                self.logger.warning(f"⚠️ Unknown backbone: {backbone}")
                return False
            
            # Validate detection layers
            layers = model_config.get('detection_layers', [])
            for layer in layers:
                if layer not in self._detection_layers:
                    self.logger.warning(f"⚠️ Unknown detection layer: {layer}")
                    return False
            
            # Validate layer mode
            layer_mode = model_config.get('layer_mode')
            if layer_mode and layer_mode not in self._layer_modes:
                self.logger.warning(f"⚠️ Unknown layer mode: {layer_mode}")
                return False
            
            # Validate layer mode compatibility
            if layer_mode == 'single' and len(layers) > 1:
                self.logger.warning("⚠️ Single layer mode with multiple layers detected")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        self._config = get_default_backbone_config()
        self.logger.info("🔄 Configuration reset to defaults")
    
    def get_available_backbones(self) -> Dict[str, Dict[str, Any]]:
        """Get available backbones.
        
        Returns:
            Dictionary of available backbones
        """
        return copy.deepcopy(self._available_backbones)
    
    def get_detection_layers_config(self) -> Dict[str, Dict[str, Any]]:
        """Get detection layers configuration.
        
        Returns:
            Dictionary of detection layers configuration
        """
        return copy.deepcopy(self._detection_layers)
    
    def get_layer_modes_config(self) -> Dict[str, Dict[str, Any]]:
        """Get layer modes configuration.
        
        Returns:
            Dictionary of layer modes configuration
        """
        return copy.deepcopy(self._layer_modes)
    
    def set_backbone(self, backbone: str) -> bool:
        """Set backbone type.
        
        Args:
            backbone: Backbone type to set
            
        Returns:
            True if successful, False otherwise
        """
        if backbone not in self._available_backbones:
            self.logger.warning(f"⚠️ Unknown backbone: {backbone}")
            return False
        
        self._config['model']['backbone'] = backbone
        self.logger.info(f"✅ Backbone set to: {backbone}")
        return True
    
    def set_detection_layers(self, layers: list) -> bool:
        """Set detection layers.
        
        Args:
            layers: List of detection layers
            
        Returns:
            True if successful, False otherwise
        """
        # Validate layers
        for layer in layers:
            if layer not in self._detection_layers:
                self.logger.warning(f"⚠️ Unknown detection layer: {layer}")
                return False
        
        # Ensure banknote layer is always included
        if 'banknote' not in layers:
            layers = ['banknote'] + layers
        
        self._config['model']['detection_layers'] = layers
        self.logger.info(f"✅ Detection layers set to: {layers}")
        return True
    
    def set_layer_mode(self, mode: str) -> bool:
        """Set layer mode.
        
        Args:
            mode: Layer mode to set
            
        Returns:
            True if successful, False otherwise
        """
        if mode not in self._layer_modes:
            self.logger.warning(f"⚠️ Unknown layer mode: {mode}")
            return False
        
        self._config['model']['layer_mode'] = mode
        
        # Auto-adjust detection layers if needed
        if mode == 'single':
            self._config['model']['detection_layers'] = ['banknote']
        
        self.logger.info(f"✅ Layer mode set to: {mode}")
        return True
    
    def set_feature_optimization(self, enabled: bool, use_attention: bool = True) -> None:
        """Set feature optimization settings.
        
        Args:
            enabled: Whether to enable feature optimization
            use_attention: Whether to use attention mechanisms
        """
        self._config['model']['feature_optimization'] = {
            'enabled': enabled,
            'use_attention': use_attention,
            'testing_mode': False
        }
        self.logger.info(f"✅ Feature optimization: enabled={enabled}, attention={use_attention}")
    
    def set_mixed_precision(self, enabled: bool) -> None:
        """Set mixed precision training.
        
        Args:
            enabled: Whether to enable mixed precision
        """
        self._config['model']['mixed_precision'] = enabled
        self.logger.info(f"✅ Mixed precision set to: {enabled}")
    
    def get_backbone_info(self, backbone: Optional[str] = None) -> Dict[str, Any]:
        """Get backbone information.
        
        Args:
            backbone: Backbone type (use current if None)
            
        Returns:
            Backbone information dictionary
        """
        backbone = backbone or self._config['model']['backbone']
        return self._available_backbones.get(backbone, {})
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries.
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            Merged dictionary
        """
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result