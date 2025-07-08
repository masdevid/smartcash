"""
File: smartcash/ui/model/pretrained/configs/pretrained_config_handler.py
Configuration handler for pretrained models module.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.ui.core.handlers.config_handler import ConfigHandler
from .pretrained_defaults import get_pretrained_defaults, get_yaml_schema
from ..constants import DEFAULT_CONFIG, PretrainedModelType


class PretrainedConfigHandler(ConfigHandler):
    """
    Configuration handler for pretrained models module.
    Manages configuration persistence and validation.
    """
    
    def __init__(self, module_name: str = "pretrained"):
        """Initialize the pretrained config handler."""
        super().__init__(module_name, "model")
        self.config_key = "pretrained_models"
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for pretrained models.
        
        Returns:
            Dictionary containing default configuration
        """
        return get_pretrained_defaults()
    
    def get_yaml_schema(self) -> Dict[str, Any]:
        """
        Get YAML schema for configuration validation.
        
        Returns:
            Dictionary containing YAML schema
        """
        return get_yaml_schema()
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the config handler (required by base class).
        
        Returns:
            Dictionary containing initialization result
        """
        return {
            'success': True,
            'config_handler': self.__class__.__name__,
            'module': self.module_name
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate pretrained models configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validated and sanitized configuration
        """
        validated_config = self.get_default_config()
        
        if not isinstance(config, dict):
            return validated_config
        
        # Validate models directory
        models_dir = config.get("models_dir", "")
        if isinstance(models_dir, str) and models_dir.strip():
            # Ensure path is absolute and normalized
            path = Path(models_dir.strip()).expanduser().resolve()
            validated_config["models_dir"] = str(path)
        
        # Validate model URLs
        model_urls = config.get("model_urls", {})
        if isinstance(model_urls, dict):
            for model_type in PretrainedModelType:
                url = model_urls.get(model_type.value, "")
                if isinstance(url, str) and url.strip():
                    # Basic URL validation
                    url = url.strip()
                    if url.startswith(("http://", "https://")):
                        validated_config["model_urls"][model_type.value] = url
        
        # Validate boolean flags
        for flag in ["auto_download", "validate_downloads", "cleanup_failed"]:
            if flag in config and isinstance(config[flag], bool):
                validated_config[flag] = config[flag]
        
        # Validate numeric values
        download_timeout = config.get("download_timeout")
        if isinstance(download_timeout, (int, float)) and download_timeout > 0:
            validated_config["download_timeout"] = int(download_timeout)
        
        return validated_config
    
    def extract_ui_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Args:
            ui_components: Dictionary containing UI components
            
        Returns:
            Dictionary containing extracted configuration
        """
        config = self.get_default_config()
        
        try:
            input_options = ui_components.get('input_options', {})
            
            # Extract models directory
            model_dir_input = input_options.get('model_dir_input')
            if model_dir_input and hasattr(model_dir_input, 'value'):
                models_dir = model_dir_input.value.strip()
                if models_dir:
                    config["models_dir"] = models_dir
            
            # Extract custom URLs
            yolo_url_input = input_options.get('yolo_url_input')
            if yolo_url_input and hasattr(yolo_url_input, 'value'):
                yolo_url = yolo_url_input.value.strip()
                if yolo_url:
                    config["model_urls"]["yolov5s"] = yolo_url
            
            efficientnet_url_input = input_options.get('efficientnet_url_input')
            if efficientnet_url_input and hasattr(efficientnet_url_input, 'value'):
                efficientnet_url = efficientnet_url_input.value.strip()
                if efficientnet_url:
                    config["model_urls"]["efficientnet_b4"] = efficientnet_url
            
        except Exception as e:
            # Log error but don't fail completely
            print(f"Warning: Error extracting UI config: {str(e)}")
        
        return self.validate_config(config)
    
    def update_ui_from_config(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Update UI components from configuration.
        
        Args:
            ui_components: Dictionary containing UI components
            config: Configuration to apply to UI
        """
        try:
            # Validate config first
            validated_config = self.validate_config(config)
            input_options = ui_components.get('input_options', {})
            
            # Update models directory input
            model_dir_input = input_options.get('model_dir_input')
            if model_dir_input and hasattr(model_dir_input, 'value'):
                model_dir_input.value = validated_config.get("models_dir", "/data/pretrained")
            
            # Update URL inputs
            model_urls = validated_config.get("model_urls", {})
            
            yolo_url_input = input_options.get('yolo_url_input')
            if yolo_url_input and hasattr(yolo_url_input, 'value'):
                yolo_url_input.value = model_urls.get("yolov5s", "")
            
            efficientnet_url_input = input_options.get('efficientnet_url_input')
            if efficientnet_url_input and hasattr(efficientnet_url_input, 'value'):
                efficientnet_url_input.value = model_urls.get("efficientnet_b4", "")
                
        except Exception as e:
            # Log error but don't fail completely
            print(f"Warning: Error updating UI from config: {str(e)}")
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """
        Get a human-readable summary of the configuration.
        
        Args:
            config: Configuration to summarize
            
        Returns:
            String containing configuration summary
        """
        try:
            validated_config = self.validate_config(config)
            
            summary_lines = [
                "🤖 Pretrained Models Configuration:",
                f"📁 Models Directory: {validated_config.get('models_dir', 'Not set')}",
                ""
            ]
            
            # Model URLs summary
            model_urls = validated_config.get("model_urls", {})
            summary_lines.append("🔗 Download URLs:")
            
            for model_type in PretrainedModelType:
                url = model_urls.get(model_type.value, "")
                model_name = model_type.value.replace("_", "-").upper()
                if url:
                    summary_lines.append(f"  • {model_name}: Custom URL provided")
                else:
                    summary_lines.append(f"  • {model_name}: Default URL")
            
            summary_lines.append("")
            
            # Settings summary
            flags = ["auto_download", "validate_downloads", "cleanup_failed"]
            summary_lines.append("⚙️ Settings:")
            for flag in flags:
                value = "✅ Enabled" if validated_config.get(flag, False) else "❌ Disabled"
                flag_name = flag.replace("_", " ").title()
                summary_lines.append(f"  • {flag_name}: {value}")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            return f"❌ Error generating config summary: {str(e)}"