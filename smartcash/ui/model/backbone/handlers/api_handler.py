"""
File: smartcash/ui/model/backbone/handlers/api_handler.py
Deskripsi: Handler untuk API integration dengan SmartCashModelAPI
"""

from typing import Dict, Any, Optional, Callable
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BackboneAPIHandler:
    """Handler untuk model API operations"""
    
    def __init__(self, logger_bridge: Any):
        """Initialize API handler
        
        Args:
            logger_bridge: Logger instance untuk logging
        """
        self.logger_bridge = logger_bridge
        self.model_api = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._is_building = False
    
    def initialize_api(self, config_path: Optional[str] = None) -> bool:
        """Initialize SmartCashModelAPI instance
        
        Args:
            config_path: Optional path to config file
            
        Returns:
            Success status
        """
        try:
            from smartcash.model import create_model_api
            
            self.logger_bridge.info("🔧 Initializing Model API...")
            self.model_api = create_model_api(config_path)
            self.logger_bridge.success("✅ Model API initialized")
            return True
        except Exception as e:
            self.logger_bridge.error(f"❌ Failed to initialize API: {str(e)}")
            return False
    
    def build_model_async(self, config: Dict[str, Any], 
                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Build model asynchronously dengan progress tracking
        
        Args:
            config: Model configuration
            progress_callback: Progress tracking callback
            
        Returns:
            Build result dictionary
        """
        if self._is_building:
            return {
                'success': False,
                'error': 'Model build already in progress'
            }
        
        self._is_building = True
        
        try:
            # Ensure API is initialized
            if not self.model_api:
                if not self.initialize_api():
                    return {
                        'success': False,
                        'error': 'Failed to initialize Model API'
                    }
            
            # Extract model config
            model_config = config.get('model', {})
            
            # Build model dengan API
            result = self.model_api.build_model(
                backbone=model_config.get('backbone', 'efficientnet_b4'),
                detection_layers=model_config.get('detection_layers', ['banknote']),
                layer_mode=model_config.get('layer_mode', 'single'),
                num_classes=model_config.get('num_classes', 7),
                img_size=model_config.get('img_size', 640),
                feature_optimization=model_config.get('feature_optimization', {'enabled': False}),
                progress_callback=progress_callback
            )
            
            return result
            
        except Exception as e:
            self.logger_bridge.error(f"❌ Model build failed: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            self._is_building = False
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get current model information
        
        Returns:
            Model info dictionary or None
        """
        if not self.model_api:
            self.logger_bridge.warning("⚠️ Model API not initialized")
            return None
        
        try:
            return self.model_api.get_model_info()
        except Exception as e:
            self.logger_bridge.error(f"❌ Failed to get model info: {str(e)}")
            return None
    
    def validate_model_state(self) -> tuple[bool, str]:
        """Validate current model state
        
        Returns:
            Tuple of (is_valid, message)
        """
        if not self.model_api:
            return False, "Model API not initialized"
        
        try:
            info = self.model_api.get_model_info()
            if info and info.get('model_loaded'):
                return True, "Model is ready"
            else:
                return False, "Model not built yet"
        except Exception as e:
            return False, f"Error checking model state: {str(e)}"
    
    def list_available_checkpoints(self) -> list[Dict[str, Any]]:
        """List available model checkpoints
        
        Returns:
            List of checkpoint info dictionaries
        """
        if not self.model_api:
            return []
        
        try:
            return self.model_api.list_checkpoints()
        except Exception as e:
            self.logger_bridge.error(f"❌ Failed to list checkpoints: {str(e)}")
            return []
    
    def load_checkpoint(self, checkpoint_path: str,
                       progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            progress_callback: Progress tracking callback
            
        Returns:
            Load result dictionary
        """
        if not self.model_api:
            if not self.initialize_api():
                return {
                    'success': False,
                    'error': 'Failed to initialize Model API'
                }
        
        try:
            return self.model_api.load_checkpoint(
                checkpoint_path,
                progress_callback=progress_callback
            )
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.executor:
            self.executor.shutdown(wait=False)
        self._is_building = False