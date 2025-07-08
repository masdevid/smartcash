"""
File: smartcash/ui/model/pretrained/handlers/pretrained_ui_handler.py
Main UI handler for pretrained models module.
"""

import asyncio
from typing import Dict, Any, Optional

from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from ..operations.download_operation import DownloadOperation
from ..services.pretrained_service import PretrainedService
from ..constants import PretrainedOperation, DEFAULT_CONFIG


class PretrainedUIHandler(ModuleUIHandler):
    """
    Main UI handler for pretrained models module.
    Manages single-button download operation for YOLOv5s and EfficientNet-B4.
    """
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Initialize the pretrained UI handler.
        
        Args:
            ui_components: Dictionary containing UI components
        """
        super().__init__(module_name="pretrained", parent_module="model")
        self.ui_components = ui_components
        self.download_operation = DownloadOperation()
        self.service = PretrainedService()
        self._setup_handlers()
    
    def initialize(self) -> Dict[str, Any]:
        """
        Initialize the UI handler (required by base class).
        
        Returns:
            Dictionary containing initialization result
        """
        return {
            'success': True,
            'ui_handler': self.__class__.__name__,
            'module': self.module_name
        }
    
    def _setup_handlers(self) -> None:
        """Setup event handlers for UI components."""
        # Setup download button handler
        download_button = self.ui_components.get('download_button')
        if download_button:
            download_button.on_click(self._handle_download_click)
    
    def _handle_download_click(self, button) -> None:
        """
        Handle download button click event.
        
        Args:
            button: The button widget that was clicked
        """
        try:
            # Disable button during operation
            button.disabled = True
            button.description = "🔄 Downloading..."
            
            # Get configuration from UI
            config = self._extract_config_from_ui()
            
            # Run download operation
            asyncio.create_task(self._run_download_operation(config, button))
            
        except Exception as e:
            self._handle_operation_error(f"Failed to start download: {str(e)}", button)
    
    async def _run_download_operation(self, config: Dict[str, Any], button) -> None:
        """
        Run the download operation asynchronously.
        
        Args:
            config: Configuration for the download operation
            button: The download button to re-enable after operation
        """
        try:
            # Execute download operation
            results = await self.download_operation.execute_operation(
                config, self.ui_components
            )
            
            # Handle operation results
            if results.get("success", False):
                self._handle_operation_success(results, button)
            else:
                error_msg = results.get("error", "Unknown error during download")
                self._handle_operation_error(error_msg, button)
                
        except Exception as e:
            self._handle_operation_error(f"Download operation failed: {str(e)}", button)
    
    def _extract_config_from_ui(self) -> Dict[str, Any]:
        """
        Extract configuration from UI components.
        
        Returns:
            Dictionary containing configuration for download operation
        """
        # Get input components
        input_options = self.ui_components.get('input_options', {})
        
        # Extract values from input widgets
        config = DEFAULT_CONFIG.copy()
        
        # Get models directory
        model_dir_input = input_options.get('model_dir_input')
        if model_dir_input and hasattr(model_dir_input, 'value'):
            config['models_dir'] = model_dir_input.value or config['models_dir']
        
        # Get custom URLs
        yolo_url_input = input_options.get('yolo_url_input')
        efficientnet_url_input = input_options.get('efficientnet_url_input')
        
        custom_urls = {}
        if yolo_url_input and hasattr(yolo_url_input, 'value') and yolo_url_input.value.strip():
            custom_urls['yolov5s'] = yolo_url_input.value.strip()
            
        if efficientnet_url_input and hasattr(efficientnet_url_input, 'value') and efficientnet_url_input.value.strip():
            custom_urls['efficientnet_b4'] = efficientnet_url_input.value.strip()
        
        if custom_urls:
            config['model_urls'].update(custom_urls)
        
        return config
    
    def _handle_operation_success(self, results: Dict[str, Any], button) -> None:
        """
        Handle successful operation completion.
        
        Args:
            results: Results from the download operation
            button: The download button to re-enable
        """
        try:
            # Update button state
            button.disabled = False
            button.description = "✅ Download Complete"
            
            # Log success message
            log_output = self.ui_components.get('log_output')
            if log_output and hasattr(log_output, 'log'):
                summary = results.get('summary', {})
                models_count = summary.get('models_count', 0)
                total_size = summary.get('total_size_mb', 0)
                log_output.log(f"🎉 Download completed! {models_count} models downloaded ({total_size:.1f} MB total)")
            
            # Reset button text after delay
            async def reset_button():
                await asyncio.sleep(3)
                if not button.disabled:
                    button.description = "📥 Download Models"
            
            asyncio.create_task(reset_button())
            
        except Exception as e:
            self._handle_operation_error(f"Error handling success: {str(e)}", button)
    
    def _handle_operation_error(self, error_msg: str, button) -> None:
        """
        Handle operation error.
        
        Args:
            error_msg: Error message to display
            button: The download button to re-enable
        """
        try:
            # Update button state
            button.disabled = False
            button.description = "❌ Download Failed"
            
            # Log error message
            log_output = self.ui_components.get('log_output')
            if log_output and hasattr(log_output, 'log'):
                log_output.log(f"❌ {error_msg}")
            
            # Reset button text after delay
            async def reset_button():
                await asyncio.sleep(3)
                if not button.disabled:
                    button.description = "📥 Download Models"
            
            asyncio.create_task(reset_button())
            
        except Exception as e:
            # Fallback error handling
            button.disabled = False
            button.description = "📥 Download Models"
    
    async def check_models_status(self) -> Dict[str, Any]:
        """
        Check the status of pretrained models (for post-init checks).
        
        Returns:
            Dictionary with models status information
        """
        try:
            config = self._extract_config_from_ui()
            models_dir = config.get('models_dir', '/data/pretrained')
            
            return await self.service.check_existing_models(models_dir)
            
        except Exception as e:
            return {
                "error": str(e),
                "models_found": [],
                "models_missing": [],
                "total_found": 0,
                "all_present": False
            }