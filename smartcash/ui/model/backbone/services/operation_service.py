"""
File: smartcash/ui/model/backbone/services/operation_service.py
Description: UI operation service for backbone module - handles validate and build operations.
"""

from typing import Dict, Any
from smartcash.model.api.backbone_api import check_data_prerequisites, check_built_models


class BackboneOperationService:
    """
    Service class for handling UI operations in the backbone module.
    
    Focuses on orchestrating backend API calls and UI interactions
    for backbone-specific operations like validate and build.
    """
    
    def __init__(self, ui_module, logger=None):
        """
        Initialize the operation service.
        
        Args:
            ui_module: Reference to the main UI module
            logger: Logger instance for operation logging
        """
        self.ui_module = ui_module
        self.logger = logger
        self.config = getattr(ui_module, 'config', {})
    
    def validate_data_prerequisites(self) -> Dict[str, Any]:
        """
        Validate data prerequisites using backend API.
        
        Returns:
            Dict containing validation results
        """
        try:
            if self.logger:
                self.logger.info("🔍 Checking data prerequisites via backend API...")
            
            # Call backend API for data validation
            result = check_data_prerequisites()
            
            # Update UI with results
            self._update_ui_with_validation_results(result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error validating data prerequisites: {e}"
            if self.logger:
                self.logger.error(error_msg)
            return {
                'success': False,
                'prerequisites_ready': False,
                'message': error_msg
            }
    
    def rescan_built_models(self) -> Dict[str, Any]:
        """
        Rescan built models using backend API.
        
        Returns:
            Dict containing model discovery results
        """
        try:
            if self.logger:
                self.logger.info("🔍 Rescanning built models via backend API...")
            
            # Call backend API for model discovery
            result = check_built_models()
            
            # Update UI with results
            self._update_ui_with_model_results(result)
            
            return result
            
        except Exception as e:
            error_msg = f"Error rescanning built models: {e}"
            if self.logger:
                self.logger.error(error_msg)
            return {
                'success': False,
                'total_models': 0,
                'by_backbone': {},
                'discovery_summary': error_msg
            }
    
    def _update_ui_with_validation_results(self, result: Dict[str, Any]) -> None:
        """
        Update UI components with data validation results.
        
        Args:
            result: Validation results from backend API
        """
        try:
            # Update summary if available
            if hasattr(self.ui_module, 'update_summary'):
                message = result.get('message', 'Validation completed')
                self.ui_module.update_summary(f"<p>{message}</p>")
            
            # Log detailed results
            if self.logger and result.get('success'):
                pretrained = result.get('pretrained_models', {})
                raw_data = result.get('raw_data', {})
                preprocessed = result.get('preprocessed_data', {})
                
                if pretrained.get('available'):
                    self.logger.info(f"✅ Found {pretrained.get('count', 0)} pretrained models")
                
                if raw_data.get('available'):
                    self.logger.info(f"✅ Found {raw_data.get('total_images', 0)} raw images")
                
                if preprocessed.get('available'):
                    self.logger.info(f"✅ Found {preprocessed.get('total_files', 0)} preprocessed files")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update UI with validation results: {e}")
    
    def _update_ui_with_model_results(self, result: Dict[str, Any]) -> None:
        """
        Update UI components with model discovery results.
        
        Args:
            result: Model discovery results from backend API
        """
        try:
            # Update summary if available
            if hasattr(self.ui_module, 'update_summary'):
                summary = result.get('discovery_summary', 'Model scan completed')
                self.ui_module.update_summary(f"<p>{summary}</p>")
            
            # Log detailed results
            if self.logger and result.get('success'):
                total_models = result.get('total_models', 0)
                by_backbone = result.get('by_backbone', {})
                
                self.logger.info(f"✅ Found {total_models} total models")
                
                # Use summary logging to reduce noise
                total_models = sum(len(models) for models in by_backbone.values())
                if total_models > 0:
                    backbone_summary = ", ".join([f"{bt}: {len(models)}" for bt, models in by_backbone.items()])
                    self.logger.info(f"  • Found {total_models} models ({backbone_summary})")
                else:
                    self.logger.info("  • No models found")
                        
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to update UI with model results: {e}")
    
    def get_operation_status(self) -> Dict[str, Any]:
        """
        Get current operation status.
        
        Returns:
            Dict with operation status information
        """
        return {
            'service_ready': True,
            'backend_api_available': True,
            'config_loaded': bool(self.config),
            'ui_module_connected': self.ui_module is not None
        }