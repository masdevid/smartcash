"""
File: smartcash/ui/dataset/downloader/services/validators/config_validator.py
Description: Service for validating downloader configurations.
"""

import re
from typing import Dict, Any, List, Tuple, Optional

from smartcash.ui.dataset.downloader.services.core.base_service import BaseService

class ConfigValidatorService(BaseService):
    """Service for validating downloader configurations."""
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate downloader configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Dictionary with validation results
        """
        try:
            if not config or not isinstance(config, dict):
                return {
                    'valid': False,
                    'errors': ['Configuration must be a non-empty dictionary']
                }
                
            # Validate Roboflow config if present
            if 'data' in config and 'roboflow' in config['data']:
                roboflow_result = self.validate_roboflow_config(config)
                if not roboflow_result.get('valid', False):
                    return roboflow_result
            
            # Add additional validations here as needed
            
            return {'valid': True, 'warnings': []}
            
        except Exception as e:
            self.log_error(f"Error validating config: {e}")
            return {
                'valid': False,
                'errors': [f'Error validating config: {str(e)}']
            }
    
    def validate_roboflow_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Roboflow configuration.
        
        Args:
            config: Configuration dictionary containing Roboflow settings
            
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Extract roboflow config with safe access
        roboflow = config.get('data', {}).get('roboflow', {})
        
        # Required fields validation with strip to avoid whitespace issues
        required_fields = {
            'workspace': roboflow.get('workspace', '').strip(),
            'project': roboflow.get('project', '').strip(),
            'version': roboflow.get('version', '').strip(),
            'api_key': roboflow.get('api_key', '').strip()
        }
        
        # Check missing required fields
        errors.extend([f"Field '{field}' is required" for field, value in required_fields.items() if not value])
        
        # Format validation with conditional checks
        if required_fields['workspace'] and len(required_fields['workspace']) < 3:
            errors.append("Workspace must be at least 3 characters")
        
        if required_fields['project'] and len(required_fields['project']) < 3:
            errors.append("Project must be at least 3 characters")
        
        if required_fields['api_key'] and len(required_fields['api_key']) < 10:
            errors.append("API key is too short (minimum 10 characters)")
        
        # Format validation for version
        if required_fields['version'] and not required_fields['version'].isdigit():
            warnings.append("Version should be a number")
        
        # Mask API key for security
        api_key = required_fields['api_key']
        api_key_masked = f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if len(api_key) > 8 else '****'
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'api_key_masked': api_key_masked
        }
