"""
ColabSecretsMixin - Google Colab secrets management for SmartCash UI.

Provides functionality to securely manage and access API keys and other secrets
in Google Colab environment, with fallback to environment variables.
"""

from typing import Optional, List, Dict, Any, Union
import os
import logging
from functools import lru_cache

class ColabSecretsMixin:
    """
    Mixin for managing API keys and secrets in Google Colab environment.
    
    This mixin provides methods to securely manage and access API keys and other secrets
    in Google Colab environment, with fallback to environment variables.
    
    Usage:
        class MyClass(ColabSecretsMixin):
            def __init__(self):
                super().__init__()
                self._secret_names = [
                    'MY_API_KEY',
                    'my_api_key',
                    'API_KEY',
                    'api_key'
                ]
                
            def get_my_api_key(self):
                return self.get_secret()
    """
    
    # Default secret names to try when looking for API keys
    _secret_names = [
        'ROBOFLOW_API_KEY',
        'roboflow_api_key', 
        'API_KEY',
        'api_key',
        'SMARTCASH_API_KEY',
        'smartcash_api_key',
        'HF_TOKEN'
    ]
    
    def __init__(self, *args, **kwargs):
        """Initialize the secrets mixin."""
        super().__init__(*args, **kwargs)
        self._logger = getattr(self, 'logger', logging.getLogger(__name__))
    
    def get_secret(self, secret_names: Optional[Union[str, List[str]]] = None) -> Optional[str]:
        """
        Get a secret value from available sources.
        
        Args:
            secret_names: Single secret name or list of secret names to try.
                        If not provided, uses the class's _secret_names.
                        
        Returns:
            The secret value as a string if found, None otherwise.
        """
        if secret_names is None:
            secret_names = self._secret_names
        elif isinstance(secret_names, str):
            secret_names = [secret_names]
            
        for name in secret_names:
            try:
                # Try Google Colab userdata first
                if self._is_colab():
                    from google.colab import userdata  # type: ignore
                    try:
                        return userdata.get(name)
                    except Exception as e:
                        self._logger.debug(f"Secret {name} not found in Colab userdata: {e}")
                
                # Fall back to environment variables
                if name in os.environ:
                    return os.environ[name]
                    
            except Exception as e:
                self._logger.debug(f"Error accessing secret {name}: {e}")
                continue
                
        self._logger.warning(f"No secret found in: {', '.join(secret_names)}")
        return None
    
    def validate_secret(self, secret: Optional[str] = None, secret_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate a secret or try to find a valid one.
        
        Args:
            secret: Optional secret value to validate.
            secret_names: Optional list of secret names to try if secret is not provided.
            
        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'message': str,
                'key_preview': str,
                'source': Optional[str]
            }
        """
        # If no secret provided, try to get one
        if secret is None:
            secret = self.get_secret(secret_names)
            if secret is None:
                return {
                    'valid': False,
                    'message': 'No secret found',
                    'key_preview': '****',
                    'source': None
                }
        
        # Basic validation (can be overridden by subclasses)
        is_valid = bool(secret and len(secret.strip()) > 10)  # Basic length check
        
        return {
            'valid': is_valid,
            'message': 'Secret is valid' if is_valid else 'Invalid secret',
            'key_preview': f"{secret[:2]}...{secret[-2:]}" if secret and len(secret) > 4 else '****',
            'source': 'provided' if secret else None
        }
    
    @staticmethod
    def _is_colab() -> bool:
        """Check if running in Google Colab environment.
        
        Returns:
            bool: True if running in Google Colab, False otherwise.
        """
        # Use unified detection from environment manager
        from smartcash.common.environment import is_colab_environment
        return is_colab_environment()
    
    def get_secret_info(self, secret_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get information about available secrets.
        
        Args:
            secret_names: Optional list of secret names to check.
                        
        Returns:
            Dictionary with information about available secrets.
        """
        if secret_names is None:
            secret_names = self._secret_names
            
        result = {
            'available': False,
            'sources': {
                'colab': self._is_colab(),
                'environment': True
            },
            'secrets': {}
        }
        
        for name in secret_names:
            value = self.get_secret(name)
            result['secrets'][name] = {
                'available': value is not None,
                'preview': f"{value[:2]}...{value[-2:]}" if value and len(value) > 4 else '****',
                'length': len(value) if value else 0
            }
            
            if value is not None:
                result['available'] = True
                
        return result
        
    def create_api_key_info_html(self, config: Dict[str, Any]) -> str:
        """
        Create HTML to display API key status.
        
        Args:
            config: Configuration dictionary containing API key information
            
        Returns:
            HTML string displaying the API key status
        """
        try:
            roboflow_config = config.get('data', {}).get('roboflow', {})
            api_key = roboflow_config.get('api_key', '')
            
            if api_key:
                key_preview = f"{api_key[:2]}...{api_key[-2:]}" if len(api_key) > 4 else '****'
                return (
                    f'<div style="padding: 8px; background: #e8f5e9; border-radius: 4px; margin: 6px 0;">'
                    f'<small style="color: #2e7d32;"><strong>🔑 API Key:</strong> {key_preview} (configured)</small>'
                    '</div>'
                )
            else:
                return (
                    '<div style="padding: 8px; background: #fff3e0; border-radius: 4px; margin: 6px 0;">'
                    '<small style="color: #e65100;"><strong>⚠️ API Key:</strong> Not configured</small>'
                    '</div>'
                )
                
        except Exception as e:
            self._logger.error(f"Error creating API key info HTML: {e}")
            return (
                '<div style="padding: 8px; background: #ffebee; border-radius: 4px; margin: 6px 0;">'
                '<small style="color: #c62828;"><strong>❌ Error:</strong> Failed to load API key info</small>'
                '</div>'
            )
