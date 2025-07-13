"""
File: smartcash/ui/dataset/downloader/services/utils/secret_manager.py
Description: Service for managing API keys and secrets.
"""

from typing import Optional, List, Dict, Any

from smartcash.ui.dataset.downloader.services.core.base_service import BaseService

class SecretManagerService(BaseService):
    """Service for managing API keys and secrets."""
    
    def __init__(self, logger=None):
        """Initialize secret manager service.
        
        Args:
            logger: Optional logger instance
        """
        super().__init__(logger)
        self._secret_names = [
            'ROBOFLOW_API_KEY',
            'roboflow_api_key', 
            'API_KEY',
            'api_key',
            'SMARTCASH_API_KEY',
            'smartcash_api_key'
        ]
    
    def get_api_key(self, secret_names: Optional[List[str]] = None) -> Optional[str]:
        """Get API key from available sources.
        
        Args:
            secret_names: Optional list of secret names to try
            
        Returns:
            API key string or None if not found
        """
        secret_names = secret_names or self._secret_names
        
        for secret_name in secret_names:
            try:
                api_key = self._get_secret(secret_name)
                if api_key and api_key.strip():
                    return api_key.strip()
            except Exception:
                continue
        
        return None
    
    def _get_secret(self, secret_name: str) -> Optional[str]:
        """Get a secret by name from available sources.
        
        Args:
            secret_name: Name of the secret to retrieve
            
        Returns:
            Secret value or None if not found
        """
        try:
            # Try Google Colab first
            from google.colab import userdata
            return userdata.get(secret_name)
        except ImportError:
            # Fall back to environment variables if not in Colab
            import os
            return os.environ.get(secret_name)
        except Exception as e:
            self.log_error(f"Error getting secret '{secret_name}': {e}")
            return None
    
    def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            Dictionary with validation results
        """
        if not api_key or not isinstance(api_key, str) or len(api_key.strip()) < 10:
            return {
                'valid': False,
                'message': 'API key is invalid or too short',
                'key_preview': '****'
            }
            
        return {
            'valid': True,
            'message': 'API key is valid',
            'key_preview': f"{api_key[:4]}{'*' * (len(api_key) - 8)}{api_key[-4:]}" if len(api_key) > 8 else '****'
        }
    
    def create_api_key_info_html(self, config: Dict[str, Any]) -> str:
        """Create HTML info for API key status.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            HTML string with API key information
        """
        api_key = config.get('data', {}).get('roboflow', {}).get('api_key', '')
        
        if api_key:
            return f"""
            <div style="color: green; font-size: 12px; margin-top: 5px;">
                ✅ API Key loaded (first 4 chars: {api_key[:4]}...)
            </div>
            """
        else:
            return """
            <div style="color: orange; font-size: 12px; margin-top: 5px;">
                ⚠️ No API key found. Please enter your Roboflow API key.
            </div>
            """
