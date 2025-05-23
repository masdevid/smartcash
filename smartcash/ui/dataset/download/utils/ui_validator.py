"""
File: smartcash/ui/dataset/download/utils/ui_validator.py
Deskripsi: Updated validator dengan Drive path validation
"""

from typing import Dict, Any
import os
from pathlib import Path
from smartcash.common.environment import get_environment_manager

def validate_download_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi parameter dengan Drive path checking."""
    
    params = {
        'workspace': ui_components.get('workspace', {}).value if 'workspace' in ui_components else '',
        'project': ui_components.get('project', {}).value if 'project' in ui_components else '',
        'version': ui_components.get('version', {}).value if 'version' in ui_components else '',
        'api_key': ui_components.get('api_key', {}).value if 'api_key' in ui_components else '',
        'output_dir': ui_components.get('output_dir', {}).value if 'output_dir' in ui_components else 'data'
    }
    
    # Get API key dari berbagai sumber
    if not params['api_key']:
        params['api_key'] = _get_api_key_from_sources()
    
    # Validasi field wajib
    required_fields = ['workspace', 'project', 'version', 'api_key']
    missing_fields = [field for field in required_fields if not params[field]]
    
    if missing_fields:
        return {
            'valid': False,
            'message': f"Parameter tidak lengkap: {', '.join(missing_fields)}",
            'params': params
        }
    
    # Validasi output directory
    output_validation = _validate_output_directory(params['output_dir'])
    if not output_validation['valid']:
        return {
            'valid': False,
            'message': output_validation['message'],
            'params': params
        }
    
    # Update params dengan validated output dir
    params['output_dir'] = output_validation['path']
    
    return {
        'valid': True,
        'message': f"Parameter valid - Storage: {output_validation['storage_type']}",
        'params': params
    }

def _validate_output_directory(output_dir: str) -> Dict[str, Any]:
    """Validate output directory dengan Drive awareness."""
    env_manager = get_environment_manager()
    
    try:
        output_path = Path(output_dir)
        
        # Jika Colab + Drive mounted, pastikan path ke Drive
        if env_manager.is_colab and env_manager.is_drive_mounted:
            # Jika path relatif, buat dalam Drive
            if not output_path.is_absolute():
                drive_output = env_manager.drive_path / 'downloads' / output_path
                drive_output.mkdir(parents=True, exist_ok=True)
                return {
                    'valid': True,
                    'path': str(drive_output),
                    'storage_type': 'Drive'
                }
            
            # Jika absolute path, check apakah dalam Drive
            if str(output_path).startswith('/content/drive/MyDrive'):
                output_path.mkdir(parents=True, exist_ok=True)
                return {
                    'valid': True,
                    'path': str(output_path),
                    'storage_type': 'Drive'
                }
            
            # Jika bukan Drive path, redirect ke Drive
            drive_output = env_manager.drive_path / 'downloads' / output_path.name
            drive_output.mkdir(parents=True, exist_ok=True)
            return {
                'valid': True,
                'path': str(drive_output),
                'storage_type': 'Drive (redirected)'
            }
        
        # Local environment
        output_path.mkdir(parents=True, exist_ok=True)
        return {
            'valid': True,
            'path': str(output_path),
            'storage_type': 'Local'
        }
        
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error output directory: {str(e)}",
            'storage_type': 'Unknown'
        }

def _get_api_key_from_sources() -> str:
    """Get API key dari environment atau Colab secrets."""
    # Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # Google Colab secrets
    try:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY', '')
    except:
        return ''