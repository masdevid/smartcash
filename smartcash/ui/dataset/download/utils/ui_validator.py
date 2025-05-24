"""
File: smartcash/ui/dataset/download/utils/ui_validator.py
Deskripsi: Fixed validator dengan None-safe parameter extraction
"""

from typing import Dict, Any
import os
from pathlib import Path
from smartcash.common.environment import get_environment_manager

def validate_download_params(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validasi parameter dengan None-safe extraction."""
    
    # Safe parameter extraction dengan fallback ke empty string
    params = {
        'workspace': _safe_get_value(ui_components, 'workspace', ''),
        'project': _safe_get_value(ui_components, 'project', ''),
        'version': _safe_get_value(ui_components, 'version', ''),
        'api_key': _safe_get_value(ui_components, 'api_key', ''),
        'output_dir': _safe_get_value(ui_components, 'output_dir', 'data')
    }
    
    # Get API key dari berbagai sumber jika kosong
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

def _safe_get_value(ui_components: Dict[str, Any], key: str, default: str = '') -> str:
    """Safely extract value dari UI component dengan fallback."""
    try:
        # Check if component exists
        if key not in ui_components:
            return default
        
        component = ui_components[key]
        
        # Check if component is None
        if component is None:
            return default
        
        # Check if component has value attribute
        if not hasattr(component, 'value'):
            return default
        
        # Get value dengan fallback
        value = getattr(component, 'value', default)
        
        # Ensure return string dan strip whitespace
        if value is None:
            return default
        
        return str(value).strip()
        
    except Exception:
        return default

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

def validate_ui_components_structure(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate struktur UI components sebelum digunakan."""
    required_components = [
        'workspace', 'project', 'version', 'api_key', 'output_dir'
    ]
    
    missing_components = []
    none_components = []
    invalid_components = []
    
    for component_key in required_components:
        if component_key not in ui_components:
            missing_components.append(component_key)
        elif ui_components[component_key] is None:
            none_components.append(component_key)
        elif not hasattr(ui_components[component_key], 'value'):
            invalid_components.append(component_key)
    
    issues = []
    if missing_components:
        issues.append(f"Missing: {', '.join(missing_components)}")
    if none_components:
        issues.append(f"None: {', '.join(none_components)}")
    if invalid_components:
        issues.append(f"Invalid: {', '.join(invalid_components)}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'missing_components': missing_components,
        'none_components': none_components,
        'invalid_components': invalid_components
    }