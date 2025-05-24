"""
File: smartcash/ui/dataset/download/handlers/validation_handler.py
Deskripsi: Handler khusus untuk validasi parameter download dengan error handling yang robust
"""

import os
from typing import Dict, Any
from pathlib import Path
from smartcash.common.environment import get_environment_manager

def validate_download_parameters(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi lengkap parameter download dengan error handling yang robust.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict berisi validasi result dan parameter yang sudah dibersihkan
    """
    try:
        # Extract parameters dengan safe method
        params = _extract_safe_parameters(ui_components)
        
        # Validasi required fields
        validation_errors = _validate_required_fields(params)
        if validation_errors:
            return {
                'valid': False,
                'message': f"Parameter tidak lengkap: {', '.join(validation_errors)}",
                'params': params
            }
        
        # Validasi API key
        api_validation = _validate_api_key(params['api_key'])
        if not api_validation['valid']:
            return {
                'valid': False,
                'message': api_validation['message'],
                'params': params
            }
        
        # Validasi dan setup output directory
        output_validation = _validate_output_directory(params['output_dir'])
        if not output_validation['valid']:
            return {
                'valid': False,
                'message': output_validation['message'],
                'params': params
            }
        
        # Update params dengan validated output directory
        params['output_dir'] = output_validation['path']
        
        return {
            'valid': True,
            'message': f"✅ Parameter valid - Storage: {output_validation['storage_type']}",
            'params': params
        }
        
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error validasi: {str(e)}",
            'params': {}
        }

def _extract_safe_parameters(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameters dari UI components dengan safe method."""
    params = {}
    
    # Field mapping
    field_mapping = {
        'workspace': 'workspace',
        'project': 'project',
        'version': 'version',
        'api_key': 'api_key',
        'output_dir': 'output_dir'
    }
    
    # Safe extraction
    for param_key, ui_key in field_mapping.items():
        params[param_key] = _safe_get_ui_value(ui_components, ui_key, '')
    
    # Get API key dari berbagai sumber jika kosong
    if not params['api_key']:
        params['api_key'] = _detect_api_key_from_sources()
    
    return params

def _safe_get_ui_value(ui_components: Dict[str, Any], key: str, default: str = '') -> str:
    """Safely extract value dari UI component."""
    try:
        if key not in ui_components or ui_components[key] is None:
            return default
        
        component = ui_components[key]
        if not hasattr(component, 'value'):
            return default
        
        value = getattr(component, 'value', default)
        return str(value).strip() if value is not None else default
        
    except Exception:
        return default

def _validate_required_fields(params: Dict[str, Any]) -> list:
    """Validasi required fields dan return list field yang missing."""
    required_fields = ['workspace', 'project', 'version', 'api_key', 'output_dir']
    return [field for field in required_fields if not params.get(field)]

def _validate_api_key(api_key: str) -> Dict[str, Any]:
    """Validasi API key dengan berbagai checks."""
    if not api_key:
        return {
            'valid': False,
            'message': "API key tidak ditemukan. Set di environment variable ROBOFLOW_API_KEY atau Colab secrets"
        }
    
    if len(api_key.strip()) < 10:
        return {
            'valid': False,
            'message': "API key terlalu pendek. Periksa kembali API key Roboflow Anda"
        }
    
    # Basic format validation (Roboflow API keys biasanya alphanumeric)
    if not api_key.replace('-', '').replace('_', '').isalnum():
        return {
            'valid': False,
            'message': "Format API key tidak valid. API key harus berupa alphanumeric"
        }
    
    return {'valid': True, 'message': "API key valid"}

def _validate_output_directory(output_dir: str) -> Dict[str, Any]:
    """Validasi output directory dengan Drive awareness."""
    env_manager = get_environment_manager()
    
    try:
        output_path = Path(output_dir)
        
        # Jika Colab + Drive mounted, pastikan path ke Drive
        if env_manager.is_colab and env_manager.is_drive_mounted:
            return _validate_drive_path(output_path, env_manager)
        
        # Local environment validation
        return _validate_local_path(output_path)
        
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error validasi output directory: {str(e)}",
            'storage_type': 'Unknown'
        }

def _validate_drive_path(output_path: Path, env_manager) -> Dict[str, Any]:
    """Validasi path untuk Drive storage."""
    try:
        # Path relatif -> buat dalam Drive
        if not output_path.is_absolute():
            drive_output = env_manager.drive_path / 'downloads' / output_path
            drive_output.mkdir(parents=True, exist_ok=True)
            return {
                'valid': True,
                'path': str(drive_output),
                'storage_type': 'Google Drive'
            }
        
        # Path absolute dalam Drive
        if str(output_path).startswith('/content/drive/MyDrive'):
            output_path.mkdir(parents=True, exist_ok=True)
            return {
                'valid': True,
                'path': str(output_path),
                'storage_type': 'Google Drive'
            }
        
        # Path bukan Drive -> redirect ke Drive
        drive_output = env_manager.drive_path / 'downloads' / output_path.name
        drive_output.mkdir(parents=True, exist_ok=True)
        return {
            'valid': True,
            'path': str(drive_output),
            'storage_type': 'Google Drive (redirected)'
        }
        
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error setup Drive path: {str(e)}",
            'storage_type': 'Drive Error'
        }

def _validate_local_path(output_path: Path) -> Dict[str, Any]:
    """Validasi path untuk local storage."""
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        return {
            'valid': True,
            'path': str(output_path),
            'storage_type': 'Local Storage'
        }
    except Exception as e:
        return {
            'valid': False,
            'message': f"Error setup local path: {str(e)}",
            'storage_type': 'Local Error'
        }

def _detect_api_key_from_sources() -> str:
    """Deteksi API key dari environment variable atau Colab secrets."""
    # Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # Google Colab secrets
    try:
        from google.colab import userdata
        
        # Try primary key name
        try:
            return userdata.get('ROBOFLOW_API_KEY', '')
        except Exception:
            pass
        
        # Try alternative key names
        alternative_keys = ['roboflow_api_key', 'ROBOFLOW_KEY', 'roboflow_key', 'API_KEY']
        for key_name in alternative_keys:
            try:
                api_key = userdata.get(key_name, '')
                if api_key:
                    return api_key
            except Exception:
                continue
                
    except ImportError:
        pass
    except Exception:
        pass
    
    return ''