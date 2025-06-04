"""
File: smartcash/ui/dataset/download/utils/parameter_extractor.py
Deskripsi: Parameter extractor tanpa opsi validasi dataset
"""

from typing import Dict, Any, Optional, Union
import os

def extract_download_parameters(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameter download dari UI components tanpa validate_dataset."""
    
    # Basic parameter extraction
    params = {
        'workspace': safe_extract_text(ui_components, 'workspace'),
        'project': safe_extract_text(ui_components, 'project'),
        'version': safe_extract_text(ui_components, 'version'),
        'api_key': safe_extract_text(ui_components, 'api_key'),
        'output_dir': safe_extract_text(ui_components, 'output_dir'),
        'backup_dir': safe_extract_text(ui_components, 'backup_dir'),
    }
    
    # Boolean parameters (tanpa validate_dataset)
    params.update({
        'backup_before_download': safe_extract_boolean(ui_components, 'backup_checkbox', False),
        'organize_dataset': safe_extract_boolean(ui_components, 'organize_dataset', True),
    })
    
    # Auto-detect API key jika kosong
    if not params['api_key']:
        params['api_key'] = detect_api_key_from_environment()
    
    # Clean dan validate parameters
    params = clean_parameters(params)
    
    return params

def safe_extract_text(ui_components: Dict[str, Any], key: str, default: str = '') -> str:
    """Safely extract text value dari UI component."""
    try:
        if key not in ui_components:
            return default
        
        component = ui_components[key]
        if component is None or not hasattr(component, 'value'):
            return default
        
        raw_value = getattr(component, 'value', default)
        if raw_value is None:
            return default
        
        cleaned_value = str(raw_value).strip()
        
        # Additional cleaning untuk specific fields
        if key in ['workspace', 'project']:
            cleaned_value = clean_identifier(cleaned_value)
        elif key == 'version':
            cleaned_value = clean_version(cleaned_value)
        elif key in ['output_dir', 'backup_dir']:
            cleaned_value = clean_path(cleaned_value)
        
        return cleaned_value if cleaned_value else default
        
    except Exception:
        return default

def safe_extract_boolean(ui_components: Dict[str, Any], key: str, default: bool = False) -> bool:
    """Safely extract boolean value dari UI component."""
    try:
        if key not in ui_components or ui_components[key] is None:
            return default
        
        component = ui_components[key]
        if not hasattr(component, 'value'):
            return default
        
        raw_value = getattr(component, 'value', default)
        
        if isinstance(raw_value, bool):
            return raw_value
        elif isinstance(raw_value, str):
            return raw_value.lower() in ['true', '1', 'yes', 'on']
        elif isinstance(raw_value, (int, float)):
            return bool(raw_value)
        else:
            return default
            
    except Exception:
        return default

def clean_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Clean dan normalize parameter values."""
    cleaned = params.copy()
    
    # Clean workspace dan project (lowercase, no spaces)
    for key in ['workspace', 'project']:
        if cleaned.get(key):
            cleaned[key] = cleaned[key].lower().replace(' ', '-').replace('_', '-')
    
    # Clean version (ensure numeric format)
    if cleaned.get('version'):
        cleaned['version'] = clean_version_number(cleaned['version'])
    
    # Clean paths
    for key in ['output_dir', 'backup_dir']:
        if cleaned.get(key):
            cleaned[key] = normalize_path(cleaned[key])
    
    # Validate API key format
    if cleaned.get('api_key'):
        cleaned['api_key'] = clean_api_key(cleaned['api_key'])
    
    return cleaned

def clean_identifier(value: str) -> str:
    """Clean identifier untuk workspace/project names."""
    if not value:
        return ''
    
    import re
    cleaned = re.sub(r'[^a-zA-Z0-9\-_]', '', value)
    cleaned = cleaned.strip('-_')
    return cleaned.lower()

def clean_version(value: str) -> str:
    """Clean version string."""
    if not value:
        return ''
    
    import re
    numeric_match = re.search(r'\d+', value)
    return numeric_match.group() if numeric_match else value

def clean_version_number(value: str) -> str:
    """Ensure version is in proper numeric format."""
    try:
        version_int = int(float(value))
        return str(version_int)
    except (ValueError, TypeError):
        return str(value)

def clean_path(value: str) -> str:
    """Clean filesystem path."""
    if not value:
        return ''
    
    import os
    normalized = os.path.normpath(value.replace('\\', '/'))
    while '//' in normalized:
        normalized = normalized.replace('//', '/')
    return normalized

def normalize_path(path: str) -> str:
    """Normalize path dengan OS-specific handling."""
    if not path:
        return ''
    
    import os
    from pathlib import Path
    
    try:
        normalized = str(Path(path).resolve())
        return normalized
    except Exception:
        return clean_path(path)

def clean_api_key(api_key: str) -> str:
    """Clean dan validate API key format."""
    if not api_key:
        return ''
    
    cleaned = api_key.strip()
    
    import re
    if not re.match(r'^[a-zA-Z0-9\-_]+$', cleaned):
        return cleaned
    
    return cleaned

def detect_api_key_from_environment() -> str:
    """Detect API key dari berbagai sumber environment."""
    # Primary environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return clean_api_key(api_key)
    
    # Alternative environment variable names
    alternative_env_vars = [
        'ROBOFLOW_KEY',
        'RF_API_KEY',
        'API_KEY_ROBOFLOW'
    ]
    
    for env_var in alternative_env_vars:
        api_key = os.environ.get(env_var, '')
        if api_key:
            return clean_api_key(api_key)
    
    # Google Colab userdata
    try:
        from google.colab import userdata
        
        # Primary userdata key
        try:
            api_key = userdata.get('ROBOFLOW_API_KEY')
            if api_key:
                return clean_api_key(api_key)
        except Exception:
            pass
        
        # Alternative userdata keys
        alternative_userdata_keys = [
            'roboflow_api_key',
            'ROBOFLOW_KEY',
            'roboflow_key',
            'RF_API_KEY',
            'API_KEY'
        ]
        
        for key_name in alternative_userdata_keys:
            try:
                api_key = userdata.get(key_name)
                if api_key:
                    return clean_api_key(api_key)
            except Exception:
                continue
                
    except ImportError:
        pass
    except Exception:
        pass
    
    return ''

def validate_extracted_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted parameters dan return validation result."""
    errors = []
    warnings = []
    
    # Required field validation
    required_fields = {
        'workspace': 'Workspace ID',
        'project': 'Project ID',
        'version': 'Dataset Version',
        'api_key': 'API Key'
    }
    
    for field, display_name in required_fields.items():
        if not params.get(field):
            errors.append(f"{display_name} tidak boleh kosong")
    
    # Format validation
    if params.get('workspace') and len(params['workspace']) < 3:
        warnings.append("Workspace ID sangat pendek, pastikan benar")
    
    if params.get('project') and len(params['project']) < 3:
        warnings.append("Project ID sangat pendek, pastikan benar")
    
    if params.get('api_key') and len(params['api_key']) < 10:
        errors.append("API Key terlalu pendek, periksa kembali")
    
    # Path validation
    if params.get('output_dir'):
        try:
            from pathlib import Path
            Path(params['output_dir'])
        except Exception:
            errors.append("Output directory path tidak valid")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'error_count': len(errors),
        'warning_count': len(warnings)
    }