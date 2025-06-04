"""
File: smartcash/ui/dataset/download/utils/parameter_extractor.py
Deskripsi: Utility untuk ekstrak dan validasi parameter download dengan enhanced processing
"""

from typing import Dict, Any, Optional
import os
import re

def extract_download_parameters(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameter download dari UI components dengan comprehensive mapping."""
    params = {}
    
    # Field mapping dengan default values
    field_mapping = {
        'workspace': ('workspace', 'smartcash-wo2us'),
        'project': ('project', 'rupiah-emisi-2022'), 
        'version': ('version', '3'),
        'api_key': ('api_key', ''),
        'output_dir': ('output_dir', 'data/downloads'),
        'backup_dir': ('backup_dir', 'data/backup'),
        'backup_before_download': ('backup_checkbox', False),
        'organize_dataset': ('organize_dataset', True)
    }
    
    # Extract dengan fallback values
    [params.update({param_key: _get_widget_value(ui_components, widget_key, default)})
     for param_key, (widget_key, default) in field_mapping.items()]
    
    # Auto-detect API key jika kosong
    if not params.get('api_key'):
        params['api_key'] = _detect_api_key_from_environment()
    
    # Normalize paths
    params['output_dir'] = _normalize_path(params['output_dir'])
    params['backup_dir'] = _normalize_path(params['backup_dir'])
    
    return params

def validate_extracted_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate extracted parameters dengan comprehensive checks."""
    validation_result = {'valid': True, 'errors': [], 'warnings': []}
    
    # Required field validation
    required_fields = [('workspace', 'Workspace'), ('project', 'Project'), ('version', 'Version')]
    missing_required = [(field, name) for field, name in required_fields if not params.get(field, '').strip()]
    validation_result['errors'].extend([f"{name} wajib diisi" for _, name in missing_required])
    
    # Format validation dengan one-liner regex
    format_validators = [
        ('workspace', r'^[a-zA-Z0-9\-_]{3,}$', 'Workspace format tidak valid (min 3 karakter, alphanumeric, dash, underscore)'),
        ('project', r'^[a-zA-Z0-9\-_]{3,}$', 'Project format tidak valid (min 3 karakter, alphanumeric, dash, underscore)'),
        ('version', r'^[a-zA-Z0-9\.\-_]+$', 'Version format tidak valid')
    ]
    
    [validation_result['errors'].append(error_msg) 
     for field, pattern, error_msg in format_validators 
     if params.get(field) and not re.match(pattern, params[field])]
    
    # API key validation
    api_key = params.get('api_key', '').strip()
    if not api_key:
        validation_result['warnings'].append('API key tidak ditemukan - pastikan sudah diset di environment')
    elif len(api_key) < 10:
        validation_result['errors'].append('API key terlalu pendek (minimal 10 karakter)')
    
    # Path validation
    path_validators = [('output_dir', 'Output directory'), ('backup_dir', 'Backup directory')]
    [validation_result['warnings'].append(f"{name} path mungkin tidak valid: {params.get(field)}")
     for field, name in path_validators if params.get(field) and not _is_valid_path(params[field])]
    
    validation_result['valid'] = len(validation_result['errors']) == 0
    return validation_result

def _get_widget_value(ui_components: Dict[str, Any], widget_key: str, default: Any) -> Any:
    """Get widget value dengan fallback ke default."""
    widget = ui_components.get(widget_key)
    if widget and hasattr(widget, 'value'):
        value = widget.value
        return value.strip() if isinstance(value, str) else value
    return default

def _detect_api_key_from_environment() -> str:
    """Detect API key dari berbagai environment sources dengan one-liner fallback."""
    # Environment variables dengan priority order
    env_keys = ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY', 'API_KEY']
    api_key = next((os.environ.get(key, '').strip() for key in env_keys if os.environ.get(key, '').strip()), '')
    
    if api_key and len(api_key) > 10:
        return api_key
    
    # Google Colab userdata fallback
    try:
        from google.colab import userdata
        colab_keys = ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 'API_KEY']
        return next((userdata.get(key, '').strip() for key in colab_keys 
                    if _safe_userdata_get(userdata, key)), '')
    except ImportError:
        return ''

def _safe_userdata_get(userdata, key: str) -> str:
    """Safe userdata get dengan error handling."""
    try:
        value = userdata.get(key, '').strip()
        return value if len(value) > 10 else ''
    except Exception:
        return ''

def _normalize_path(path: str) -> str:
    """Normalize path dengan consistent format."""
    if not path:
        return ''
    
    # Remove extra spaces and normalize separators
    normalized = path.strip().replace('\\', '/')
    
    # Handle relative paths
    if not os.path.isabs(normalized) and not normalized.startswith('./'):
        normalized = normalized.lstrip('./')
    
    return normalized

def _is_valid_path(path: str) -> bool:
    """Validate path format dengan comprehensive checks."""
    if not path or not isinstance(path, str):
        return False
    
    # Basic format validation
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in path for char in invalid_chars):
        return False
    
    # Length validation
    if len(path) > 260:  # Windows MAX_PATH limitation
        return False
    
    return True

def get_parameter_summary(params: Dict[str, Any]) -> str:
    """Get human-readable parameter summary."""
    workspace = params.get('workspace', 'N/A')
    project = params.get('project', 'N/A')
    version = params.get('version', 'N/A')
    has_api_key = bool(params.get('api_key', '').strip())
    
    return f"📊 {workspace}/{project}:{version} | API: {'✅' if has_api_key else '❌'}"

def sanitize_roboflow_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize parameters untuk Roboflow API compatibility."""
    sanitized = params.copy()
    
    # Lowercase dan replace spaces/underscores dengan dash untuk workspace/project
    for field in ['workspace', 'project']:
        if sanitized.get(field):
            sanitized[field] = sanitized[field].lower().replace(' ', '-').replace('_', '-')
    
    # Normalize version
    if sanitized.get('version'):
        sanitized['version'] = str(sanitized['version']).strip()
    
    # Ensure API key tidak ada leading/trailing spaces
    if sanitized.get('api_key'):
        sanitized['api_key'] = sanitized['api_key'].strip()
    
    return sanitized

def create_download_url(params: Dict[str, Any], format_type: str = 'yolov5pytorch') -> str:
    """Create download URL dari parameters."""
    sanitized = sanitize_roboflow_parameters(params)
    
    workspace = sanitized.get('workspace', '')
    project = sanitized.get('project', '')
    version = sanitized.get('version', '')
    api_key = sanitized.get('api_key', '')
    
    if not all([workspace, project, version, api_key]):
        raise ValueError("Missing required parameters untuk create download URL")
    
    return f"https://api.roboflow.com/{workspace}/{project}/{version}/{format_type}?api_key={api_key}"

def validate_parameter_combination(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameter combination untuk specific scenarios."""
    result = {'valid': True, 'warnings': [], 'recommendations': []}
    
    # Check backup scenario
    if params.get('backup_before_download', False) and not params.get('backup_dir'):
        result['warnings'].append('Backup enabled tapi backup directory tidak diset')
        result['recommendations'].append('Set backup directory atau disable backup option')
    
    # Check output directory same as backup
    if (params.get('output_dir') and params.get('backup_dir') and 
        _normalize_path(params['output_dir']) == _normalize_path(params['backup_dir'])):
        result['warnings'].append('Output dan backup directory sama')
        result['recommendations'].append('Gunakan directory berbeda untuk output dan backup')
    
    # Check workspace/project combination validity
    workspace = params.get('workspace', '')
    project = params.get('project', '')
    if workspace and project and workspace == project:
        result['warnings'].append('Workspace dan project name sama - pastikan ini benar')
    
    return result