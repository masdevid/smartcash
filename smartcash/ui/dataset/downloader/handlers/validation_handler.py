"""
File: smartcash/ui/dataset/downloader/handlers/validation_handler.py
Deskripsi: Validation handler untuk form validation tanpa post-download actions
"""

from typing import Dict, Any, Tuple
from smartcash.ui.dataset.downloader.utils.colab_secrets import validate_api_key

def setup_validation_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> None:
    """Setup real-time form validation untuk downloader UI"""
    # Setup validation untuk form fields
    _setup_form_field_validation(ui_components, logger)
    logger.debug("✅ Validation handler siap")

def _setup_form_field_validation(ui_components: Dict[str, Any], logger) -> None:
    """Setup real-time validation untuk form fields"""
    
    # Workspace validation
    workspace_input = ui_components.get('workspace_input')
    if workspace_input:
        workspace_input.observe(_create_workspace_validator(logger), names='value')
    
    # Project validation  
    project_input = ui_components.get('project_input')
    if project_input:
        project_input.observe(_create_project_validator(logger), names='value')
    
    # Version validation
    version_input = ui_components.get('version_input')
    if version_input:
        version_input.observe(_create_version_validator(logger), names='value')
    
    # API key validation
    api_key_input = ui_components.get('api_key_input')
    if api_key_input:
        api_key_input.observe(_create_api_key_validator(logger), names='value')

def _create_workspace_validator(logger):
    """Create workspace field validator dengan real-time feedback"""
    def validate_workspace(change):
        value = change['new'].strip()
        is_valid, message = validate_workspace_format(value)
        
        if value and not is_valid:
            logger.warning(f"⚠️ Workspace: {message}")
    
    return validate_workspace

def _create_project_validator(logger):
    """Create project field validator dengan real-time feedback"""
    def validate_project(change):
        value = change['new'].strip()
        is_valid, message = validate_project_format(value)
        
        if value and not is_valid:
            logger.warning(f"⚠️ Project: {message}")
    
    return validate_project

def _create_version_validator(logger):
    """Create version field validator dengan real-time feedback"""
    def validate_version(change):
        value = change['new'].strip()
        is_valid, message = validate_version_format(value)
        
        if value and not is_valid:
            logger.warning(f"⚠️ Version: {message}")
    
    return validate_version

def _create_api_key_validator(logger):
    """Create API key validator dengan real-time feedback"""
    def validate_api_key_field(change):
        value = change['new'].strip()
        
        if value:
            validation_result = validate_api_key(value)
            if not validation_result['valid']:
                logger.warning(f"⚠️ API Key: {validation_result['message']}")
    
    return validate_api_key_field

def validate_workspace_format(workspace: str) -> Tuple[bool, str]:
    """Validate workspace format dengan one-liner checks"""
    if not workspace:
        return True, "OK"  # Empty is allowed untuk real-time validation
    
    if len(workspace) < 3:
        return False, "Workspace minimal 3 karakter"
    
    if not workspace.replace('-', '').replace('_', '').isalnum():
        return False, "Hanya alphanumeric, dash, dan underscore"
    
    if workspace.startswith('-') or workspace.endswith('-'):
        return False, "Tidak boleh diawali atau diakhiri dengan dash"
    
    return True, "Format valid"

def validate_project_format(project: str) -> Tuple[bool, str]:
    """Validate project format dengan one-liner checks"""
    if not project:
        return True, "OK"  # Empty is allowed untuk real-time validation
    
    if len(project) < 3:
        return False, "Project minimal 3 karakter"
    
    if not project.replace('-', '').replace('_', '').isalnum():
        return False, "Hanya alphanumeric, dash, dan underscore"
    
    return True, "Format valid"

def validate_version_format(version: str) -> Tuple[bool, str]:
    """Validate version format dengan one-liner checks"""
    if not version:
        return True, "OK"  # Empty is allowed untuk real-time validation
    
    # Version bisa berupa angka atau string seperti "v1", "1.0", dll
    if not version.replace('.', '').replace('v', '').replace('V', '').isdigit():
        return False, "Format version tidak valid"
    
    return True, "Format valid"

def validate_complete_form(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate complete form untuk final submission check"""
    config_handler = ui_components.get('config_handler')
    if not config_handler:
        return {'valid': False, 'message': 'Config handler tidak tersedia'}
    
    # Extract current values
    current_config = config_handler.extract_config_from_ui(ui_components)
    roboflow = current_config.get('data', {}).get('roboflow', {})
    
    # Required field validation
    required_fields = {
        'workspace': roboflow.get('workspace', '').strip(),
        'project': roboflow.get('project', '').strip(),
        'version': roboflow.get('version', '').strip(),
        'api_key': roboflow.get('api_key', '').strip()
    }
    
    # Check missing fields
    missing_fields = [field for field, value in required_fields.items() if not value]
    
    if missing_fields:
        return {
            'valid': False,
            'message': f"Field wajib kosong: {', '.join(missing_fields)}",
            'missing_fields': missing_fields
        }
    
    # Format validation
    format_validations = [
        ('workspace', validate_workspace_format(required_fields['workspace'])),
        ('project', validate_project_format(required_fields['project'])),
        ('version', validate_version_format(required_fields['version'])),
        ('api_key', validate_api_key(required_fields['api_key']))
    ]
    
    # Check format errors
    format_errors = []
    for field_name, (is_valid, message) in format_validations:
        if not is_valid:
            format_errors.append(f"{field_name}: {message}")
    
    if format_errors:
        return {
            'valid': False,
            'message': f"Format error: {'; '.join(format_errors)}",
            'format_errors': format_errors
        }
    
    return {
        'valid': True,
        'message': 'Form validation passed',
        'config': current_config
    }

# Export
__all__ = ['setup_validation_handler', 'validate_complete_form']