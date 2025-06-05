"""
File: smartcash/ui/dataset/downloader/handlers/validation_handler.py
Deskripsi: Updated validation handler tanpa format validation (hardcoded yolov5pytorch)
"""

from typing import Dict, Any, List, Callable
from smartcash.ui.dataset.downloader.utils.colab_secrets import validate_api_key, get_available_secrets
from smartcash.ui.dataset.downloader.utils.operation_utils import validate_dataset_identifier, validate_space
from smartcash.ui.dataset.downloader.handlers.defaults import get_download_validation_rules

def setup_validation_handler(ui_components: Dict[str, Any], config: Dict[str, Any], logger) -> Dict[str, Callable]:
    """Setup validation handlers untuk different validation scenarios"""
    
    def validate_config_handler(config_to_validate: Dict[str, Any]) -> Dict[str, Any]:
        """Validate download config dengan comprehensive checks"""
        return validate_download_config(config_to_validate, logger)
    
    def validate_environment_handler() -> Dict[str, Any]:
        """Validate environment untuk download operations"""
        return validate_download_environment(logger)
    
    def validate_api_access_handler(workspace: str, project: str, api_key: str) -> Dict[str, Any]:
        """Validate API access untuk specific dataset"""
        return validate_api_access(workspace, project, api_key, logger)
    
    def validate_disk_space_handler(required_mb: float) -> Dict[str, Any]:
        """Validate disk space untuk download"""
        return validate_space(required_mb)
    
    return {
        'validate_config': validate_config_handler,
        'validate_environment': validate_environment_handler,
        'validate_api_access': validate_api_access_handler,
        'validate_disk_space': validate_disk_space_handler
    }

def validate_download_config(config: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Comprehensive download config validation tanpa format validation (hardcoded yolov5pytorch)"""
    
    validation_rules = get_download_validation_rules()
    errors = []
    warnings = []
    
    # Required fields validation
    required_fields = validation_rules['required_fields']
    missing_fields = [field for field in required_fields if not config.get(field, '').strip()]
    
    if missing_fields:
        errors.extend([f"Field {field} wajib diisi" for field in missing_fields])
    
    # Field constraint validation
    constraints = validation_rules['field_constraints']
    for field, constraint in constraints.items():
        value = config.get(field, '').strip()
        if not value:
            continue  # Skip empty fields (handled by required check)
        
        # Length validation
        if len(value) < constraint.get('min_length', 0):
            errors.append(f"{field} minimal {constraint['min_length']} karakter")
        
        if len(value) > constraint.get('max_length', 1000):
            errors.append(f"{field} maksimal {constraint['max_length']} karakter")
        
        # Pattern validation
        import re
        pattern = constraint.get('pattern')
        if pattern and not re.match(pattern, value):
            errors.append(f"{field} format tidak valid: {constraint.get('description', 'Invalid format')}")
    
    # API key specific validation
    api_key = config.get('api_key', '').strip()
    if api_key:
        api_validation = validate_api_key(api_key)
        if not api_validation['valid']:
            errors.append(f"API key tidak valid: {api_validation['message']}")
    
    # Format validation - ensure hardcoded format
    output_format = config.get('output_format', '')
    if output_format and output_format != 'yolov5pytorch':
        warnings.append(f"Format akan di-override ke yolov5pytorch (dari: {output_format})")
    
    # Force set hardcoded format
    config['output_format'] = 'yolov5pytorch'
    
    # Dataset identifier validation
    workspace = config.get('workspace', '').strip()
    project = config.get('project', '').strip() 
    version = config.get('version', '').strip()
    
    if workspace and project and version:
        identifier_validation = validate_dataset_identifier(workspace, project, version)
        if not identifier_validation['valid']:
            errors.extend(identifier_validation['errors'])
    
    # Boolean fields validation
    boolean_fields = validation_rules['boolean_fields']
    for field in boolean_fields:
        value = config.get(field)
        if value is not None and not isinstance(value, bool):
            warnings.append(f"{field} should be boolean, got {type(value).__name__}")
    
    # Generate recommendations
    recommendations = []
    if not config.get('backup_existing', False) and config.get('organize_dataset', True):
        recommendations.append("💡 Pertimbangkan enable backup untuk keamanan data")
    
    if not config.get('validate_download', True):
        recommendations.append("⚠️ Disable validasi download dapat menyebabkan data corruption")
    
    # Add format info
    recommendations.append("📦 Format dataset: YOLOv5 PyTorch (hardcoded untuk konsistensi)")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'recommendations': recommendations,
        'field_count': len([k for k, v in config.items() if v and not k.startswith('_')]),
        'required_complete': len(missing_fields) == 0,
        'format_locked': True,
        'format': 'yolov5pytorch'
    }

def validate_download_environment(logger=None) -> Dict[str, Any]:
    """Validate environment untuk download operations"""
    
    validation_result = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'environment_info': {}
    }
    
    try:
        # Check Colab environment
        try:
            import google.colab
            validation_result['environment_info']['platform'] = 'Google Colab'
            validation_result['environment_info']['colab_available'] = True
        except ImportError:
            validation_result['environment_info']['platform'] = 'Local/Other'
            validation_result['environment_info']['colab_available'] = False
            validation_result['warnings'].append("Tidak berjalan di Google Colab - beberapa fitur mungkin terbatas")
        
        # Check Drive mount
        from pathlib import Path
        drive_mounted = Path('/content/drive/MyDrive').exists()
        validation_result['environment_info']['drive_mounted'] = drive_mounted
        
        if not drive_mounted:
            validation_result['warnings'].append("Google Drive tidak ter-mount - data disimpan lokal")
        
        # Check internet connectivity
        try:
            import requests
            response = requests.get('https://api.roboflow.com', timeout=5)
            validation_result['environment_info']['internet_available'] = True
            validation_result['environment_info']['roboflow_accessible'] = response.status_code < 500
        except Exception:
            validation_result['environment_info']['internet_available'] = False
            validation_result['environment_info']['roboflow_accessible'] = False
            validation_result['issues'].append("❌ Tidak dapat mengakses Roboflow API - periksa koneksi internet")
            validation_result['valid'] = False
        
        # Check available secrets
        secrets_info = get_available_secrets()
        validation_result['environment_info']['secrets_available'] = secrets_info.get('colab_available', False)
        validation_result['environment_info']['secrets_count'] = secrets_info.get('secrets_found', 0)
        
        # Check disk space
        try:
            import shutil
            total, used, free = shutil.disk_usage('/content' if validation_result['environment_info']['colab_available'] else '.')
            free_gb = free / (1024**3)
            validation_result['environment_info']['free_space_gb'] = free_gb
            
            if free_gb < 1.0:  # Less than 1GB
                validation_result['issues'].append("❌ Disk space kurang dari 1GB - download mungkin gagal")
                validation_result['valid'] = False
            elif free_gb < 2.0:  # Less than 2GB
                validation_result['warnings'].append("⚠️ Disk space terbatas - monitor penggunaan saat download")
                
        except Exception as e:
            validation_result['warnings'].append(f"Tidak dapat cek disk space: {str(e)}")
        
        # Check required packages
        required_packages = ['requests', 'tqdm', 'pathlib']
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            validation_result['issues'].append(f"❌ Missing packages: {', '.join(missing_packages)}")
            validation_result['valid'] = False
        
        validation_result['environment_info']['required_packages_available'] = len(missing_packages) == 0
        
        # Add format compatibility info
        validation_result['environment_info']['format_support'] = {
            'yolov5pytorch': True,  # Always supported
            'format_locked': True,
            'supported_formats': ['yolov5pytorch']  # Only this format
        }
        
    except Exception as e:
        validation_result['valid'] = False
        validation_result['issues'].append(f"❌ Error validating environment: {str(e)}")
    
    return validation_result

def validate_api_access(workspace: str, project: str, api_key: str, logger=None) -> Dict[str, Any]:
    """Validate API access untuk specific dataset dengan actual API call"""
    
    validation_result = {
        'valid': False,
        'accessible': False,
        'metadata_available': False,
        'issues': [],
        'dataset_info': {}
    }
    
    try:
        # Basic parameter validation
        if not all([workspace.strip(), project.strip(), api_key.strip()]):
            validation_result['issues'].append("❌ Workspace, project, dan API key wajib diisi")
            return validation_result
        
        # API key format validation
        api_validation = validate_api_key(api_key)
        if not api_validation['valid']:
            validation_result['issues'].append(f"❌ API key tidak valid: {api_validation['message']}")
            return validation_result
        
        # Test API access
        from smartcash.dataset.downloader.roboflow_client import create_roboflow_client
        
        client = create_roboflow_client(api_key, logger)
        
        # Validate credentials
        cred_result = client.validate_credentials(workspace, project)
        if not cred_result['valid']:
            validation_result['issues'].append(f"❌ Kredensial tidak valid: {cred_result['message']}")
            return validation_result
        
        validation_result['accessible'] = True
        
        # Try to get metadata dengan hardcoded format
        try:
            metadata_result = client.get_dataset_metadata(workspace, project, '1', 'yolov5pytorch')  # Test with hardcoded format
            
            if metadata_result['status'] == 'success':
                validation_result['metadata_available'] = True
                validation_result['dataset_info'] = {
                    'project_name': metadata_result['data'].get('project', {}).get('name', project),
                    'classes_count': len(metadata_result['data'].get('project', {}).get('classes', [])),
                    'workspace_accessible': True,
                    'format_supported': 'yolov5pytorch'  # Always this format
                }
                validation_result['valid'] = True
            else:
                validation_result['issues'].append(f"⚠️ Metadata tidak dapat diakses: {metadata_result['message']}")
                validation_result['valid'] = True  # Credentials valid but specific version might not exist
                
        except Exception as e:
            validation_result['issues'].append(f"⚠️ Error accessing metadata: {str(e)}")
            validation_result['valid'] = True  # Credentials valid but metadata access failed
        
    except Exception as e:
        validation_result['issues'].append(f"❌ Error validating API access: {str(e)}")
    
    return validation_result

def create_validation_summary(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create comprehensive validation summary dari multiple validations"""
    
    summary = {
        'overall_valid': True,
        'total_checks': len(validation_results),
        'passed_checks': 0,
        'failed_checks': 0,
        'warning_count': 0,
        'all_issues': [],
        'all_warnings': [],
        'all_recommendations': [],
        'validation_categories': {},
        'format_info': {
            'locked': True,
            'format': 'yolov5pytorch',
            'description': 'Format hardcoded untuk konsistensi'
        }
    }
    
    for i, result in enumerate(validation_results):
        is_valid = result.get('valid', False)
        
        if is_valid:
            summary['passed_checks'] += 1
        else:
            summary['failed_checks'] += 1
            summary['overall_valid'] = False
        
        # Collect issues
        issues = result.get('issues', []) or result.get('errors', [])
        warnings = result.get('warnings', [])
        recommendations = result.get('recommendations', [])
        
        summary['all_issues'].extend(issues)
        summary['all_warnings'].extend(warnings)
        summary['all_recommendations'].extend(recommendations)
        summary['warning_count'] += len(warnings)
        
        # Categorize validation
        category = f"validation_{i+1}"
        summary['validation_categories'][category] = {
            'valid': is_valid,
            'issues_count': len(issues),
            'warnings_count': len(warnings)
        }
    
    # Calculate success rate
    summary['success_rate'] = (summary['passed_checks'] / summary['total_checks'] * 100) if summary['total_checks'] > 0 else 0
    
    # Generate overall status
    if summary['overall_valid']:
        summary['status'] = '✅ All validations passed (YOLOv5 format ready)'
    elif summary['passed_checks'] > 0:
        summary['status'] = f"⚠️ {summary['failed_checks']} validation(s) failed"
    else:
        summary['status'] = '❌ All validations failed'
    
    return summary

# One-liner validation utilities (updated tanpa format validation)
quick_validate_api_key = lambda key: validate_api_key(key)['valid']
quick_validate_identifier = lambda w, p, v: validate_dataset_identifier(w, p, v)['valid']
validate_required_fields = lambda config: all(config.get(field, '').strip() for field in get_download_validation_rules()['required_fields'])
get_validation_errors = lambda config: validate_download_config(config).get('errors', [])
is_config_ready = lambda config: validate_download_config(config)['valid']
get_hardcoded_format = lambda: 'yolov5pytorch'
is_format_locked = lambda: True