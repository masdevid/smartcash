"""
File: smartcash/ui/dataset/downloader/handlers/validation_handler.py
Deskripsi: Handler untuk validasi parameter download dengan comprehensive checking
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from smartcash.common.environment import get_environment_manager
from smartcash.common.constants.paths import get_paths_for_environment

def validate_download_parameters(ui_components: Dict[str, Any], include_api_test: bool = False) -> Dict[str, Any]:
    """Validasi parameter download dengan comprehensive checking."""
    logger = ui_components.get('logger')
    
    try:
        # Extract parameters
        try:
            params = _extract_parameters(ui_components)
        except ValueError as e:
            logger and logger.error(f"❌ {str(e)}")
            return {
                'valid': False,
                'message': str(e),
                'params': {}
            }
        
        # Validate required fields
        required_validation = _validate_required_fields(params)
        if not required_validation['valid']:
            logger and logger.error(f"❌ {required_validation['message']}")
            return required_validation
        
        # Validate API key
        api_validation = _validate_api_key(params['api_key'])
        if not api_validation['valid']:
            logger and logger.error(f"❌ {api_validation['message']}")
            return api_validation
        
        # Validate paths
        path_validation = _validate_paths(params)
        if not path_validation['valid']:
            logger and logger.error(f"❌ {path_validation['message']}")
            return path_validation
        
        # Environment-specific validation
        env_validation = _validate_environment(params)
        
        # Test API connection jika diminta
        if include_api_test:
            logger and logger.info("🔍 Menguji koneksi API Roboflow...")
            api_test = validate_workspace_project(
                params['workspace'], 
                params['project'], 
                params['version'], 
                params['api_key']
            )
            
            if not api_test['valid']:
                logger and logger.error(f"❌ {api_test['message']}")
                return api_test
            
            # Tambahkan metadata dataset
            params['dataset_metadata'] = api_test.get('metadata', {})
            logger and logger.success(f"✅ Dataset ditemukan: {params['dataset_metadata'].get('images', 0)} gambar, {params['dataset_metadata'].get('classes', 0)} kelas")
        
        # Buat config untuk download
        config = {
            'workspace': params['workspace'],
            'project': params['project'],
            'version': params['version'],
            'api_key': params['api_key'],
            'output_dir': params['output_dir'],
            'format': params['format'],
            'backup_existing': params['backup_existing'],
            'organize_files': params['organize_files']
        }
        
        logger and logger.success(f"✅ Validasi berhasil - Storage: {env_validation.get('storage_type', 'Local')}")
        return {
            'valid': True,
            'params': params,
            'config': config,
            'warnings': env_validation.get('warnings', []),
            'message': f"✅ Validasi berhasil - Storage: {env_validation.get('storage_type', 'Local')}"
        }
        
    except Exception as e:
        logger and logger.error(f"❌ Error validasi: {str(e)}")
        return {
            'valid': False,
            'message': f"Error validasi: {str(e)}",
            'params': {}
        }

def _extract_parameters(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract parameter dari UI components dengan one-liner."""
    # Validasi komponen yang diperlukan
    required_components = ['workspace_field', 'project_field', 'version_field', 'api_key_field']
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    if missing_components:
        raise ValueError(f"Komponen tidak ditemukan: {', '.join(missing_components)}")
    
    return {
        'workspace': _safe_get_value(ui_components, 'workspace_field', '').strip(),
        'project': _safe_get_value(ui_components, 'project_field', '').strip(),
        'version': _safe_get_value(ui_components, 'version_field', '').strip(),
        'api_key': _safe_get_value(ui_components, 'api_key_field', '').strip() or _detect_api_key(),
        'output_dir': _safe_get_value(ui_components, 'output_dir_field', '').strip(),
        'format': _safe_get_value(ui_components, 'format_dropdown', 'yolov5pytorch'),
        'backup_existing': _safe_get_value(ui_components, 'backup_checkbox', False),
        'organize_files': _safe_get_value(ui_components, 'organize_checkbox', True)
    }

def _safe_get_value(ui_components: Dict[str, Any], key: str, default: Any) -> Any:
    """Safely get value dari UI component dengan one-liner."""
    component = ui_components.get(key)
    if component is None:
        return default
    return getattr(component, 'value', default)

def _validate_required_fields(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate required fields dengan one-liner."""
    required = ['workspace', 'project', 'version', 'api_key']
    missing = [field for field in required if not params.get(field)]
    
    return {
        'valid': not missing,
        'message': f"Field required tidak lengkap: {', '.join(missing)}" if missing else "Required fields valid",
        'params': params
    }

def _validate_api_key(api_key: str) -> Dict[str, Any]:
    """Validate API key format dan panjang."""
    if not api_key:
        return {'valid': False, 'message': "API key tidak ditemukan"}
    
    if len(api_key) < 10:
        return {'valid': False, 'message': "API key terlalu pendek"}
    
    # Basic format validation
    import re
    if not re.match(r'^[a-zA-Z0-9\-_]+$', api_key):
        return {'valid': False, 'message': "Format API key tidak valid"}
    
    return {'valid': True, 'message': "API key valid"}

def _validate_paths(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate output paths dengan environment detection."""
    env_manager = get_environment_manager()
    paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
    
    # Use default output dir jika kosong
    if not params.get('output_dir'):
        params['output_dir'] = paths['downloads']
    
    try:
        output_path = Path(params['output_dir'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Test write permission
        test_file = output_path / '.test_write'
        test_file.touch()
        test_file.unlink()
        
        return {'valid': True, 'message': "Path validation passed"}
        
    except Exception as e:
        return {'valid': False, 'message': f"Path tidak dapat diakses: {str(e)}"}

def _validate_environment(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate environment-specific settings."""
    env_manager = get_environment_manager()
    warnings = []
    
    storage_type = "Google Drive" if env_manager.is_drive_mounted else "Local Storage"
    
    if env_manager.is_colab and not env_manager.is_drive_mounted:
        warnings.append("⚠️ Google Drive tidak terhubung - dataset akan hilang saat runtime restart")
    
    if env_manager.is_drive_mounted:
        # Check Drive space (if possible)
        try:
            drive_path = env_manager.drive_path
            if drive_path and drive_path.exists():
                warnings.append("💾 Dataset akan disimpan permanen di Google Drive")
        except Exception:
            pass
    
    return {
        'valid': True,
        'storage_type': storage_type,
        'warnings': warnings,
        'environment': {
            'is_colab': env_manager.is_colab,
            'drive_mounted': env_manager.is_drive_mounted
        }
    }

def _detect_api_key() -> str:
    """Detect API key dari environment variables atau Colab secrets."""
    # Environment variable
    api_key = os.environ.get('ROBOFLOW_API_KEY', '')
    if api_key:
        return api_key
    
    # Colab userdata
    try:
        from google.colab import userdata
        return userdata.get('ROBOFLOW_API_KEY', '')
    except Exception:
        return ''

def validate_workspace_project(workspace: str, project: str, version: str, api_key: str) -> Dict[str, Any]:
    """Validate workspace/project existence via API call (optional check)."""
    try:
        import requests
        
        url = f"https://api.roboflow.com/{workspace}/{project}/{version}/yolov5pytorch"
        response = requests.get(url, params={'api_key': api_key}, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'valid': True,
                'message': "Dataset ditemukan",
                'metadata': {
                    'classes': len(data.get('project', {}).get('classes', [])),
                    'images': data.get('version', {}).get('images', 0)
                }
            }
        elif response.status_code == 404:
            return {'valid': False, 'message': "Dataset tidak ditemukan"}
        elif response.status_code in [401, 403]:
            return {'valid': False, 'message': "API key tidak valid atau tidak memiliki akses"}
        else:
            return {'valid': False, 'message': f"API error: {response.status_code}"}
            
    except requests.RequestException:
        return {'valid': False, 'message': "Tidak dapat terhubung ke API Roboflow"}
    except Exception as e:
        return {'valid': False, 'message': f"Error validasi API: {str(e)}"}

def get_validation_summary(validation_result: Dict[str, Any]) -> str:
    """Generate validation summary untuk display."""
    if not validation_result.get('valid'):
        return f"❌ {validation_result.get('message', 'Validasi gagal')}"
    
    params = validation_result.get('params', {})
    warnings = validation_result.get('warnings', [])
    metadata = params.get('dataset_metadata', {})
    
    summary = [
        f"✅ Parameter valid:",
        f"  • Dataset: {params.get('workspace', 'N/A')}/{params.get('project', 'N/A')}:{params.get('version', 'N/A')}",
        f"  • Format: {params.get('format', 'yolov5pytorch')}",
        f"  • Output: {params.get('output_dir', 'N/A')}"
    ]
    
    # Tambahkan metadata jika tersedia
    if metadata:
        summary.append(f"  • Jumlah gambar: {metadata.get('images', 'N/A')}")
        summary.append(f"  • Jumlah kelas: {metadata.get('classes', 'N/A')}")
    
    if warnings:
        summary.append("⚠️ Peringatan:")
        summary.extend([f"  • {warning}" for warning in warnings])
    
    return "\n".join(summary)