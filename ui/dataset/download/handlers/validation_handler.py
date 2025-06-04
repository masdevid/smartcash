"""
File: smartcash/ui/dataset/download/handlers/validation_handler.py
Deskripsi: Fixed validation handler dengan diagnosis lengkap dan toleransi error yang tepat
"""

import os
from typing import Dict, Any, List
from pathlib import Path
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.ui.dataset.download.utils.parameter_extractor import extract_download_parameters, validate_extracted_parameters

def validate_download_parameters(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced validation dengan diagnosis komprehensif dan toleransi yang tepat.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dict hasil validasi dengan detail diagnosis
    """
    logger = ui_components.get('logger')
    
    try:
        logger and logger.info("🔍 Memulai validasi parameter download...")
        
        # Step 1: Extract parameters dengan comprehensive error handling
        params = extract_download_parameters(ui_components)
        
        if not params:
            return _create_validation_error("Gagal mengekstrak parameter dari UI", {})
        
        # Step 2: Validate extracted parameters (relaxed)
        param_validation = validate_extracted_parameters(params)
        
        # Step 3: Enhanced parameter validation dengan auto-fix
        enhanced_params = _enhance_and_fix_parameters(params, ui_components)
        
        # Step 4: Environment-aware validation
        env_validation = _validate_environment_paths(enhanced_params, ui_components)
        
        # Step 5: Dataset structure awareness (non-blocking)
        dataset_awareness = _check_existing_dataset_structure(enhanced_params)
        
        # Step 6: Compile comprehensive validation result
        validation_result = _compile_validation_result(
            enhanced_params, param_validation, env_validation, dataset_awareness, logger
        )
        
        return validation_result
        
    except Exception as e:
        error_msg = f"Critical validation error: {str(e)}"
        logger and logger.error(f"💥 {error_msg}")
        
        return _create_validation_error(error_msg, {})

def _enhance_and_fix_parameters(params: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced parameter processing dengan auto-fix common issues."""
    enhanced_params = params.copy()
    
    # Auto-detect API key jika kosong
    if not enhanced_params.get('api_key'):
        enhanced_params['api_key'] = _detect_comprehensive_api_key()
    
    # Fix workspace/project formatting
    for key in ['workspace', 'project']:
        if enhanced_params.get(key):
            enhanced_params[key] = enhanced_params[key].lower().replace(' ', '-').replace('_', '-')
    
    # Normalize version
    if enhanced_params.get('version'):
        enhanced_params['version'] = str(enhanced_params['version']).strip()
    
    # Smart output directory resolution
    enhanced_params['output_dir'] = _resolve_smart_output_directory(
        enhanced_params.get('output_dir', ''), ui_components
    )
    
    return enhanced_params

def _detect_comprehensive_api_key() -> str:
    """Comprehensive API key detection dari berbagai sumber."""
    # 1. Environment variables
    env_keys = ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY', 'API_KEY_ROBOFLOW']
    for env_key in env_keys:
        api_key = os.environ.get(env_key, '').strip()
        if api_key and len(api_key) > 10:
            return api_key
    
    # 2. Google Colab userdata
    try:
        from google.colab import userdata
        userdata_keys = [
            'ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 
            'roboflow_key', 'RF_API_KEY', 'API_KEY'
        ]
        
        for key_name in userdata_keys:
            try:
                api_key = userdata.get(key_name, '').strip()
                if api_key and len(api_key) > 10:
                    return api_key
            except Exception:
                continue
                
    except ImportError:
        pass
    except Exception:
        pass
    
    return ''

def _resolve_smart_output_directory(output_dir: str, ui_components: Dict[str, Any]) -> str:
    """Smart resolution untuk output directory berdasarkan environment."""
    try:
        env_manager = get_environment_manager()
        path_validator = get_path_validator()
        
        # Get environment-appropriate paths
        paths = path_validator.get_dataset_paths()
        
        # Jika output_dir kosong atau default, gunakan downloads path
        if not output_dir or output_dir in ['data', 'downloads', '/content/data']:
            return paths['downloads']
        
        # Jika output_dir sudah absolute dan valid, gunakan itu
        if os.path.isabs(output_dir):
            output_path = Path(output_dir)
            if output_path.exists() or output_path.parent.exists():
                return str(output_path)
        
        # Jika relative path, resolve ke downloads directory
        downloads_path = Path(paths['downloads'])
        resolved_path = downloads_path / output_dir
        
        return str(resolved_path)
        
    except Exception:
        # Fallback ke default
        return 'data/downloads'

def _validate_environment_paths(params: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate environment paths dengan comprehensive checks."""
    try:
        env_manager = get_environment_manager()
        path_validator = get_path_validator()
        
        # Get paths berdasarkan environment
        paths = path_validator.get_dataset_paths()
        
        validation_result = {
            'valid': True,
            'paths': paths,
            'environment': {
                'is_colab': env_manager.is_colab,
                'drive_mounted': env_manager.is_drive_mounted,
                'base_dir': str(env_manager.base_dir),
                'storage_type': 'Google Drive' if env_manager.is_drive_mounted else 'Local'
            },
            'warnings': [],
            'recommendations': []
        }
        
        # Environment-specific validations
        if env_manager.is_colab:
            if not env_manager.is_drive_mounted:
                validation_result['warnings'].append(
                    "Google Drive tidak terhubung - dataset akan tersimpan lokal (hilang saat restart)"
                )
                validation_result['recommendations'].append(
                    "Hubungkan Google Drive untuk penyimpanan permanen"
                )
        
        # Validate output directory accessibility
        output_dir = params.get('output_dir', paths['downloads'])
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Update params dengan resolved path
            params['output_dir'] = str(output_path)
            
        except Exception as e:
            validation_result['warnings'].append(f"Output directory issue: {str(e)}")
            
            # Fallback ke downloads path
            fallback_path = Path(paths['downloads'])
            fallback_path.mkdir(parents=True, exist_ok=True)
            params['output_dir'] = str(fallback_path)
            
            validation_result['recommendations'].append(
                f"Using fallback directory: {fallback_path}"
            )
        
        return validation_result
        
    except Exception as e:
        return {
            'valid': False,
            'error': f"Environment validation failed: {str(e)}",
            'paths': {},
            'environment': {}
        }

def _check_existing_dataset_structure(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check existing dataset structure dengan detailed analysis."""
    try:
        path_validator = get_path_validator()
        
        # Get dataset paths
        paths = path_validator.get_dataset_paths()
        
        # Validate existing structure
        structure_validation = path_validator.validate_dataset_structure(paths['data_root'])
        
        # Enhanced analysis
        analysis = {
            'has_existing_data': structure_validation.get('total_images', 0) > 0,
            'structure_valid': structure_validation.get('valid', False),
            'total_images': structure_validation.get('total_images', 0),
            'total_labels': structure_validation.get('total_labels', 0),
            'splits_status': structure_validation.get('splits', {}),
            'issues': structure_validation.get('issues', []),
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['has_existing_data']:
            if analysis['total_images'] > 100:
                analysis['recommendations'].append(
                    f"Dataset existing detected: {analysis['total_images']} images found"
                )
                analysis['recommendations'].append(
                    "Consider backing up existing data before download"
                )
            else:
                analysis['recommendations'].append(
                    f"Small dataset found: {analysis['total_images']} images - safe to overwrite"
                )
        
        if analysis['issues']:
            analysis['recommendations'].append(
                f"Issues detected: {len(analysis['issues'])} - will be resolved after download"
            )
        
        return analysis
        
    except Exception as e:
        return {
            'has_existing_data': False,
            'structure_valid': False,
            'error': f"Dataset structure check failed: {str(e)}",
            'recommendations': ["Could not analyze existing dataset structure"]
        }

def _compile_validation_result(params: Dict[str, Any], param_validation: Dict[str, Any], 
                             env_validation: Dict[str, Any], dataset_awareness: Dict[str, Any],
                             logger) -> Dict[str, Any]:
    """Compile comprehensive validation result dengan smart decision making."""
    
    # Check critical errors yang benar-benar blocking
    critical_errors = []
    
    # Required parameters check
    required_fields = ['workspace', 'project', 'version', 'api_key']
    missing_required = [field for field in required_fields if not params.get(field)]
    
    if missing_required:
        critical_errors.extend([f"Missing required field: {field}" for field in missing_required])
    
    # API key validation
    api_key = params.get('api_key', '')
    if not api_key or len(api_key.strip()) < 10:
        critical_errors.append("API key invalid or too short")
    
    # Environment path validation
    if not env_validation.get('valid', True):
        env_error = env_validation.get('error', 'Environment validation failed')
        # Only critical if we can't create any output directory
        if 'output_dir' not in params:
            critical_errors.append(env_error)
    
    # Determine final validation result
    is_valid = len(critical_errors) == 0
    
    # Collect all warnings
    all_warnings = []
    
    if param_validation.get('warnings'):
        all_warnings.extend(param_validation['warnings'])
    
    if env_validation.get('warnings'):
        all_warnings.extend(env_validation['warnings'])
    
    if dataset_awareness.get('recommendations'):
        all_warnings.extend(dataset_awareness['recommendations'])
    
    # Create result
    result = {
        'valid': is_valid,
        'params': params,
        'validation_details': {
            'parameters': param_validation,
            'environment': env_validation,
            'dataset_structure': dataset_awareness
        },
        'warnings': all_warnings,
        'critical_errors': critical_errors
    }
    
    # Set appropriate message
    if is_valid:
        storage_type = env_validation.get('environment', {}).get('storage_type', 'Local')
        warning_count = len(all_warnings)
        
        if warning_count > 0:
            result['message'] = f"✅ Validation passed with {warning_count} warnings - Storage: {storage_type}"
        else:
            result['message'] = f"✅ Validation passed - Storage: {storage_type}"
        
        if logger:
            logger.success(result['message'])
            
            # Log key parameters (without sensitive info)
            logger.info("📋 Validated parameters:")
            logger.info(f"   • Workspace: {params.get('workspace', 'N/A')}")
            logger.info(f"   • Project: {params.get('project', 'N/A')}")
            logger.info(f"   • Version: {params.get('version', 'N/A')}")
            logger.info(f"   • Output: {params.get('output_dir', 'N/A')}")
            
            if all_warnings:
                logger.info(f"⚠️ Warnings ({len(all_warnings)}):")
                for warning in all_warnings[:3]:  # Show first 3
                    logger.info(f"   • {warning}")
                if len(all_warnings) > 3:
                    logger.info(f"   • ... and {len(all_warnings) - 3} more")
    else:
        result['message'] = f"❌ Validation failed: {', '.join(critical_errors)}"
        
        if logger:
            logger.error(result['message'])
            logger.info("🔧 Please fix the following issues:")
            for error in critical_errors:
                logger.info(f"   • {error}")
    
    return result

def _create_validation_error(message: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized validation error response."""
    return {
        'valid': False,
        'message': message,
        'params': params,
        'validation_details': {'error': message},
        'warnings': [],
        'critical_errors': [message]
    }

def validate_api_connectivity(params: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
    """
    Optional API connectivity test dengan reasonable timeout.
    
    Args:
        params: Parameters dengan API key dan dataset info
        timeout: Timeout untuk request (default 10s)
        
    Returns:
        API connectivity validation result
    """
    try:
        import requests
        
        # Build metadata URL
        metadata_url = (
            f"https://api.roboflow.com/{params['workspace']}"
            f"/{params['project']}/{params['version']}"
            f"/yolov5pytorch?api_key={params['api_key']}"
        )
        
        # Test request dengan timeout
        response = requests.get(metadata_url, timeout=timeout)
        
        if response.status_code == 200:
            metadata = response.json()
            
            # Validate response structure
            if 'export' in metadata and 'link' in metadata['export']:
                dataset_info = {}
                
                # Extract dataset info jika tersedia
                if 'project' in metadata:
                    project_info = metadata['project']
                    dataset_info['classes'] = len(project_info.get('classes', []))
                    
                if 'version' in metadata:
                    version_info = metadata['version']
                    dataset_info['images'] = version_info.get('images', 0)
                
                return {
                    'valid': True,
                    'message': 'API connectivity successful',
                    'metadata': metadata,
                    'dataset_info': dataset_info,
                    'download_url': metadata['export']['link']
                }
            else:
                return {
                    'valid': False,
                    'message': 'Invalid API response structure',
                    'status_code': response.status_code
                }
        
        elif response.status_code == 404:
            return {
                'valid': False,
                'message': f'Dataset not found: {params["workspace"]}/{params["project"]}:{params["version"]}',
                'status_code': 404,
                'suggestion': 'Check workspace, project name, and version number'
            }
        
        elif response.status_code in [401, 403]:
            return {
                'valid': False,
                'message': 'API key invalid or insufficient permissions',
                'status_code': response.status_code,
                'suggestion': 'Check your Roboflow API key'
            }
        
        else:
            return {
                'valid': False,
                'message': f'API error: HTTP {response.status_code}',
                'status_code': response.status_code
            }
            
    except requests.Timeout:
        return {
            'valid': False,
            'message': f'API request timeout (>{timeout}s)',
            'error_type': 'timeout',
            'suggestion': 'Check internet connection or try again later'
        }
    
    except requests.ConnectionError:
        return {
            'valid': False,
            'message': 'Cannot connect to Roboflow API',
            'error_type': 'connection_error',
            'suggestion': 'Check internet connection'
        }
    
    except Exception as e:
        return {
            'valid': False,
            'message': f'API validation error: {str(e)}',
            'error_type': 'unknown',
            'error_details': str(e)
        }

def run_full_validation_with_api_test(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full validation termasuk API connectivity test.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Comprehensive validation result dengan API test
    """
    logger = ui_components.get('logger')
    
    try:
        logger and logger.info("🔍 Starting full validation with API test...")
        
        # Basic validation first
        basic_result = validate_download_parameters(ui_components)
        
        if not basic_result['valid']:
            return basic_result
        
        params = basic_result['params']
        
        # API connectivity test
        logger and logger.info("🌐 Testing API connectivity...")
        api_result = validate_api_connectivity(params)
        
        # Add API result to validation details
        basic_result['validation_details']['api_connectivity'] = api_result
        
        if api_result['valid']:
            logger and logger.success("✅ API connectivity test passed")
            
            # Log dataset info jika tersedia
            if 'dataset_info' in api_result:
                info = api_result['dataset_info']
                logger and logger.info(f"📊 Dataset info: {info.get('classes', 'N/A')} classes, {info.get('images', 'N/A')} images")
        else:
            # Add API warning
            basic_result['warnings'].append(f"API connectivity: {api_result['message']}")
            logger and logger.warning(f"⚠️ API connectivity issue: {api_result['message']}")
            
            if 'suggestion' in api_result:
                basic_result['warnings'].append(f"Suggestion: {api_result['suggestion']}")
        
        # Update message to include API test status
        if basic_result.get('message', '').startswith('✅'):
            api_status = "✅" if api_result['valid'] else "⚠️"
            basic_result['message'] += f" | API: {api_status}"
        
        logger and logger.success(f"✅ Full validation completed with {'API connectivity' if api_result['valid'] else 'API warnings'}")
        
        return basic_result
        
    except Exception as e:
        logger and logger.error(f"💥 Full validation error: {str(e)}")
        
        return _create_validation_error(f'Full validation failed: {str(e)}', {})