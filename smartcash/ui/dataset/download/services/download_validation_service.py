"""
File: smartcash/ui/dataset/download/services/download_validation_service.py
Deskripsi: Service untuk comprehensive validation dengan enhanced parameter processing
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.utils.parameter_extractor import extract_download_parameters, validate_extracted_parameters
from smartcash.common.environment import get_environment_manager
from smartcash.dataset.utils.path_validator import get_path_validator

class DownloadValidationService:
    """Service untuk comprehensive download validation dengan enhanced features."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.env_manager = get_environment_manager()
        self.path_validator = get_path_validator()
    
    def validate_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive validation dengan enhanced parameter processing."""
        try:
            self.logger and self.logger.info("🔍 Memulai validasi parameter download...")
            
            # Step 1: Extract parameters
            params = extract_download_parameters(self.ui_components)
            if not params:
                return self._create_validation_error("Gagal mengekstrak parameter dari UI", {})
            
            # Step 2: Basic parameter validation
            param_validation = validate_extracted_parameters(params)
            
            # Step 3: Enhanced parameter processing
            enhanced_params = self._enhance_parameters(params)
            
            # Step 4: Environment validation
            env_validation = self._validate_environment(enhanced_params)
            
            # Step 5: API connectivity (optional, non-blocking)
            api_validation = self._validate_api_connectivity(enhanced_params, timeout=5)
            
            # Step 6: Compile comprehensive result
            return self._compile_validation_result(
                enhanced_params, param_validation, env_validation, api_validation
            )
            
        except Exception as e:
            error_msg = f"Critical validation error: {str(e)}"
            self.logger and self.logger.error(f"💥 {error_msg}")
            return self._create_validation_error(error_msg, {})
    
    def _enhance_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced parameter processing dengan auto-fix."""
        enhanced = params.copy()
        
        # Auto-detect API key jika kosong
        if not enhanced.get('api_key'):
            enhanced['api_key'] = self._detect_api_key()
        
        # Fix workspace/project formatting
        for key in ['workspace', 'project']:
            if enhanced.get(key):
                enhanced[key] = enhanced[key].lower().replace(' ', '-').replace('_', '-')
        
        # Normalize version
        if enhanced.get('version'):
            enhanced['version'] = str(enhanced['version']).strip()
        
        # Smart output directory resolution
        enhanced['output_dir'] = self._resolve_output_directory(enhanced.get('output_dir', ''))
        
        return enhanced
    
    def _validate_environment(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate environment dengan comprehensive checks."""
        try:
            paths = self.path_validator.get_dataset_paths()
            
            validation_result = {
                'valid': True,
                'paths': paths,
                'environment': {
                    'is_colab': self.env_manager.is_colab,
                    'drive_mounted': self.env_manager.is_drive_mounted,
                    'storage_type': 'Google Drive' if self.env_manager.is_drive_mounted else 'Local'
                },
                'warnings': [],
                'recommendations': []
            }
            
            # Environment-specific warnings
            if self.env_manager.is_colab and not self.env_manager.is_drive_mounted:
                validation_result['warnings'].append(
                    "Google Drive tidak terhubung - dataset akan tersimpan lokal"
                )
                validation_result['recommendations'].append(
                    "Hubungkan Google Drive untuk penyimpanan permanen"
                )
            
            # Validate output directory
            output_dir = params.get('output_dir', paths['downloads'])
            try:
                from pathlib import Path
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                params['output_dir'] = str(output_path)
                
            except Exception as e:
                validation_result['warnings'].append(f"Output directory issue: {str(e)}")
                fallback_path = Path(paths['downloads'])
                fallback_path.mkdir(parents=True, exist_ok=True)
                params['output_dir'] = str(fallback_path)
                validation_result['recommendations'].append(f"Using fallback directory: {fallback_path}")
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Environment validation failed: {str(e)}",
                'paths': {},
                'environment': {}
            }
    
    def _validate_api_connectivity(self, params: Dict[str, Any], timeout: int = 5) -> Dict[str, Any]:
        """Optional API connectivity test dengan reasonable timeout."""
        try:
            import requests
            
            metadata_url = (
                f"https://api.roboflow.com/{params['workspace']}"
                f"/{params['project']}/{params['version']}"
                f"/yolov5pytorch?api_key={params['api_key']}"
            )
            
            response = requests.get(metadata_url, timeout=timeout)
            
            if response.status_code == 200:
                metadata = response.json()
                if 'export' in metadata and 'link' in metadata['export']:
                    dataset_info = {}
                    if 'project' in metadata:
                        project_info = metadata['project']
                        dataset_info['classes'] = len(project_info.get('classes', []))
                    if 'version' in metadata:
                        version_info = metadata['version']
                        dataset_info['images'] = version_info.get('images', 0)
                    
                    return {
                        'valid': True,
                        'message': 'API connectivity successful',
                        'dataset_info': dataset_info,
                        'download_url': metadata['export']['link']
                    }
                else:
                    return {'valid': False, 'message': 'Invalid API response structure'}
            
            elif response.status_code == 404:
                return {
                    'valid': False,
                    'message': f'Dataset not found: {params["workspace"]}/{params["project"]}:{params["version"]}',
                    'suggestion': 'Check workspace, project name, dan version number'
                }
            elif response.status_code in [401, 403]:
                return {
                    'valid': False,
                    'message': 'API key invalid atau insufficient permissions',
                    'suggestion': 'Check your Roboflow API key'
                }
            else:
                return {'valid': False, 'message': f'API error: HTTP {response.status_code}'}
                
        except requests.Timeout:
            return {
                'valid': False,
                'message': f'API request timeout (>{timeout}s)',
                'suggestion': 'Check internet connection atau try again later'
            }
        except requests.ConnectionError:
            return {
                'valid': False,
                'message': 'Cannot connect to Roboflow API',
                'suggestion': 'Check internet connection'
            }
        except Exception as e:
            return {'valid': False, 'message': f'API validation error: {str(e)}'}
    
    def _compile_validation_result(self, params: Dict[str, Any], param_validation: Dict[str, Any], 
                                 env_validation: Dict[str, Any], api_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Compile comprehensive validation result dengan smart decision making."""
        # Critical errors check
        critical_errors = []
        
        # Required parameters
        required_fields = ['workspace', 'project', 'version', 'api_key']
        missing_required = [field for field in required_fields if not params.get(field)]
        critical_errors.extend([f"Missing required field: {field}" for field in missing_required])
        
        # API key validation
        api_key = params.get('api_key', '')
        if not api_key or len(api_key.strip()) < 10:
            critical_errors.append("API key invalid atau too short")
        
        # Environment path validation
        if not env_validation.get('valid', True):
            env_error = env_validation.get('error', 'Environment validation failed')
            if 'output_dir' not in params:
                critical_errors.append(env_error)
        
        # Determine final validation result
        is_valid = len(critical_errors) == 0
        
        # Collect warnings
        all_warnings = []
        if param_validation.get('warnings'):
            all_warnings.extend(param_validation['warnings'])
        if env_validation.get('warnings'):
            all_warnings.extend(env_validation['warnings'])
        if not api_validation.get('valid', True):
            all_warnings.append(f"API connectivity: {api_validation.get('message', 'Unknown error')}")
        
        # Create result
        result = {
            'valid': is_valid,
            'params': params,
            'validation_details': {
                'parameters': param_validation,
                'environment': env_validation,
                'api_connectivity': api_validation
            },
            'warnings': all_warnings,
            'critical_errors': critical_errors
        }
        
        # Set message
        if is_valid:
            storage_type = env_validation.get('environment', {}).get('storage_type', 'Local')
            warning_count = len(all_warnings)
            
            if warning_count > 0:
                result['message'] = f"✅ Validation passed with {warning_count} warnings - Storage: {storage_type}"
            else:
                result['message'] = f"✅ Validation passed - Storage: {storage_type}"
            
            if self.logger:
                self.logger.success(result['message'])
                self.logger.info("📋 Validated parameters:")
                self.logger.info(f"   • Workspace: {params.get('workspace', 'N/A')}")
                self.logger.info(f"   • Project: {params.get('project', 'N/A')}")
                self.logger.info(f"   • Version: {params.get('version', 'N/A')}")
                self.logger.info(f"   • Output: {params.get('output_dir', 'N/A')}")
                
                if all_warnings and len(all_warnings) <= 3:
                    self.logger.info(f"⚠️ Warnings:")
                    for warning in all_warnings:
                        self.logger.info(f"   • {warning}")
        else:
            result['message'] = f"❌ Validation failed: {', '.join(critical_errors)}"
            if self.logger:
                self.logger.error(result['message'])
                self.logger.info("🔧 Please fix the following issues:")
                for error in critical_errors:
                    self.logger.info(f"   • {error}")
        
        return result
    
    def _detect_api_key(self) -> str:
        """Detect API key dari berbagai environment sources."""
        import os
        
        # Environment variables
        for env_key in ['ROBOFLOW_API_KEY', 'ROBOFLOW_KEY', 'RF_API_KEY']:
            api_key = os.environ.get(env_key, '').strip()
            if api_key and len(api_key) > 10:
                return api_key
        
        # Google Colab userdata
        try:
            from google.colab import userdata
            for key_name in ['ROBOFLOW_API_KEY', 'roboflow_api_key', 'ROBOFLOW_KEY', 'API_KEY']:
                try:
                    api_key = userdata.get(key_name, '').strip()
                    if api_key and len(api_key) > 10:
                        return api_key
                except:
                    continue
        except:
            pass
        
        return ''
    
    def _resolve_output_directory(self, output_dir: str) -> str:
        """Smart resolution untuk output directory."""
        try:
            paths = self.path_validator.get_dataset_paths()
            
            # Jika kosong atau default, gunakan downloads path
            if not output_dir or output_dir in ['data', 'downloads', '/content/data']:
                return paths['downloads']
            
            # Jika absolute dan valid, gunakan itu
            if os.path.isabs(output_dir):
                from pathlib import Path
                output_path = Path(output_dir)
                if output_path.exists() or output_path.parent.exists():
                    return str(output_path)
            
            # Relative path, resolve ke downloads
            from pathlib import Path
            downloads_path = Path(paths['downloads'])
            resolved_path = downloads_path / output_dir
            return str(resolved_path)
            
        except Exception:
            return 'data/downloads'
    
    def _create_validation_error(self, message: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create standardized validation error response."""
        return {
            'valid': False,
            'message': message,
            'params': params,
            'validation_details': {'error': message},
            'warnings': [],
            'critical_errors': [message]
        }