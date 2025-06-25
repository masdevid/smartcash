"""
File: smartcash/ui/setup/dependency/utils/validators.py
Deskripsi: Dependency-specific utilities untuk validation dan safe operations
"""
from typing import Dict, Any, List, Optional
from smartcash.ui.utils.fallback_utils import try_operation_safe
from smartcash.common.logger import get_logger, safe_log_to_ui

class DependencyValidator:
    """Validator untuk dependency configuration dan status"""
    
    def __init__(self, logger_name: str = "dependency.validator"):
        self.logger = get_logger(logger_name)
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dependency config dengan specific rules"""
        issues = []
        
        # Check required fields
        required_fields = ['dependencies', 'install_options']
        for field in required_fields:
            if field not in config:
                issues.append(f"Missing required field: {field}")
        
        # Validate dependency structure
        if 'dependencies' in config:
            for dep_name, dep_config in config['dependencies'].items():
                if not isinstance(dep_config, dict):
                    issues.append(f"Invalid config for {dep_name}: must be dict")
                elif 'required' not in dep_config:
                    dep_config['required'] = True  # Default
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'config': config
        }
    
    def check_critical_dependencies(self, dependencies: Dict[str, Any]) -> List[str]:
        """Return list of missing critical dependencies"""
        critical_deps = ['torch', 'torchvision', 'ultralytics']
        missing = []
        
        for dep in critical_deps:
            if dep not in dependencies or not dependencies[dep].get('required', True):
                missing.append(dep)
        
        return missing

def extract_dependency_config_safe(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Safe extraction dengan fallback ke defaults"""
    def extract_operation():
        from .config_extractor import extract_dependency_config
        return extract_dependency_config(ui_components)
    
    def fallback_config(error):
        logger = get_logger("dependency.config_extract")
        logger.warning(f"⚠️ Using fallback config: {str(error)}")
        return {
            'dependencies': {
                'torch': {'version': 'latest', 'required': True},
                'torchvision': {'version': 'latest', 'required': True},
                'ultralytics': {'version': 'latest', 'required': True}
            },
            'install_options': {'force_reinstall': False, 'upgrade': True}
        }
    
    return try_operation_safe(
        extract_operation,
        fallback_value={},
        on_error=fallback_config,
        operation_name="extract_dependency_config"
    )

def update_dependency_ui_safe(ui_components: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Safe UI update dengan error logging"""
    def update_operation():
        from .config_updater import update_dependency_ui
        update_dependency_ui(ui_components, config)
        return True
    
    def log_error(error):
        safe_log_to_ui(ui_components, f"❌ Update UI failed: {str(error)}", "error")
        return False
    
    return try_operation_safe(
        update_operation,
        fallback_value=False,
        on_error=log_error,
        operation_name="update_dependency_ui"
    )

def create_dependency_defaults() -> Dict[str, Any]:
    """Create dependency-specific defaults"""
    return {
        'module_name': 'dependency',
        'dependencies': {
            'torch': {
                'version': 'latest',
                'required': True,
                'install_args': ['--index-url', 'https://download.pytorch.org/whl/cu118']
            },
            'torchvision': {
                'version': 'latest', 
                'required': True,
                'install_args': ['--index-url', 'https://download.pytorch.org/whl/cu118']
            },
            'ultralytics': {
                'version': 'latest',
                'required': True
            },
            'roboflow': {
                'version': 'latest',
                'required': False
            }
        },
        'install_options': {
            'force_reinstall': False,
            'upgrade': True,
            'quiet': False,
            'parallel_install': True,
            'max_workers': 2
        },
        'validation': {
            'check_imports': True,
            'verify_versions': True,
            'test_gpu': True
        }
    }