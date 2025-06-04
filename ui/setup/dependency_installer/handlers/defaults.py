"""
File: smartcash/ui/setup/dependency_installer/handlers/defaults.py
Deskripsi: Default configuration untuk dependency installer sesuai pattern CommonInitializer
"""

from typing import Dict, Any, List

# Default configuration sesuai pattern CommonInitializer
DEFAULT_CONFIG = {
    'module_name': 'dependency_installer',
    'version': '1.0.0',
    'created_by': 'SmartCash',
    'description': 'Dependency installer configuration untuk SmartCash project',
    
    # Package selections
    'selected_packages': [],
    'custom_packages': '',
    'auto_analyze': True,
    
    # Installation settings
    'installation': {
        'parallel_workers': 3,
        'force_reinstall': False,
        'use_cache': True,
        'timeout': 300,
        'max_retries': 2,
        'retry_delay': 1.0
    },
    
    # Analysis settings
    'analysis': {
        'check_compatibility': True,
        'include_dev_deps': False,
        'batch_size': 10,
        'detailed_info': True
    },
    
    # UI settings
    'ui_settings': {
        'auto_analyze_on_render': True,
        'show_progress': True,
        'log_level': 'info',
        'compact_view': False
    },
    
    # Advanced settings
    'advanced': {
        'pip_extra_args': [],
        'environment_variables': {},
        'pre_install_commands': [],
        'post_install_commands': []
    }
}

def get_default_dependency_config() -> Dict[str, Any]:
    """Get complete default config untuk dependency installer - one-liner copy"""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)

def get_minimal_config() -> Dict[str, Any]:
    """Get minimal config untuk basic functionality - one-liner selection"""
    return {
        'module_name': DEFAULT_CONFIG['module_name'],
        'selected_packages': DEFAULT_CONFIG['selected_packages'],
        'installation': DEFAULT_CONFIG['installation'],
        'analysis': DEFAULT_CONFIG['analysis']
    }

def get_installation_defaults() -> Dict[str, Any]:
    """Get installation defaults saja - one-liner"""
    return DEFAULT_CONFIG['installation'].copy()

def get_analysis_defaults() -> Dict[str, Any]:
    """Get analysis defaults saja - one-liner"""
    return DEFAULT_CONFIG['analysis'].copy()

def get_ui_defaults() -> Dict[str, Any]:
    """Get UI defaults saja - one-liner"""
    return DEFAULT_CONFIG['ui_settings'].copy()

def validate_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate config structure terhadap default - one-liner approach"""
    
    validation_result = {
        'valid': True,
        'missing_keys': [],
        'invalid_types': [],
        'warnings': []
    }
    
    # Check required top-level keys - one-liner
    required_keys = ['module_name', 'installation', 'analysis']
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        validation_result['valid'] = False
        validation_result['missing_keys'] = missing_keys
    
    # Check installation config structure - one-liner
    if 'installation' in config:
        installation_issues = _validate_installation_config(config['installation'])
        validation_result['invalid_types'].extend(installation_issues)
    
    # Check analysis config structure - one-liner
    if 'analysis' in config:
        analysis_issues = _validate_analysis_config(config['analysis'])
        validation_result['invalid_types'].extend(analysis_issues)
    
    return validation_result

def _validate_installation_config(installation_config: Dict[str, Any]) -> List[str]:
    """Validate installation config types - one-liner checks"""
    
    type_checks = [
        ('parallel_workers', int, lambda x: 1 <= x <= 10),
        ('timeout', int, lambda x: 60 <= x <= 1800),
        ('force_reinstall', bool, lambda x: True),
        ('use_cache', bool, lambda x: True)
    ]
    
    return [f"installation.{key}: expected {expected_type.__name__}, got {type(installation_config.get(key))}"
            for key, expected_type, validator in type_checks
            if key in installation_config and 
            (not isinstance(installation_config[key], expected_type) or 
             not validator(installation_config[key]))]

def _validate_analysis_config(analysis_config: Dict[str, Any]) -> List[str]:
    """Validate analysis config types - one-liner checks"""
    
    type_checks = [
        ('check_compatibility', bool),
        ('include_dev_deps', bool),
        ('batch_size', int),
        ('detailed_info', bool)
    ]
    
    return [f"analysis.{key}: expected {expected_type.__name__}, got {type(analysis_config.get(key))}"
            for key, expected_type in type_checks
            if key in analysis_config and not isinstance(analysis_config[key], expected_type)]

def merge_with_defaults(partial_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge partial config dengan defaults - one-liner deep merge"""
    import copy
    
    merged_config = copy.deepcopy(DEFAULT_CONFIG)
    
    # Deep merge dengan recursive approach - one-liner style
    _deep_merge(merged_config, partial_config)
    
    return merged_config

def _deep_merge(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """Deep merge dictionaries - one-liner recursive"""
    
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_merge(base_dict[key], value)
        else:
            base_dict[key] = value

def get_config_template() -> str:
    """Get config template sebagai YAML string untuk reference"""
    
    return """
# Dependency Installer Configuration Template
module_name: dependency_installer
version: 1.0.0

# Package selections
selected_packages: []
custom_packages: ""
auto_analyze: true

# Installation settings
installation:
  parallel_workers: 3
  force_reinstall: false
  use_cache: true
  timeout: 300

# Analysis settings  
analysis:
  check_compatibility: true
  include_dev_deps: false
  batch_size: 10
  detailed_info: true

# UI settings
ui_settings:
  auto_analyze_on_render: true
  show_progress: true
  log_level: info
  compact_view: false
    """.strip()

def create_config_from_template(**overrides) -> Dict[str, Any]:
    """Create config dari template dengan overrides - one-liner approach"""
    
    config = get_default_dependency_config()
    
    # Apply overrides dengan flat key support (e.g., 'installation.timeout': 600)
    for key, value in overrides.items():
        if '.' in key:
            # Handle nested keys seperti 'installation.timeout'
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
        else:
            config[key] = value
    
    return config

def get_preset_configs() -> Dict[str, Dict[str, Any]]:
    """Get preset configurations untuk different scenarios - one-liner presets"""
    
    return {
        'development': create_config_from_template(
            auto_analyze=True,
            **{'installation.parallel_workers': 2, 'installation.timeout': 600}
        ),
        'production': create_config_from_template(
            auto_analyze=False,
            **{'installation.parallel_workers': 4, 'installation.force_reinstall': True}
        ),
        'minimal': get_minimal_config(),
        'testing': create_config_from_template(
            auto_analyze=True,
            **{'installation.parallel_workers': 1, 'analysis.include_dev_deps': True}
        )
    }

def get_environment_specific_config(environment: str = 'development') -> Dict[str, Any]:
    """Get config untuk environment tertentu dengan fallback - one-liner"""
    presets = get_preset_configs()
    return presets.get(environment, presets['development'])

# Configuration constants untuk validation
CONFIG_CONSTRAINTS = {
    'installation': {
        'parallel_workers': {'min': 1, 'max': 10},
        'timeout': {'min': 60, 'max': 1800},
        'max_retries': {'min': 0, 'max': 5},
        'retry_delay': {'min': 0.1, 'max': 10.0}
    },
    'analysis': {
        'batch_size': {'min': 1, 'max': 50}
    }
}

def validate_config_constraints(config: Dict[str, Any]) -> List[str]:
    """Validate config constraints - one-liner constraint checking"""
    
    violations = []
    
    for section, constraints in CONFIG_CONSTRAINTS.items():
        if section in config:
            section_config = config[section]
            for key, constraint in constraints.items():
                if key in section_config:
                    value = section_config[key]
                    if 'min' in constraint and value < constraint['min']:
                        violations.append(f"{section}.{key}: {value} < minimum {constraint['min']}")
                    if 'max' in constraint and value > constraint['max']:
                        violations.append(f"{section}.{key}: {value} > maximum {constraint['max']}")
    
    return violations