"""
File: smartcash/ui/dataset/preprocessing/utils/__init__.py
Deskripsi: Integrated utils module dengan factory pattern yang konsisten dan dependency injection
"""

from typing import Dict, Any, Optional
from .config_extractor import get_config_extractor
from .validation_helper import get_validation_helper
from .dialog_manager import get_dialog_manager
from .progress_bridge import get_progress_bridge
from .ui_state_manager import get_ui_state_manager

__all__ = [
    'get_config_extractor',
    'get_validation_helper', 
    'get_dialog_manager',
    'get_progress_bridge',
    'get_ui_state_manager',
    'create_preprocessing_utils_bundle',
    'validate_utils_integration'
]

def create_preprocessing_utils_bundle(ui_components: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """
    Create complete utils bundle untuk preprocessing dengan proper integration.
    
    Args:
        ui_components: UI components dictionary
        logger: Logger instance
        
    Returns:
        Dictionary berisi semua utility instances yang terintegrasi
    """
    utils_bundle = {}
    
    try:
        # Initialize semua utilities dengan proper dependency injection
        utils_bundle['config_extractor'] = get_config_extractor(ui_components)
        utils_bundle['validation_helper'] = get_validation_helper(ui_components, logger)
        utils_bundle['dialog_manager'] = get_dialog_manager(ui_components)
        utils_bundle['progress_bridge'] = get_progress_bridge(ui_components, logger)
        utils_bundle['ui_state_manager'] = get_ui_state_manager(ui_components)
        
        # Cross-reference dependencies untuk integration
        _setup_utils_cross_references(utils_bundle, ui_components, logger)
        
        # Validate integration
        validation_result = validate_utils_integration(utils_bundle, ui_components)
        utils_bundle['integration_status'] = validation_result
        
        if logger:
            if validation_result['valid']:
                logger.debug("✅ Preprocessing utils bundle created dan terintegrasi")
            else:
                logger.warning(f"⚠️ Utils integration issues: {validation_result['issues']}")
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating utils bundle: {str(e)}")
        # Return partial bundle dengan fallbacks
        utils_bundle = _create_fallback_utils_bundle(ui_components, logger)
    
    return utils_bundle

def _setup_utils_cross_references(utils_bundle: Dict[str, Any], ui_components: Dict[str, Any], logger=None):
    """Setup cross-references antar utilities untuk better integration."""
    try:
        # Validation helper reference ke dialog manager untuk user confirmations
        if 'validation_helper' in utils_bundle and 'dialog_manager' in utils_bundle:
            utils_bundle['validation_helper'].dialog_manager = utils_bundle['dialog_manager']
        
        # Progress bridge reference ke UI state manager untuk coordinated updates
        if 'progress_bridge' in utils_bundle and 'ui_state_manager' in utils_bundle:
            utils_bundle['progress_bridge'].ui_state_manager = utils_bundle['ui_state_manager']
        
        # Dialog manager reference ke validation helper untuk smart confirmations
        if 'dialog_manager' in utils_bundle and 'validation_helper' in utils_bundle:
            utils_bundle['dialog_manager'].validation_helper = utils_bundle['validation_helper']
        
        # Config extractor reference ke validation untuk smart defaults
        if 'config_extractor' in utils_bundle and 'validation_helper' in utils_bundle:
            utils_bundle['config_extractor'].validation_helper = utils_bundle['validation_helper']
        
        if logger:
            logger.debug("🔗 Utils cross-references established")
            
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Utils cross-reference setup error: {str(e)}")

def _create_fallback_utils_bundle(ui_components: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Create fallback utils bundle dengan minimal functionality."""
    fallback_bundle = {}
    
    # Create fallback instances untuk each utility
    try:
        fallback_bundle['config_extractor'] = get_config_extractor(ui_components)
    except Exception:
        fallback_bundle['config_extractor'] = None
    
    try:
        fallback_bundle['validation_helper'] = get_validation_helper(ui_components, logger)
    except Exception:
        fallback_bundle['validation_helper'] = None
    
    try:
        fallback_bundle['dialog_manager'] = get_dialog_manager(ui_components)
    except Exception:
        fallback_bundle['dialog_manager'] = None
    
    try:
        fallback_bundle['progress_bridge'] = get_progress_bridge(ui_components, logger)
    except Exception:
        fallback_bundle['progress_bridge'] = None
    
    try:
        fallback_bundle['ui_state_manager'] = get_ui_state_manager(ui_components)
    except Exception:
        fallback_bundle['ui_state_manager'] = None
    
    fallback_bundle['integration_status'] = {
        'valid': False,
        'issues': ['Fallback bundle created due to initialization errors'],
        'available_utils': [k for k, v in fallback_bundle.items() if v is not None]
    }
    
    if logger:
        available_count = len(fallback_bundle['integration_status']['available_utils'])
        logger.warning(f"⚠️ Fallback utils bundle created: {available_count}/5 utilities available")
    
    return fallback_bundle

def validate_utils_integration(utils_bundle: Dict[str, Any], ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate integration antar utilities dan UI components.
    
    Args:
        utils_bundle: Bundle utilities yang sudah dibuat
        ui_components: UI components dictionary
        
    Returns:
        Dictionary berisi validation result
    """
    validation_result = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'available_utils': [],
        'missing_utils': [],
        'ui_component_compatibility': {}
    }
    
    # Check ketersediaan utilities
    required_utils = ['config_extractor', 'validation_helper', 'dialog_manager', 'progress_bridge', 'ui_state_manager']
    
    for util_name in required_utils:
        if util_name in utils_bundle and utils_bundle[util_name] is not None:
            validation_result['available_utils'].append(util_name)
        else:
            validation_result['missing_utils'].append(util_name)
            validation_result['issues'].append(f"Missing utility: {util_name}")
    
    # Check UI component compatibility
    critical_ui_components = ['preprocess_button', 'cleanup_button', 'check_button', 'save_button', 'reset_button']
    
    for component_name in critical_ui_components:
        if component_name in ui_components and ui_components[component_name] is not None:
            validation_result['ui_component_compatibility'][component_name] = True
        else:
            validation_result['ui_component_compatibility'][component_name] = False
            validation_result['warnings'].append(f"UI component not available: {component_name}")
    
    # Check progress tracking integration
    progress_functions = ['show_for_operation', 'update_progress', 'complete_operation', 'error_operation']
    progress_available = all(func in ui_components for func in progress_functions)
    
    if not progress_available:
        missing_progress = [func for func in progress_functions if func not in ui_components]
        validation_result['issues'].append(f"Missing progress functions: {missing_progress}")
    
    # Check confirmation area availability
    if 'confirmation_area' not in ui_components:
        validation_result['warnings'].append("Confirmation area not available - dialogs akan fallback ke standard display")
    
    # Determine overall validity
    if validation_result['missing_utils'] or validation_result['issues']:
        validation_result['valid'] = False
    
    return validation_result

def get_utils_health_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get health status dari utilities untuk debugging dan monitoring.
    
    Args:
        ui_components: UI components dictionary
        
    Returns:
        Dictionary berisi health status semua utilities
    """
    health_status = {
        'timestamp': None,
        'overall_health': 'unknown',
        'utils_status': {},
        'integration_issues': [],
        'recommendations': []
    }
    
    try:
        from datetime import datetime
        health_status['timestamp'] = datetime.now().isoformat()
        
        # Check individual utility health
        util_checks = {
            'config_extractor': _check_config_extractor_health,
            'validation_helper': _check_validation_helper_health,
            'dialog_manager': _check_dialog_manager_health,
            'progress_bridge': _check_progress_bridge_health,
            'ui_state_manager': _check_ui_state_manager_health
        }
        
        healthy_count = 0
        
        for util_name, health_check_func in util_checks.items():
            try:
                util_instance = ui_components.get(util_name)
                if util_instance:
                    health_result = health_check_func(util_instance, ui_components)
                    health_status['utils_status'][util_name] = health_result
                    
                    if health_result.get('healthy', False):
                        healthy_count += 1
                else:
                    health_status['utils_status'][util_name] = {'healthy': False, 'reason': 'Instance not found'}
            except Exception as e:
                health_status['utils_status'][util_name] = {'healthy': False, 'reason': f'Health check error: {str(e)}'}
        
        # Determine overall health
        total_utils = len(util_checks)
        if healthy_count == total_utils:
            health_status['overall_health'] = 'excellent'
        elif healthy_count >= total_utils * 0.8:
            health_status['overall_health'] = 'good'
        elif healthy_count >= total_utils * 0.6:
            health_status['overall_health'] = 'fair'
        else:
            health_status['overall_health'] = 'poor'
        
        # Generate recommendations
        if healthy_count < total_utils:
            unhealthy_utils = [name for name, status in health_status['utils_status'].items() if not status.get('healthy', False)]
            health_status['recommendations'].append(f"Reinitialize utilities: {', '.join(unhealthy_utils)}")
        
        if health_status['overall_health'] in ['fair', 'poor']:
            health_status['recommendations'].append("Consider restarting preprocessing module")
        
    except Exception as e:
        health_status['overall_health'] = 'error'
        health_status['integration_issues'].append(f"Health check error: {str(e)}")
    
    return health_status

def _check_config_extractor_health(instance, ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check health of config extractor."""
    try:
        # Test basic functionality
        config = instance.get_preprocessing_config()
        ui_config = instance.get_current_ui_config()
        
        return {
            'healthy': True,
            'capabilities': ['get_preprocessing_config', 'get_current_ui_config'],
            'last_config_keys': list(config.keys()) if config else []
        }
    except Exception as e:
        return {'healthy': False, 'reason': str(e)}

def _check_validation_helper_health(instance, ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check health of validation helper."""
    try:
        # Test basic validation
        dataset_exists, _ = instance.check_dataset_exists()
        
        return {
            'healthy': True,
            'capabilities': ['check_dataset_exists', 'check_preprocessed_data'],
            'dataset_exists': dataset_exists
        }
    except Exception as e:
        return {'healthy': False, 'reason': str(e)}

def _check_dialog_manager_health(instance, ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check health of dialog manager."""
    try:
        # Check confirmation area availability
        confirmation_area_status = instance.get_confirmation_area_status()
        
        return {
            'healthy': True,
            'capabilities': ['show_confirmation_dialog', 'show_destructive_confirmation'],
            'confirmation_area_available': confirmation_area_status.get('area_available', False)
        }
    except Exception as e:
        return {'healthy': False, 'reason': str(e)}

def _check_progress_bridge_health(instance, ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check health of progress bridge."""
    try:
        # Check progress functions availability
        required_functions = ['show_for_operation', 'update_progress', 'complete_operation']
        available_functions = [func for func in required_functions if func in ui_components]
        
        return {
            'healthy': len(available_functions) == len(required_functions),
            'capabilities': ['setup_for_operation', 'update_progress'],
            'available_progress_functions': available_functions,
            'missing_progress_functions': [func for func in required_functions if func not in available_functions]
        }
    except Exception as e:
        return {'healthy': False, 'reason': str(e)}

def _check_ui_state_manager_health(instance, ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Check health of UI state manager."""
    try:
        # Check button availability
        critical_buttons = ['preprocess_button', 'cleanup_button', 'check_button']
        available_buttons = [btn for btn in critical_buttons if btn in ui_components and ui_components[btn] is not None]
        
        running_operations = instance.get_running_operations()
        
        return {
            'healthy': len(available_buttons) >= 2,  # At least 2 critical buttons available
            'capabilities': ['can_start_operation', 'set_button_processing'],
            'available_buttons': available_buttons,
            'running_operations': running_operations
        }
    except Exception as e:
        return {'healthy': False, 'reason': str(e)}

def cleanup_utils_bundle(utils_bundle: Dict[str, Any]) -> None:
    """
    Cleanup utils bundle dan release resources.
    
    Args:
        utils_bundle: Utils bundle yang akan di-cleanup
    """
    try:
        # Cleanup individual utilities
        for util_name, util_instance in utils_bundle.items():
            if util_instance and hasattr(util_instance, 'cleanup'):
                try:
                    util_instance.cleanup()
                except Exception:
                    pass
        
        # Clear bundle
        utils_bundle.clear()
        
    except Exception:
        pass  # Silent cleanup