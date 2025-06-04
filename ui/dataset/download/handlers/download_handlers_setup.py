"""
File: smartcash/ui/dataset/download/handlers/download_handlers_setup.py
Deskripsi: Consolidated handler setup untuk download module dengan proper delegation
"""

from typing import Dict, Any
from smartcash.ui.dataset.download.handlers.download_action_handlers import setup_download_action_handlers
from smartcash.ui.dataset.download.handlers.download_config_setup import setup_download_config_handlers
from smartcash.ui.dataset.download.handlers.download_progress_setup import setup_download_progress_handlers

def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers untuk download module dengan consolidated approach."""
    logger = ui_components.get('logger')
    setup_results = {'config_handlers': False, 'progress_handlers': False, 'action_handlers': False}
    
    try:
        # Setup config handlers (environment-aware)
        ui_components = setup_download_config_handlers(ui_components, config, env)
        setup_results['config_handlers'] = True
        logger and logger.debug("✅ Config handlers setup berhasil")
        
        # Setup progress handlers (observer integration)
        ui_components = setup_download_progress_handlers(ui_components)
        setup_results['progress_handlers'] = True
        logger and logger.debug("✅ Progress handlers setup berhasil")
        
        # Setup action handlers (button handlers dengan progress integration)
        ui_components = setup_download_action_handlers(ui_components, env)
        setup_results['action_handlers'] = True
        logger and logger.debug("✅ Action handlers setup berhasil")
        
        # Store setup results
        ui_components['_handler_setup_results'] = setup_results
        
        # Log overall success
        success_count = sum(setup_results.values())
        logger and logger.info(f"🔧 Download handlers: {success_count}/3 setup berhasil")
        
    except Exception as e:
        logger and logger.error(f"❌ Error setup handlers: {str(e)}")
        ui_components['_handler_setup_error'] = str(e)
    
    return ui_components

def get_handler_setup_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get status setup handlers untuk debugging."""
    setup_results = ui_components.get('_handler_setup_results', {})
    setup_error = ui_components.get('_handler_setup_error')
    
    # Check button functionality
    button_status = {}
    for btn_key in ['download_button', 'check_button', 'cleanup_button', 'save_button', 'reset_button']:
        if btn_key in ui_components:
            button_status[btn_key] = {
                'exists': True,
                'has_onclick': hasattr(ui_components[btn_key], 'on_click'),
                'enabled': not getattr(ui_components[btn_key], 'disabled', True)
            }
        else:
            button_status[btn_key] = {'exists': False, 'has_onclick': False, 'enabled': False}
    
    # Check progress integration
    progress_methods = ['update_progress', 'complete_operation', 'error_operation', 'show_for_operation']
    progress_status = {method: method in ui_components for method in progress_methods}
    
    return {
        'setup_results': setup_results,
        'setup_error': setup_error,
        'overall_success': sum(setup_results.values()) == 3,
        'button_status': button_status,
        'progress_integration': progress_status,
        'functional_buttons': [btn for btn, status in button_status.items() if status['has_onclick']],
        'total_setup_score': int((sum(setup_results.values()) / 3) * 100)
    }

def validate_download_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate download handlers setup dengan comprehensive checks."""
    status = get_handler_setup_status(ui_components)
    
    # Critical validations
    critical_buttons = ['download_button', 'check_button']
    missing_critical = [btn for btn in critical_buttons if not status['button_status'][btn]['has_onclick']]
    
    # Progress integration validation
    required_progress = ['update_progress', 'show_for_operation']
    missing_progress = [method for method in required_progress if not status['progress_integration'][method]]
    
    validation_result = {
        'valid': len(missing_critical) == 0 and len(missing_progress) == 0,
        'critical_issues': missing_critical + missing_progress,
        'warnings': [],
        'recommendations': []
    }
    
    # Generate warnings
    if not status['overall_success']:
        validation_result['warnings'].append("Tidak semua handler setup berhasil")
    
    if missing_progress:
        validation_result['warnings'].append("Progress integration tidak lengkap")
    
    # Generate recommendations
    if missing_critical:
        validation_result['recommendations'].append(f"Setup ulang button handlers: {', '.join(missing_critical)}")
    
    if status['setup_error']:
        validation_result['recommendations'].append("Check setup error details dan restart handler setup")
    
    return validation_result

def restart_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> bool:
    """Restart download handlers jika ada yang gagal."""
    logger = ui_components.get('logger')
    
    try:
        logger and logger.info("🔄 Restarting download handlers...")
        
        # Clear previous setup results
        ui_components.pop('_handler_setup_results', None)
        ui_components.pop('_handler_setup_error', None)
        
        # Re-setup handlers
        ui_components = setup_download_handlers(ui_components, config, env)
        
        # Validate restart
        validation = validate_download_handlers(ui_components)
        
        if validation['valid']:
            logger and logger.success("✅ Download handlers restart berhasil")
            return True
        else:
            logger and logger.warning(f"⚠️ Download handlers restart parsial: {validation['critical_issues']}")
            return False
            
    except Exception as e:
        logger and logger.error(f"❌ Error restart handlers: {str(e)}")
        return False