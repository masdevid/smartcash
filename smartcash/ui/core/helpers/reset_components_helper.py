"""
File: smartcash/ui/core/helpers/reset_components_helper.py
Deskripsi: Helper utilities untuk reset UI components dengan safe null checks
"""

from typing import Dict, Any, Optional, List, Union
from smartcash.ui.utils.ui_logger import get_module_logger

def safe_reset_ui_components(ui_components: Dict[str, Any], 
                           component_names: Optional[List[str]] = None,
                           logger_name: str = "reset_helper") -> Dict[str, bool]:
    """Reset UI components dengan comprehensive null checks dan error handling.
    
    Args:
        ui_components: Dictionary berisi UI components
        component_names: List of component names to reset (None = reset all)
        logger_name: Logger name untuk tracking
        
    Returns:
        Dict dengan component name sebagai key dan success status sebagai value
    """
    from smartcash.ui.core.shared.logger import get_enhanced_logger
    logger = get_enhanced_logger(logger_name)
    results = {}
    
    if not ui_components:
        logger.warning("⚠️ UI components dictionary kosong atau None")
        return results
    
    # Determine target components
    target_components = component_names if component_names else list(ui_components.keys())
    
    for component_name in target_components:
        try:
            # Check if component exists
            if component_name not in ui_components:
                logger.warning(f"⚠️ Component '{component_name}' tidak ditemukan, skip reset")
                results[component_name] = False
                continue
            
            component = ui_components[component_name]
            
            # Check if component is None
            if component is None:
                logger.warning(f"⚠️ Component '{component_name}' adalah None, skip reset")
                results[component_name] = False
                continue
            
            # Reset component based on type
            success = _reset_component_by_type(component_name, component, logger)
            results[component_name] = success
            
        except Exception as e:
            logger.error(f"❌ Error resetting component '{component_name}': {e}")
            results[component_name] = False
    
    # Summary logging
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    if success_count > 0:
        logger.info(f"✅ Reset {success_count}/{total_count} components")
    else:
        logger.warning(f"⚠️ Tidak ada component yang berhasil direset")
    
    return results

def _reset_component_by_type(component_name: str, component: Any, logger) -> bool:
    """Reset component berdasarkan type dan available methods."""
    
    # Progress tracker components
    if component_name in ['progress_tracker', 'progress_bar']:
        return _reset_progress_component(component, logger)
    
    # Log components
    elif component_name in ['log_output', 'log_accordion', 'log_components']:
        return _reset_log_component(component, logger)
    
    # Dialog components
    elif component_name in ['confirmation_area', 'dialog_area', 'error_area']:
        return _reset_dialog_component(component, logger)
    
    # Summary components
    elif component_name in ['summary_container', 'summary_output']:
        return _reset_summary_component(component, logger)
    
    # Generic reset
    else:
        return _reset_generic_component(component, logger)

def _reset_progress_component(component: Any, logger) -> bool:
    """Reset progress component dengan safe method check."""
    try:
        if hasattr(component, 'reset') and callable(component.reset):
            component.reset()
            return True
        elif hasattr(component, 'hide') and callable(component.hide):
            component.hide()
            return True
        elif hasattr(component, 'value') and hasattr(component, 'max'):
            component.value = 0
            return True
        
        logger.debug(f"⚠️ Progress component tidak memiliki reset method yang dikenali")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error resetting progress component: {e}")
        return False

def _reset_log_component(component: Any, logger) -> bool:
    """Reset log component dengan berbagai fallback methods."""
    try:
        # Try log-specific methods first
        if hasattr(component, 'clear_logs') and callable(component.clear_logs):
            component.clear_logs()
            return True
        elif hasattr(component, 'clear_output') and callable(component.clear_output):
            component.clear_output(wait=True)
            return True
        elif isinstance(component, dict):
            # Handle log_accordion structure
            success = False
            for key in ['log_output', 'output', 'accordion']:
                if key in component and component[key] is not None:
                    log_widget = component[key]
                    if hasattr(log_widget, 'clear_output') and callable(log_widget.clear_output):
                        log_widget.clear_output(wait=True)
                        success = True
                        break
            return success
        elif hasattr(component, 'value'):
            component.value = ""
            return True
        
        logger.debug(f"⚠️ Log component tidak memiliki clear method yang dikenali")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error resetting log component: {e}")
        return False

def _reset_dialog_component(component: Any, logger) -> bool:
    """Reset dialog component dengan safe clearing."""
    try:
        # Try dialog-specific methods
        if hasattr(component, 'clear_dialog') and callable(component.clear_dialog):
            component.clear_dialog()
            return True
        elif hasattr(component, 'hide') and callable(component.hide):
            component.hide()
            return True
        elif hasattr(component, 'clear') and callable(component.clear):
            component.clear()
            return True
        elif hasattr(component, 'children') and hasattr(component, 'layout'):
            component.children = []
            component.layout.display = 'none'
            return True
        elif hasattr(component, 'value'):
            component.value = ""
            return True
        
        logger.debug(f"⚠️ Dialog component tidak memiliki clear method yang dikenali")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error resetting dialog component: {e}")
        return False

def _reset_summary_component(component: Any, logger) -> bool:
    """Reset summary component dengan content clearing."""
    try:
        # Try summary-specific methods
        if hasattr(component, 'clear_content') and callable(component.clear_content):
            component.clear_content()
            return True
        elif hasattr(component, 'set_content') and callable(component.set_content):
            component.set_content("")
            return True
        elif hasattr(component, 'clear_output') and callable(component.clear_output):
            component.clear_output(wait=True)
            return True
        elif hasattr(component, 'value'):
            component.value = ""
            return True
        
        logger.debug(f"⚠️ Summary component tidak memiliki clear method yang dikenali")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error resetting summary component: {e}")
        return False

def _reset_generic_component(component: Any, logger) -> bool:
    """Reset generic component dengan fallback methods."""
    try:
        # Try common reset methods
        if hasattr(component, 'reset') and callable(component.reset):
            component.reset()
            return True
        elif hasattr(component, 'clear') and callable(component.clear):
            component.clear()
            return True
        elif hasattr(component, 'clear_output') and callable(component.clear_output):
            component.clear_output(wait=True)
            return True
        elif hasattr(component, 'value'):
            # Handle different value types
            if isinstance(component.value, str):
                component.value = ""
            elif isinstance(component.value, (int, float)):
                component.value = 0
            elif isinstance(component.value, bool):
                component.value = False
            elif isinstance(component.value, list):
                component.value = []
            elif isinstance(component.value, dict):
                component.value = {}
            return True
        
        logger.debug(f"⚠️ Generic component tidak memiliki reset method yang dikenali")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error resetting generic component: {e}")
        return False

def check_component_availability(ui_components: Dict[str, Any], 
                               required_components: List[str],
                               logger_name: str = "component_checker") -> Dict[str, bool]:
    """Check availability dari required components.
    
    Args:
        ui_components: Dictionary berisi UI components
        required_components: List of required component names
        logger_name: Logger name
        
    Returns:
        Dict dengan component name sebagai key dan availability sebagai value
    """
    from smartcash.ui.core.shared.logger import get_enhanced_logger
    logger = get_enhanced_logger(logger_name)
    availability = {}
    
    if not ui_components:
        logger.warning("⚠️ UI components dictionary kosong atau None")
        return {comp: False for comp in required_components}
    
    for component_name in required_components:
        try:
            # Check existence
            exists = component_name in ui_components
            
            # Check if not None
            not_none = exists and ui_components[component_name] is not None
            
            # Check if has expected methods (basic validation)
            has_methods = not_none and _has_expected_methods(ui_components[component_name])
            
            availability[component_name] = exists and not_none and has_methods
            
            if not availability[component_name]:
                if not exists:
                    logger.warning(f"⚠️ Component '{component_name}' tidak ditemukan")
                elif not not_none:
                    logger.warning(f"⚠️ Component '{component_name}' adalah None")
                else:
                    logger.debug(f"ℹ️ Component '{component_name}' tersedia tapi tanpa expected methods")
            
        except Exception as e:
            logger.error(f"❌ Error checking component '{component_name}': {e}")
            availability[component_name] = False
    
    return availability

def _has_expected_methods(component: Any) -> bool:
    """Check if component has expected methods untuk basic operations."""
    # Check for any common methods that indicate it's a proper widget
    common_methods = ['layout', 'observe', 'close', 'value', 'children', 'reset', 'clear']
    
    return any(hasattr(component, method) for method in common_methods)