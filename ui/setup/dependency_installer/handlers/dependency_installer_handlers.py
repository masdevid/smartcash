"""
File: smartcash/ui/setup/dependency_installer/handlers/dependency_installer_handlers.py
Deskripsi: Updated main handlers coordinator dengan CommonInitializer pattern integration
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager

# Import SRP handlers (unchanged)
from smartcash.ui.setup.dependency_installer.handlers.installation_handler import setup_installation_handler
from smartcash.ui.setup.dependency_installer.handlers.analysis_handler import setup_analysis_handler
from smartcash.ui.setup.dependency_installer.handlers.status_check_handler import setup_status_check_handler

from smartcash.ui.setup.dependency_installer.handlers.config_extractor import extract_dependency_installer_config
from smartcash.ui.setup.dependency_installer.handlers.config_updater import update_dependency_installer_ui, reset_dependency_installer_ui

def setup_dependency_installer_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers untuk dependency installer dengan updated pattern dan check/uncheck buttons"""
    
    # Setup progress callback untuk semua handlers - one-liner
    ui_components['progress_callback'] = lambda **kwargs: ui_components.get('update_progress', lambda *a: None)(
        kwargs.get('type', 'overall'), kwargs.get('progress', 0), kwargs.get('message', 'Processing...'), kwargs.get('color', None))
    
    # Setup button state manager
    ui_components['button_manager'] = get_button_state_manager(ui_components)
    
    # Setup check/uncheck all buttons untuk package selection
    _setup_package_check_uncheck_buttons(ui_components)
    
    # Setup individual handlers dengan SRP (unchanged)
    setup_installation_handler(ui_components, config)
    setup_analysis_handler(ui_components, config)
    setup_status_check_handler(ui_components, config)
    
    # Setup config handlers menggunakan CommonInitializer pattern (simplified logging)
    _setup_config_handlers_new_pattern(ui_components, config)
    
    # Setup auto-analyze setelah UI render jika enabled
    if ui_components.get('auto_analyze_on_render', True):
        _setup_auto_analyze_on_render(ui_components)
    
    return ui_components

def _setup_package_check_uncheck_buttons(ui_components: Dict[str, Any]) -> None:
    """Setup check/uncheck all buttons untuk package selection"""
    
    try:
        from smartcash.ui.components.check_uncheck_buttons import create_package_check_uncheck_buttons
        
        # Create dan integrate check/uncheck buttons
        check_uncheck_components = create_package_check_uncheck_buttons(ui_components, show_count=True)
        
        # Add ke ui_components untuk access
        ui_components.update({
            'check_uncheck_container': check_uncheck_components['container'],
            'check_all_button': check_uncheck_components['check_all_button'],
            'uncheck_all_button': check_uncheck_components['uncheck_all_button'],
            'package_count_display': check_uncheck_components.get('count_display')
        })
        
        # Log success dengan built-in logger
        logger = ui_components.get('logger')
        logger and logger.info("✅ Check/uncheck all buttons berhasil disetup")
        
    except Exception as e:
        # Log error dengan built-in logger
        logger = ui_components.get('logger')
        logger and logger.warning(f"⚠️ Error setting up check/uncheck buttons: {str(e)}") (unchanged)
    setup_installation_handler(ui_components, config)
    setup_analysis_handler(ui_components, config)
    setup_status_check_handler(ui_components, config)
    
    # Setup config handlers menggunakan CommonInitializer pattern
    _setup_config_handlers_new_pattern(ui_components, config)
    
    # Setup auto-analyze setelah UI render jika enabled
    if ui_components.get('auto_analyze_on_render', True):
        _setup_auto_analyze_on_render(ui_components)
    
    return ui_components

def _setup_config_handlers_new_pattern(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup config handlers menggunakan pattern CommonInitializer"""
    
    # Config handler sudah disetup oleh CommonInitializer, tinggal bind button events
    config_handler = ui_components.get('config_handler')
    
    if config_handler:
        # Setup save button handler dengan config handler integration
        def save_config_handler(button=None):
            """Save config menggunakan config handler dari CommonInitializer"""
            button_manager = ui_components.get('button_manager')
            
            if button_manager:
                with button_manager.config_context('save_config'):
                    success = config_handler.save_config(ui_components)
                    _handle_config_operation_result(ui_components, success, 'save')
            else:
                # Fallback tanpa button state management
                button and setattr(button, 'disabled', True)
                try:
                    success = config_handler.save_config(ui_components)
                    _handle_config_operation_result(ui_components, success, 'save')
                finally:
                    button and setattr(button, 'disabled', False)
        
        # Setup reset button handler dengan config handler integration
        def reset_config_handler(button=None):
            """Reset config menggunakan config handler dari CommonInitializer"""
            button_manager = ui_components.get('button_manager')
            
            if button_manager:
                with button_manager.config_context('reset_config'):
                    success = config_handler.reset_config(ui_components)
                    _handle_config_operation_result(ui_components, success, 'reset')
            else:
                # Fallback tanpa button state management
                button and setattr(button, 'disabled', True)
                try:
                    success = config_handler.reset_config(ui_components)
                    _handle_config_operation_result(ui_components, success, 'reset')
                finally:
                    button and setattr(button, 'disabled', False)
        
        # Bind handlers ke buttons - one-liner
        ui_components.get('save_button') and ui_components['save_button'].on_click(save_config_handler)
        ui_components.get('reset_button') and ui_components['reset_button'].on_click(reset_config_handler)

def _handle_config_operation_result(ui_components: Dict[str, Any], success: bool, operation: str) -> None:
    """Handle hasil config operation dengan consistent feedback - simplified logging"""
    
    # Use built-in logger dari CommonInitializer
    logger = ui_components.get('logger')
    
    if success:
        success_msg = f"✅ Konfigurasi berhasil {'disimpan' if operation == 'save' else 'direset'}"
        logger and logger.success(success_msg)
        
        # Update status panel jika ada
        status_panel = ui_components.get('status_panel')
        if status_panel:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(status_panel, success_msg, "success")
    else:
        error_msg = f"❌ Gagal {'menyimpan' if operation == 'save' else 'mereset'} konfigurasi"
        logger and logger.error(error_msg)
        
        # Update status panel jika ada
        status_panel = ui_components.get('status_panel')
        if status_panel:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(status_panel, error_msg, "error")

def _setup_auto_analyze_on_render(ui_components: Dict[str, Any]):
    """Setup auto-analyze packages setelah UI render dengan simplified logging"""
    import threading
    import time
    
    def auto_analyze_delayed():
        """Auto analyze dengan delay untuk ensure UI fully rendered"""
        time.sleep(1)  # Delay 1 detik untuk ensure UI ready
        
        # Use built-in logger dari CommonInitializer
        logger = ui_components.get('logger')
        logger and logger.info("🔍 Auto-analyzing packages after UI render...")
        
        # Trigger analysis handler
        trigger_analysis = ui_components.get('trigger_analysis')
        trigger_analysis and callable(trigger_analysis) and trigger_analysis()
    
    # Check auto-analyze checkbox status
    auto_analyze_checkbox = ui_components.get('auto_analyze_checkbox')
    
    if auto_analyze_checkbox and getattr(auto_analyze_checkbox, 'value', True):
        # Run auto-analyze di background thread
        analysis_thread = threading.Thread(target=auto_analyze_delayed, daemon=True)
        analysis_thread.start()

# Utility functions untuk config operations (backward compatibility dengan simplified logging)
def extract_current_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract current config dari UI - one-liner delegation"""
    return extract_dependency_installer_config(ui_components)

def apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Apply config ke UI - one-liner delegation"""
    update_dependency_installer_ui(ui_components, config)

def reset_ui_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset UI ke defaults - one-liner delegation"""
    reset_dependency_installer_ui(ui_components)

def validate_ui_components(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Validate UI components untuk dependency installer"""
    
    required_components = [
        'install_button', 'analyze_button', 'check_button',
        'save_button', 'reset_button', 'log_output'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    
    return {
        'valid': len(missing_components) == 0,
        'missing_components': missing_components,
        'has_config_handler': 'config_handler' in ui_components,
        'has_button_manager': 'button_manager' in ui_components,
        'has_logger': 'logger' in ui_components,
        'has_check_uncheck': 'check_uncheck_container' in ui_components
    }

def get_handlers_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get status semua handlers untuk debugging - enhanced dengan check/uncheck info"""
    
    return {
        'installation_handler': 'install_button' in ui_components,
        'analysis_handler': 'analyze_button' in ui_components and 'trigger_analysis' in ui_components,
        'status_check_handler': 'check_button' in ui_components,
        'config_handler': 'config_handler' in ui_components,
        'button_manager': 'button_manager' in ui_components,
        'progress_callback': 'progress_callback' in ui_components,
        'auto_analyze_enabled': ui_components.get('auto_analyze_on_render', False),
        'check_uncheck_buttons': 'check_uncheck_container' in ui_components,
        'package_count_display': 'package_count_display' in ui_components
    }

# Helper functions untuk shared operations (simplified - menggunakan built-in logger)
def clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs dengan one-liner"""
    [widget.clear_output(wait=True) for key in ['log_output', 'status', 'confirmation_area'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def update_status_panel_safe(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan safe fallback"""
    status_panel = ui_components.get('status_panel')
    if status_panel:
        from smartcash.ui.components.status_panel import update_status_panel
        update_status_panel(status_panel, message, status_type)

def get_button_manager_safe(ui_components: Dict[str, Any]):
    """Get button manager dengan safe fallback"""
    return ui_components.get('button_manager') or ui_components.setdefault('button_manager', get_button_state_manager(ui_components))

def update_package_count_display(ui_components: Dict[str, Any]) -> None:
    """Update package count display jika ada check/uncheck buttons"""
    if 'check_uncheck_container' in ui_components:
        try:
            from smartcash.ui.components.check_uncheck_buttons import update_check_uncheck_count
            
            # Package-specific filter
            def package_filter(key: str) -> bool:
                return (key.endswith('_checkbox') and 
                       any(category in key for category in ['core', 'ml', 'data', 'ui', 'dev']) and
                       key != 'auto_analyze_checkbox')
            
            check_uncheck_components = {
                'target_prefix': 'package',
                'count_display': ui_components.get('package_count_display')
            }
            
            update_check_uncheck_count(check_uncheck_components, ui_components, package_filter)
            
        except Exception as e:
            logger = ui_components.get('logger')
            logger and logger.warning(f"⚠️ Error updating package count: {str(e)}")

# One-liner utilities dengan simplified logging
log_to_ui = lambda ui_components, message, level="info": ui_components.get('logger') and getattr(ui_components['logger'], level, ui_components['logger'].info)(message)
safe_execute = lambda ui_components, func, error_msg="Operation failed": (lambda result: result if result else log_to_ui(ui_components, error_msg, "error"))(func() if callable(func) else None)