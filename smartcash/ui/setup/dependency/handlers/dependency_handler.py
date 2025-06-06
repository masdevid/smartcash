"""
File: smartcash/ui/setup/dependency/handlers/dependency_handler.py
Deskripsi: Fixed dependency handler dengan proper button binding dan generator cleanup
"""

from typing import Dict, Any
from smartcash.ui.utils.button_state_manager import get_button_state_manager

# Import SRP handlers
from smartcash.ui.setup.dependency.handlers.installation_handler import setup_installation_handler
from smartcash.ui.setup.dependency.handlers.analysis_handler import setup_analysis_handler
from smartcash.ui.setup.dependency.handlers.status_check_handler import setup_status_check_handler

from smartcash.ui.setup.dependency.handlers.config_extractor import extract_dependency_config
from smartcash.ui.setup.dependency.handlers.config_updater import update_dependency_ui, reset_dependency_ui
from smartcash.ui.setup.dependency.utils.ui_state_utils import log_to_ui_safe

def setup_dependency_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup semua handlers dengan fixed button binding dan generator cleanup"""
    
    # Force cleanup generators terlebih dahulu
    _cleanup_existing_generators(ui_components)
    
    # Setup progress callback dengan support untuk progress tracker baru
    def progress_callback(**kwargs):
        progress_tracker = ui_components.get('progress_tracker')
        if progress_tracker:
            level = kwargs.get('type', 'overall')
            progress = kwargs.get('progress', 0)
            message = kwargs.get('message', 'Processing...')
            color = kwargs.get('color', None)
            
            # Gunakan metode yang benar sesuai API progress tracker
            if level == 'overall' or level == 'level1':
                progress_tracker.update_overall(progress, message, color)
            elif level == 'step' or level == 'level2':
                progress_tracker.update_current(progress, message, color)
        else:
            # Fallback untuk progress tracking lama
            ui_components.get('update_progress', lambda *a: None)(
                kwargs.get('type', 'overall'), 
                kwargs.get('progress', 0), 
                kwargs.get('message', 'Processing...'), 
                kwargs.get('color', None)
            )
    
    ui_components['progress_callback'] = progress_callback
    
    # Setup button state manager dengan error handling
    try:
        ui_components['button_manager'] = get_button_state_manager(ui_components)
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Button manager error: {str(e)}", "warning")
        ui_components['button_manager'] = None
    
    # Setup individual handlers dengan enhanced error handling
    _setup_handlers_with_validation(ui_components, config)
    
    # Setup config handlers dengan fixed button binding
    _setup_config_handlers_fixed(ui_components, config)
    
    # Setup auto-analyze dengan validation
    if ui_components.get('auto_analyze_on_render', True):
        _setup_auto_analyze_on_render(ui_components)
    
    return ui_components

def _cleanup_existing_generators(ui_components: Dict[str, Any]) -> None:
    """Cleanup existing generators untuk prevent RuntimeError - one-liner batch close"""
    generators_closed = 0
    try:
        generators = [v for v in ui_components.values() if hasattr(v, 'close') and hasattr(v, '__next__')]
        [gen.close() for gen in generators]
        generators_closed = len(generators)
        generators_closed > 0 and log_to_ui_safe(ui_components, f"🧹 Closed {generators_closed} generators", "info")
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Generator cleanup error: {str(e)}", "warning")



def _setup_handlers_with_validation(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup handlers dengan enhanced validation dan error handling"""
    
    handlers = [
        ('installation', setup_installation_handler),
        ('analysis', setup_analysis_handler),
        ('status_check', setup_status_check_handler)
    ]
    
    for handler_name, handler_func in handlers:
        try:
            handler_func(ui_components, config)
            log_to_ui_safe(ui_components, f"✅ {handler_name} handler setup", "debug")
        except Exception as e:
            log_to_ui_safe(ui_components, f"⚠️ {handler_name} handler error: {str(e)}", "warning")

def _setup_config_handlers_fixed(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup config handlers dengan fixed button binding dan generator cleanup"""
    
    config_handler = ui_components.get('config_handler')
    
    if not config_handler:
        log_to_ui_safe(ui_components, "⚠️ No config handler found", "warning")
        return
    
    def safe_save_config_handler(button=None):
        """Safe save handler dengan generator cleanup - one-liner protection"""
        _cleanup_existing_generators(ui_components)
        button_manager = ui_components.get('button_manager')
        
        if button_manager:
            try:
                with button_manager.config_context('save_config'):
                    success = config_handler.save_config(ui_components)
                    _handle_config_operation_result(ui_components, success, 'save')
            except Exception as e:
                log_to_ui_safe(ui_components, f"💥 Save handler error: {str(e)}", "error")
        else:
            # Fallback tanpa button manager
            if button: button.disabled = True
            try:
                success = config_handler.save_config(ui_components)
                _handle_config_operation_result(ui_components, success, 'save')
            except Exception as e:
                log_to_ui_safe(ui_components, f"💥 Save error: {str(e)}", "error")
            finally:
                if button: button.disabled = False
    
    def safe_reset_config_handler(button=None):
        """Safe reset handler dengan generator cleanup - one-liner protection"""
        _cleanup_existing_generators(ui_components)
        button_manager = ui_components.get('button_manager')
        
        if button_manager:
            try:
                with button_manager.config_context('reset_config'):
                    success = config_handler.reset_config(ui_components)
                    _handle_config_operation_result(ui_components, success, 'reset')
            except Exception as e:
                log_to_ui_safe(ui_components, f"💥 Reset handler error: {str(e)}", "error")
        else:
            # Fallback tanpa button manager
            if button: button.disabled = True
            try:
                success = config_handler.reset_config(ui_components)
                _handle_config_operation_result(ui_components, success, 'reset')
            except Exception as e:
                log_to_ui_safe(ui_components, f"💥 Reset error: {str(e)}", "error")
            finally:
                if button: button.disabled = False
    
    # Enhanced button binding dengan validation
    save_button = ui_components.get('save_button')
    reset_button = ui_components.get('reset_button')
    
    if save_button and hasattr(save_button, 'on_click'):
        # Clear existing handlers terlebih dahulu
        save_button._click_handlers.callbacks.clear()
        save_button.on_click(safe_save_config_handler)
        log_to_ui_safe(ui_components, "✅ Save button bound", "debug")
    else:
        log_to_ui_safe(ui_components, "⚠️ Save button tidak valid", "warning")
    
    if reset_button and hasattr(reset_button, 'on_click'):
        # Clear existing handlers terlebih dahulu
        reset_button._click_handlers.callbacks.clear()
        reset_button.on_click(safe_reset_config_handler)
        log_to_ui_safe(ui_components, "✅ Reset button bound", "debug")
    else:
        log_to_ui_safe(ui_components, "⚠️ Reset button tidak valid", "warning")

def _handle_config_operation_result(ui_components: Dict[str, Any], success: bool, operation: str) -> None:
    """Handle config operation result dengan enhanced feedback"""
    
    logger = ui_components.get('logger')
    
    if success:
        success_msg = f"✅ Konfigurasi berhasil {'disimpan' if operation == 'save' else 'direset'}"
        if logger:
            logger.success(success_msg)
        
        # Update status panel dengan safe handling
        try:
            status_panel = ui_components.get('status_panel')
            if status_panel:
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(status_panel, success_msg, "success")
        except Exception as e:
            log_to_ui_safe(ui_components, f"⚠️ Status panel update error: {str(e)}", "warning")
    else:
        error_msg = f"❌ Gagal {'menyimpan' if operation == 'save' else 'mereset'} konfigurasi"
        if logger:
            logger.error(error_msg)
        
        # Update status panel dengan safe handling
        try:
            status_panel = ui_components.get('status_panel')
            if status_panel:
                from smartcash.ui.components.status_panel import update_status_panel
                update_status_panel(status_panel, error_msg, "error")
        except Exception as e:
            log_to_ui_safe(ui_components, f"⚠️ Status panel update error: {str(e)}", "warning")

def _setup_auto_analyze_on_render(ui_components: Dict[str, Any]):
    """Setup auto-analyze dengan enhanced validation"""
    import threading
    import time
    
    def auto_analyze_delayed():
        """Auto analyze dengan enhanced error handling"""
        try:
            time.sleep(1)  # Delay untuk ensure UI ready
            
            logger = ui_components.get('logger')
            if logger:
                logger.info("🔍 Auto-analyzing packages after UI render...")
            
            # Validate trigger function
            trigger_analysis = ui_components.get('trigger_analysis')
            if trigger_analysis and callable(trigger_analysis):
                trigger_analysis()
                if logger:
                    logger.info("✅ Auto-analysis triggered")
            else:
                log_to_ui_safe(ui_components, "⚠️ Auto-analysis trigger tidak tersedia", "warning")
                
        except Exception as e:
            log_to_ui_safe(ui_components, f"💥 Auto-analysis error: {str(e)}", "error")
    
    # Check auto-analyze checkbox dengan validation
    auto_analyze_checkbox = ui_components.get('auto_analyze_checkbox')
    
    if auto_analyze_checkbox and hasattr(auto_analyze_checkbox, 'value') and getattr(auto_analyze_checkbox, 'value', True):
        analysis_thread = threading.Thread(target=auto_analyze_delayed, daemon=True)
        analysis_thread.start()
        log_to_ui_safe(ui_components, "🚀 Auto-analysis scheduled", "info")
    else:
        log_to_ui_safe(ui_components, "ℹ️ Auto-analysis disabled", "info")

# Enhanced utility functions dengan generator cleanup
def extract_current_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract current config dengan generator cleanup - one-liner"""
    _cleanup_existing_generators(ui_components)
    return extract_dependency_config(ui_components)

def apply_config_to_ui(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Apply config dengan generator cleanup - one-liner"""
    _cleanup_existing_generators(ui_components)
    update_dependency_ui(ui_components, config)

def reset_ui_to_defaults(ui_components: Dict[str, Any]) -> None:
    """Reset UI dengan generator cleanup - one-liner"""
    _cleanup_existing_generators(ui_components)
    reset_dependency_ui(ui_components)

def validate_ui_components(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced validate dengan generator check"""
    
    required_components = [
        'install_button', 'analyze_button', 'check_button',
        'save_button', 'reset_button', 'log_output'
    ]
    
    missing_components = [comp for comp in required_components if comp not in ui_components]
    generator_count = len([v for v in ui_components.values() if hasattr(v, '__next__')])
    button_binding_issues = len([k for k, v in ui_components.items() if 'button' in k and not hasattr(v, 'on_click')])
    
    return {
        'valid': len(missing_components) == 0 and generator_count == 0 and button_binding_issues == 0,
        'missing_components': missing_components,
        'has_config_handler': 'config_handler' in ui_components,
        'has_button_manager': 'button_manager' in ui_components,
        'has_logger': 'logger' in ui_components,
        'has_check_uncheck': 'check_uncheck_container' in ui_components,
        'generator_count': generator_count,
        'button_binding_issues': button_binding_issues
    }

def get_handlers_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced status dengan generator dan binding info"""
    
    return {
        'installation_handler': 'install_button' in ui_components and hasattr(ui_components.get('install_button'), 'on_click'),
        'analysis_handler': 'analyze_button' in ui_components and 'trigger_analysis' in ui_components,
        'status_check_handler': 'check_button' in ui_components and hasattr(ui_components.get('check_button'), 'on_click'),
        'config_handler': 'config_handler' in ui_components,
        'button_manager': 'button_manager' in ui_components,
        'progress_callback': 'progress_callback' in ui_components,
        'auto_analyze_enabled': ui_components.get('auto_analyze_on_render', False),
        'generator_count': len([v for v in ui_components.values() if hasattr(v, '__next__')]),
        'button_binding_count': len([k for k, v in ui_components.items() if 'button' in k and hasattr(v, 'on_click')])
    }

# Enhanced helper functions dengan generator cleanup
def clear_ui_outputs(ui_components: Dict[str, Any]) -> None:
    """Clear UI outputs dengan generator cleanup - one-liner"""
    _cleanup_existing_generators(ui_components)
    [widget.clear_output(wait=True) for key in ['log_output', 'status', 'confirmation_area'] 
     if (widget := ui_components.get(key)) and hasattr(widget, 'clear_output')]

def update_status_panel_safe(ui_components: Dict[str, Any], message: str, status_type: str = "info") -> None:
    """Update status panel dengan safe handling"""
    try:
        status_panel = ui_components.get('status_panel')
        if status_panel:
            from smartcash.ui.components.status_panel import update_status_panel
            update_status_panel(status_panel, message, status_type)
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Status update error: {str(e)}", "warning")

def get_button_manager_safe(ui_components: Dict[str, Any]):
    """Get button manager dengan enhanced fallback"""
    try:
        return ui_components.get('button_manager') or ui_components.setdefault('button_manager', get_button_state_manager(ui_components))
    except Exception as e:
        log_to_ui_safe(ui_components, f"⚠️ Button manager error: {str(e)}", "warning")
        return None

def update_package_count_display(ui_components: Dict[str, Any]) -> None:
    """Update package count - check/uncheck functionality removed"""
    # Check/uncheck functionality removed from handlers
    pass

# Enhanced one-liner utilities
safe_execute = lambda ui_components, func, error_msg="Operation failed": (lambda result: result if result else log_to_ui_safe(ui_components, error_msg, "error"))(func() if callable(func) else None)