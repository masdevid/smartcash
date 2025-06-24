"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Simple initializer untuk environment configuration dengan minimal dependencies
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets

def initialize_env_config_ui(config: Optional[Dict[str, Any]] = None) -> widgets.VBox:
    """
    🚀 Initialize environment configuration UI dengan setup sederhana
    
    Args:
        config: Konfigurasi opsional untuk setup
        
    Returns:
        UI container siap pakai
    """
    try:
        # 1. Create UI components
        ui_components = _create_ui_components()
        
        # 2. Setup basic handlers (includes initial status check)
        _setup_handlers(ui_components, config or {})
        
        return ui_components['ui']
        
    except Exception as e:
        return _create_error_fallback(str(e))

def _create_ui_components() -> Dict[str, Any]:
    """🎨 Buat komponen UI dengan shared components"""
    from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
    return create_env_config_ui()

def _setup_handlers(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """🔧 Setup handlers dan logger bridge"""
    try:
        from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
        from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
        from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
        
        # Initialize handlers first with a temporary logger
        temp_logger = _create_simple_logger()
        setup_handler = SetupHandler(temp_logger)
        status_checker = StatusChecker(temp_logger)
        
        # Store handlers for later use
        ui_components['_setup_handler'] = setup_handler
        ui_components['_status_checker'] = status_checker
        
        # Setup logger bridge after UI components are ready
        logger = create_ui_logger_bridge(ui_components, 'ENV_CONFIG')
        ui_components['logger'] = logger
        
        # Update handlers with the new logger
        setup_handler.logger = logger
        status_checker.logger = logger
        
        # Setup button click handler
        def on_setup_click(button):
            try:
                logger.info("🚀 Starting environment setup...")
                setup_handler.run_full_setup(ui_components)
            except Exception as e:
                logger.error(f"❌ Setup error: {str(e)}")
                
        ui_components['setup_button'].on_click(on_setup_click)
        
        # Now that everything is set up, perform initial status check
        _perform_initial_status_check(ui_components)
        
    except Exception as e:
        ui_components['logger'] = _create_simple_logger()
        ui_components['logger'].error(f"❌ Handler setup failed: {str(e)}")

def _perform_initial_status_check(ui_components: Dict[str, Any]) -> None:
    """🔍 Perform initial status check"""
    try:
        status_checker = ui_components.get('_status_checker')
        if status_checker:
            status_checker.check_initial_status(ui_components)
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Initial status check failed: {str(e)}")

def _create_simple_logger():
    """📝 Create simple fallback logger"""
    class SimpleLogger:
        def info(self, msg): print(f"ℹ️ {msg}")
        def warning(self, msg): print(f"⚠️ {msg}")
        def error(self, msg): print(f"❌ {msg}")
        def success(self, msg): print(f"✅ {msg}")
    return SimpleLogger()

def _create_error_fallback(error_message: str, traceback: Optional[str] = None) -> widgets.VBox:
    """❌ Create error fallback UI with optional traceback
    
    Args:
        error_message: The error message to display
        traceback: Optional traceback information
        
    Returns:
        A widget containing the error message and optional traceback
    """
    from smartcash.ui.components import create_error_component
    
    # Create error component with optional traceback
    error_component = create_error_component(
        error_message=error_message,
        traceback=traceback,
        title="Environment Setup Error",
        error_type="error"
    )
    
    return error_component['container']