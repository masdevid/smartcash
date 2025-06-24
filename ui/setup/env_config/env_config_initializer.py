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
        
        # 2. Setup basic handlers
        _setup_handlers(ui_components, config or {})
        
        # 3. Initial status check
        _perform_initial_status_check(ui_components)
        
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
        
        # Setup logger bridge
        logger = create_ui_logger_bridge(ui_components, 'ENV_CONFIG')
        ui_components['logger'] = logger
        
        # Initialize handlers
        setup_handler = SetupHandler(logger)
        status_checker = StatusChecker(logger)
        
        # Store handlers
        ui_components['_setup_handler'] = setup_handler
        ui_components['_status_checker'] = status_checker
        
        # Setup button click handler
        def on_setup_click(button):
            try:
                logger.info("🚀 Starting environment setup...")
                setup_handler.run_full_setup(ui_components)
            except Exception as e:
                logger.error(f"❌ Setup error: {str(e)}")
                
        ui_components['setup_button'].on_click(on_setup_click)
        
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

def _create_error_fallback(error_message: str) -> widgets.VBox:
    """❌ Create error fallback UI"""
    error_widget = widgets.HTML(
        value=f"""
        <div style="background: #ffebee; border: 1px solid #f44336; border-radius: 4px; padding: 15px; margin: 10px 0;">
            <h3 style="color: #d32f2f; margin-top: 0;">❌ Environment Setup Error</h3>
            <p><strong>Error:</strong> {error_message}</p>
            <p><em>Silakan coba refresh cell atau periksa dependencies.</em></p>
        </div>
        """,
        layout=widgets.Layout(width='100%')
    )
    
    return widgets.VBox([error_widget], layout=widgets.Layout(width='100%', padding='10px'))