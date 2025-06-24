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
                logger.error(f"❌ Setup failed: {str(e)}")
        
        ui_components['setup_button'].on_click(on_setup_click)
        
        logger.info("🎯 Environment configuration handlers initialized")
        
    except ImportError as e:
        # Fallback handler sederhana jika handler utama tidak tersedia
        _setup_fallback_handler(ui_components)
        print(f"⚠️  Using fallback handler: {str(e)}")

def _setup_fallback_handler(ui_components: Dict[str, Any]) -> None:
    """🔄 Setup fallback handler sederhana"""
    def on_setup_click(button):
        button.description = "🔄 Setting up..."
        button.disabled = True
        
        # Update status
        ui_components['status_panel'].children = [
            widgets.HTML(value="<div style='color: orange;'>⚠️ Using fallback setup...</div>")
        ]
        
        # Reset button
        import time
        time.sleep(2)
        button.description = "🚀 Setup Environment" 
        button.disabled = False
    
    ui_components['setup_button'].on_click(on_setup_click)

def _perform_initial_status_check(ui_components: Dict[str, Any]) -> None:
    """🔍 Perform initial environment status check"""
    try:
        from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
        
        # Create logger if available
        logger = ui_components.get('logger')
        if not logger:
            # Simple fallback logger
            class SimpleLogger:
                def info(self, msg): print(msg)
                def success(self, msg): print(msg)
                def warning(self, msg): print(msg)
                def error(self, msg): print(msg)
            logger = SimpleLogger()
        
        # Run status check
        status_checker = StatusChecker(logger)
        status_checker.check_environment_status(ui_components)
        
    except ImportError:
        # Simple status display
        ui_components['status_panel'].children = [
            widgets.HTML(value="<div style='color: #666;'>📋 Ready for environment setup</div>")
        ]

def _create_error_fallback(error_msg: str) -> widgets.VBox:
    """❌ Create error fallback UI"""
    return widgets.VBox([
        widgets.HTML(
            value=f"""
            <div style="padding: 20px; background: #ffebee; border: 1px solid #f44336; border-radius: 5px;">
                <h3 style="color: #c62828; margin-top: 0;">❌ Environment Config Error</h3>
                <p>Failed to initialize environment configuration:</p>
                <code>{error_msg}</code>
                <p><strong>Solusi:</strong> Restart runtime dan coba lagi</p>
            </div>
            """,
            layout=widgets.Layout(width='100%', padding='20px')
        )
    ])

# Entry point function
def create_environment_setup_ui(config=None):
    """🔧 Entry point untuk membuat environment setup UI"""
    return initialize_env_config_ui(config)