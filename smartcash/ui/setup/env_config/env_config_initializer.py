"""
File: smartcash/ui/setup/env_config/env_config_initializer.py
Deskripsi: Simple initializer untuk environment configuration tanpa config yaml dependency
"""

from typing import Dict, Any
from smartcash.ui.setup.env_config.handlers.status_checker import StatusChecker
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler


class EnvConfigInitializer:
    """🔧 Simple initializer untuk environment configuration"""
    
    def __init__(self):
        pass
        
    def initialize_ui(self, **kwargs) -> Dict[str, Any]:
        """Initialize UI components dan handlers"""
        try:
            # Create UI components
            ui_components = self._create_ui_components()
            
            # Setup handlers
            ui_components = self._setup_handlers(ui_components)
            
            # Initial status check
            self._perform_initial_status_check(ui_components)
            
            return ui_components
            
        except Exception as e:
            return self._create_error_fallback(str(e))
    
    def _create_ui_components(self) -> Dict[str, Any]:
        """Buat komponen UI dengan shared components"""
        from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
        
        ui_components = create_env_config_ui()
        
        # Validasi required components
        required = ['setup_button', 'status_panel', 'progress_bar', 'progress_text', 
                   'log_accordion', 'left_summary_panel', 'right_colab_panel']
        missing = [w for w in required if w not in ui_components]
        if missing:
            raise ValueError(f"Missing UI components: {', '.join(missing)}")
        
        return ui_components
    
    def _setup_handlers(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Setup handlers dengan logger bridge"""
        from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
        
        # Setup logger
        logger = create_ui_logger_bridge(ui_components, 'env_config')
        
        # Initialize handlers
        status_checker = StatusChecker(logger)
        setup_handler = SetupHandler(logger)
        
        # Store handlers
        ui_components.update({
            '_status_checker': status_checker,
            '_setup_handler': setup_handler,
            '_logger': logger
        })
        
        # Setup button click handler
        def on_setup_click(button):
            try:
                logger.info("🚀 Starting environment setup...")
                setup_handler.run_full_setup(ui_components)
            except Exception as e:
                logger.error(f"❌ Setup failed: {str(e)}")
        
        ui_components['setup_button'].on_click(on_setup_click)
        
        return ui_components
    
    def _perform_initial_status_check(self, ui_components: Dict[str, Any]) -> None:
        """Lakukan status check awal"""
        try:
            status_checker = ui_components['_status_checker']
            status_checker.check_and_update_status(ui_components)
        except Exception as e:
            logger = ui_components.get('_logger')
            if logger:
                logger.warning(f"⚠️ Initial status check failed: {str(e)}")
    
    def _create_error_fallback(self, error_msg: str) -> Dict[str, Any]:
        """Create minimal error fallback UI"""
        import ipywidgets as widgets
        
        return {
            'status_panel': widgets.HTML(f"<p style='color: red;'>❌ Initialization failed: {error_msg}</p>"),
            'setup_button': widgets.Button(description="Retry Setup", disabled=True),
            'progress_bar': widgets.IntProgress(value=0, max=100),
            'progress_text': widgets.HTML("❌ Failed to initialize"),
            'log_accordion': widgets.Accordion([widgets.Output()]),
            'left_summary_panel': widgets.HTML(""),
            'right_colab_panel': widgets.HTML("")
        }


def initialize_env_config_ui(**kwargs):
    """Entry point untuk environment config initialization"""
    initializer = EnvConfigInitializer()
    return initializer.initialize_ui(**kwargs)