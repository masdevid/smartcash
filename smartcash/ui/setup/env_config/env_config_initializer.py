# File: smartcash/ui/setup/env_config/env_config_initializer.py
# Deskripsi: Main initializer untuk environment configuration - Simple orchestrator

from typing import Dict, Any
from smartcash.ui.setup.env_config.components.ui_components import create_env_config_ui
from smartcash.ui.setup.env_config.handlers.setup_handler import SetupHandler
from smartcash.ui.setup.env_config.handlers.ui_logger_handler import UILoggerHandler
from smartcash.ui.setup.env_config.handlers.system_info_handler import SystemInfoHandler

class EnvConfigInitializer:
    """🚀 Orchestrator untuk environment configuration setup"""
    
    def __init__(self):
        self.setup_handler = SetupHandler()
        self.ui_logger_handler = UILoggerHandler() 
        self.system_info_handler = SystemInfoHandler()
        
    def initialize_env_config_ui(self) -> Dict[str, Any]:
        """Inisialisasi UI environment config"""
        # Create UI components
        ui_components = create_env_config_ui()
        
        # Setup logger dengan accordion auto-open
        logger = self.ui_logger_handler.create_ui_logger(ui_components)
        ui_components['logger'] = logger
        
        # Setup event handlers  
        self._setup_event_handlers(ui_components, logger)
        
        # Update system info panel
        self.system_info_handler.update_system_info_panel(ui_components)
        
        # Initial status check
        self.setup_handler.perform_initial_status_check(ui_components, logger)
        
        return ui_components
    
    def _setup_event_handlers(self, ui_components: Dict[str, Any], logger):
        """Setup event handlers untuk UI components"""
        if 'setup_button' in ui_components:
            ui_components['setup_button'].on_click(
                lambda b: self.setup_handler.handle_setup_click(ui_components, logger)
            )

def initialize_environment_config_ui() -> Dict[str, Any]:
    """🚀 Entry point untuk environment config UI"""
    from IPython.display import display
    
    initializer = EnvConfigInitializer()
    ui_components = initializer.initialize_env_config_ui()
    
    # Display UI
    if 'ui' in ui_components:
        display(ui_components['ui'])
    
    return ui_components