"""
File: smartcash/utils/cell_header_utils.py
Utility functions for consistent Jupyter/Colab notebook cell setup
"""

import sys
import os
import importlib
import atexit
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

def setup_notebook_environment(
    cell_name: str,
    config_path: Optional[str] = None,
    create_dirs: Optional[List[str]] = None,
    register_cleanup: bool = True,
    log_config: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Setup standard notebook environment with configuration and logging.
    
    Args:
        cell_name: Unique cell identifier
        config_path: Path to configuration file
        create_dirs: Additional directories to create
        register_cleanup: Auto-register cleanup handler
        log_config: Custom logging configuration
    
    Returns:
        Tuple of (environment components, configuration)
    """
    # Ensure smartcash is in path
    if not any('smartcash' in p for p in sys.path):
        sys.path.append('.')
    
    # Create standard directories
    os.makedirs("configs", exist_ok=True)
    os.makedirs("smartcash/ui_components", exist_ok=True)
    os.makedirs("smartcash/ui_handlers", exist_ok=True)
    
    # Create additional directories
    if create_dirs:
        for directory in create_dirs:
            os.makedirs(directory, exist_ok=True)
    
    # Initialize components
    env = {}
    config = {}
    
    try:
        # Dynamic imports to prevent early dependency loading
        from smartcash.utils.logging_factory import LoggingFactory
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.observer.observer_manager import ObserverManager
        
        # Configure LoggingFactory
        default_logging_config = {
            'logs_dir': 'logs',
            'log_level': 'INFO',
            'use_emojis': True,
            'log_to_file': True,
            'log_to_console': False
        }
        # Merge default config with provided config
        logging_config = {**default_logging_config, **(log_config or {})}
        LoggingFactory.configure(logging_config)

        # Setup logger using LoggingFactory
        env['logger'] = LoggingFactory.get_logger(cell_name)
        
        # Setup config manager
        config_manager = get_config_manager(env['logger'])
        env['config_manager'] = config_manager
        
        # Load configuration
        config = config_manager.load_config(
            filename=config_path, 
            logger=env['logger']
        )
        
        # Setup observer manager
        observer_manager = ObserverManager(auto_register=True)
        env['observer_manager'] = observer_manager
        
        # Create cleanup function
        if register_cleanup:
            def cleanup():
                try:
                    observer_manager.unregister_all()
                except Exception as e:
                    env['logger'].error(f"❌ Cleanup error: {str(e)}")
            
            atexit.register(cleanup)
            env['cleanup'] = cleanup
        
    except ImportError as e:
        print(f"⚠️ Some components unavailable: {str(e)}")
    
    return env, config

def setup_ui_component(
    env: Dict[str, Any], 
    config: Dict[str, Any], 
    component_name: str
) -> Dict[str, Any]:
    """
    Setup UI component and its handler.
    
    Args:
        env: Environment dictionary from setup_notebook_environment
        config: Configuration dictionary
        component_name: Name of UI component to load
    
    Returns:
        Dictionary of UI components
    """
    # Fallback UI for import errors
    fallback_ui = {
        'ui': f'<h3>⚠️ {component_name.capitalize()} UI Unavailable</h3>'
    }
    
    try:
        # Dynamically import UI component and handler
        ui_module = importlib.import_module(f"smartcash.ui_components.{component_name}")
        handler_module = importlib.import_module(f"smartcash.ui_handlers.{component_name}")
        
        # Get creation and setup functions
        ui_create_func = getattr(ui_module, f"create_{component_name}_ui")
        handler_setup_func = getattr(handler_module, f"setup_{component_name}_handlers")
        
        # Create and setup UI
        ui_components = ui_create_func()
        ui_components = handler_setup_func(ui_components, config)
        
        # Add cleanup to atexit if available
        if 'cleanup' in ui_components and callable(ui_components['cleanup']):
            atexit.register(ui_components['cleanup'])
        
        # Log success
        
        return ui_components
    
    except Exception as e:
        # Log and return fallback
        if env.get('logger'):
            env['logger'].error(f"❌ {component_name.capitalize()} UI setup failed: {str(e)}")
        return fallback_ui

def display_ui(ui_components: Dict[str, Any]) -> None:
    """
    Display UI component safely.
    
    Args:
        ui_components: Dictionary of UI components
    """
    try:
        from IPython.display import display, HTML
        
        if 'ui' in ui_components:
            display(ui_components['ui'])
        else:
            display(HTML("<h3>⚠️ Invalid UI Components</h3>"))
    
    except ImportError:
        print("❌ IPython display not available")
    except Exception as e:
        print(f"❌ UI Display Error: {str(e)}")

def create_minimal_cell(
    component_name: str, 
    config_path: Optional[str] = None,
    log_level: str = 'INFO'
) -> str:
    """
    Generate minimal notebook cell code.
    
    Args:
        component_name: UI component name
        config_path: Optional configuration file path
        log_level: Logging level for environment setup
    
    Returns:
        Minimal cell setup code
    """
    config_line = f'config_path="{config_path}"' if config_path else 'config_path=None'
    log_level_line = f'log_level="{log_level}"'
    
    return f"""# Cell: {component_name.capitalize()}
from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="{component_name}",
    {config_line},
    {log_level_line}
)

# Setup UI component
ui_components = setup_ui_component(env, config, "{component_name}")

# Display UI
display_ui(ui_components)
"""