"""
File: smartcash/ui/setup/dependency/handlers/dependency_handler.py

Dependency Handler Orchestrator.

This module serves as the main orchestrator for the three core handlers:
- Installation Handler
- Analysis Handler
- Status Check Handler

It provides a unified interface for managing dependencies and their states.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Optional, TypedDict

# Local imports
from smartcash.ui.setup.dependency.utils.types import UIComponents, HandlerMap

# Initialize module logger
logger = logging.getLogger('smartcash.ui.setup.dependency.handlers')

class HandlerStatus(TypedDict):
    """Status information for a handler."""
    status: str
    version: str
    last_updated: str

class HandlerConfig(TypedDict):
    """Configuration for a handler."""
    name: str
    setup_function: Callable[[UIComponents], HandlerMap]
    required_components: List[str]
    enabled: bool = True

def get_handler_logger(ui_components: Optional[UIComponents] = None) -> logging.Logger:
    """Get the appropriate logger for handlers.
    
    Args:
        ui_components: Optional UI components dictionary that might contain a logger
        
    Returns:
        Configured logger instance
    """
    if ui_components and 'logger' in ui_components:
        return ui_components['logger']
    try:
        return get_current_ui_logger() or logger
    except Exception:
        return logger

class HandlerStatus(TypedDict):
    """Status information for a handler."""
    status: str
    version: str
    last_updated: str

@dataclass
class HandlerConfig:
    """Configuration for a handler."""
    name: str
    setup_function: Callable[[UIComponents], HandlerMap]
    required_components: List[str]
    enabled: bool = True

def setup_dependency_handlers(ui_components: UIComponents) -> HandlerMap:
    """Initialize and configure all dependency handler components.
    
    This function sets up the main dependency management handlers including:
    - Installation handler
    - Analysis handler
    - Status check handler
    - Configuration management
    
    Args:
        ui_components: Dictionary containing UI components needed by the handlers
        
    Returns:
        Dictionary mapping handler names to their respective functions
        
    Raises:
        RuntimeError: If any required handler fails to initialize
    """
    logger = get_handler_logger(ui_components)
    
    try:
        # Import handler setups (lazy imports to avoid circular dependencies)
        from smartcash.ui.setup.dependency.handlers.installation_handler import setup_installation_handler
        from smartcash.ui.setup.dependency.handlers.analysis_handler import setup_analysis_handler  
        from smartcash.ui.setup.dependency.handlers.status_check_handler import setup_status_check_handler
        
        # Define handler configurations
        handler_configs = [
            HandlerConfig(
                name='installation',
                setup_function=setup_installation_handler,
                required_components=['install_button', 'package_selector']
            ),
            HandlerConfig(
                name='analysis',
                setup_function=setup_analysis_handler,
                required_components=['analyze_button', 'package_selector']
            ),
            HandlerConfig(
                name='status_check',
                setup_function=setup_status_check_handler,
                required_components=['status_check_button']
            )
        ]
        
        # Initialize handlers
        handlers: HandlerMap = {}
        
        for config in handler_configs:
            if not config.enabled:
                logger.debug(f"Skipping disabled handler: {config.name}")
                continue
                
            try:
                # Validate required components
                missing = [c for c in config.required_components 
                         if c not in ui_components]
                if missing:
                    logger.warning(
                        f"Skipping {config.name} handler - missing components: {missing}"
                    )
                    continue
                
                # Initialize handler
                handler = config.setup_function(ui_components)
                handlers.update(handler)
                logger.debug(f"Initialized {config.name} handler")
                
            except Exception as e:
                logger.error(f"Failed to initialize {config.name} handler: {str(e)}")
                logger.debug(f"Handler initialization error: {str(e)}", exc_info=True)
        
        # Add utility functions
        handlers.update({
            'extract_current_config': lambda: extract_current_config(ui_components),
            'apply_config_to_ui': lambda config: apply_config_to_ui(ui_components, config),
            'reset_ui_to_defaults': lambda: reset_ui_to_defaults(ui_components),
            'get_handlers_status': get_handlers_status
        })
        
        logger.info("✅ Successfully initialized all dependency handlers")
        return handlers
        
    except Exception as e:
        error_msg = f"Failed to initialize dependency handlers: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

def extract_current_config(ui_components: UIComponents) -> Dict[str, Any]:
    """Extract the current configuration from UI components.
    
    Args:
        ui_components: Dictionary containing UI components
        
    Returns:
        Dictionary containing the current configuration extracted from UI
        
    Example:
        config = extract_current_config(ui_components)
        # Returns: {'installation': {...}, 'analysis': {...}, ...}
    """
    logger = ui_components.get('logger', get_logger())
    
    try:
        from .config_handler import extract_dependency_config
        config = extract_dependency_config(ui_components)
        logger.debug("Successfully extracted configuration from UI")
        return config
    except ImportError as e:
        logger.error(f"Configuration handler module not found: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Failed to extract configuration: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def apply_config_to_ui(ui_components: UIComponents, config: Dict[str, Any]) -> None:
    """Apply configuration to UI components.
    
    Args:
        ui_components: Dictionary containing UI components to update
        config: Configuration dictionary to apply
        
    Raises:
        RuntimeError: If configuration application fails
    """
    logger = ui_components.get('logger', get_logger())
    
    if not config:
        logger.warning("No configuration provided to apply")
        return
        
    try:
        from .config_handler import apply_dependency_config
        apply_dependency_config(ui_components, config)
        logger.info("✅ Successfully applied configuration to UI")
    except ImportError as e:
        logger.error(f"Configuration handler module not found: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Failed to apply configuration: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def reset_ui_to_defaults(ui_components: UIComponents) -> None:
    """Reset all UI components to their default values.
    
    Args:
        ui_components: Dictionary containing UI components to reset
        
    Raises:
        RuntimeError: If reset operation fails
    """
    logger = ui_components.get('logger', get_logger())
    
    try:
        from .config_updater import reset_dependency_ui
        reset_dependency_ui(ui_components)
        logger.info("✅ Successfully reset UI to default values")
    except ImportError as e:
        logger.error(f"Configuration updater module not found: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Failed to reset UI to defaults: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e

def validate_ui_components(ui_components: UIComponents) -> Dict[str, Any]:
    """Validate that all required UI components are present.
    
    Args:
        ui_components: Dictionary of UI components to validate
        
    Returns:
        Dictionary with validation results:
        - valid: bool - Whether all required components are present
        - missing_components: List[str] - Missing component names
        - available_components: List[str] - Available component names
    """
    required_components = [
        'install_button',
        'analyze_button',
        'status_check_button',
        'progress_tracker',
        'status_panel',
        'package_selector',
        'custom_packages'
    ]
    
    # Find missing components
    missing = [comp for comp in required_components if comp not in ui_components]
    
    return {
        'valid': not bool(missing),
        'missing_components': missing,
        'available_components': list(ui_components.keys())
    }

def get_handlers_status() -> Dict[str, HandlerStatus]:
    """Get status of all handlers.
    
    Returns:
        Dictionary mapping handler names to their status information
    """
    from datetime import datetime
    
    current_time = datetime.utcnow().isoformat()
    
    return {
        'installation_handler': {
            'status': 'available',
            'version': '1.0.0',
            'last_updated': current_time
        },
        'analysis_handler': {
            'status': 'available',
            'version': '1.0.0',
            'last_updated': current_time
        },
        'status_check_handler': {
            'status': 'available',
            'version': '1.0.0',
            'last_updated': current_time
        },
        'config_handlers': {
            'status': 'available',
            'version': '1.0.0',
            'last_updated': current_time
        },
        'orchestrator': {
            'status': 'active',
            'version': '1.0.0',
            'last_updated': current_time
        }
    }