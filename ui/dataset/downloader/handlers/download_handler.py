"""
Orchestrator for dataset downloader operations.

This module provides the main entry point for setting up all downloader handlers.
It acts as a thin orchestrator that delegates to specialized handler classes.
"""
from typing import Dict, Any
from .orchestrator import setup_download_handlers as setup_handlers


def setup_download_handlers(ui_components: Dict[str, Any], config: Dict[str, Any], env=None) -> Dict[str, Any]:
    """Setup all downloader handlers.
    
    This is the main entry point that initializes all handlers.
    """
    if '_downloader_orchestrator' not in ui_components:
        setup_handlers(ui_components, config)
    return ui_components


def setup_check_handler(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup check handler with backend scanner.
    
    This is a backward compatibility wrapper that delegates to the new handler.
    """
    if '_downloader_orchestrator' not in ui_components:
        setup_handlers(ui_components, config)


def setup_cleanup_handler(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Setup cleanup handler with confirmation dialog.
    
    This is a backward compatibility wrapper that delegates to the new handler.
    """
    if '_downloader_orchestrator' not in ui_components:
        setup_handlers(ui_components, config)


def setup_config_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup save/reset handlers.
    
    This is a backward compatibility wrapper that delegates to the new handler.
    """
    if '_downloader_orchestrator' not in ui_components:
        setup_handlers(ui_components, {})