"""
File: smartcash/ui/core/handlers/base_handler.py
Deskripsi: Base handler dengan fail-fast principle dan centralized error handling
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Union, List
from smartcash.ui.utils.ui_logger import get_module_logger
from smartcash.ui.core.shared.ui_component_manager import UIComponentManager

class BaseHandler(ABC):
    """Base handler dengan fail-fast principle dan centralized error handling."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        """Initialize base handler.
        
        Args:
            module_name: Nama module (e.g., 'downloader')
            parent_module: Parent module (e.g., 'dataset')
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Setup logger
        from smartcash.ui.core.shared.logger import get_enhanced_logger
        self.logger = get_enhanced_logger(f"smartcash.ui.{self.full_module_name}")
        
        # Component manager
        self._component_manager = UIComponentManager(self.full_module_name)
        
        # Internal state
        self._is_initialized = False
        self._error_count = 0
        self._last_error = None
        
        self.logger.debug(f"🚀 Initialized {self.__class__.__name__} for {self.full_module_name}")
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    @property
    def error_count(self) -> int:
        return self._error_count
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
    
    from contextlib import contextmanager

    @contextmanager
    def error_context(self, context_msg: str, fail_fast: bool = True):
        """Context manager for contextualized error handling.

        Args:
            context_msg: Description of the operation being performed.
            fail_fast: If True, re-raise errors via ``handle_error``; otherwise just log.
        """
        try:
            yield
        except Exception as e:
            # Build full message
            full_msg = f"{context_msg} failed: {e}"
            if fail_fast:
                self.handle_error(full_msg, exc_info=True)
            else:
                # Only log and store error without raising
                self._error_count += 1
                self._last_error = full_msg
                self.logger.error(f"❌ {full_msg}", exc_info=True)
        finally:
            pass

    def handle_error(self, error_msg: str, exc_info: bool = False, **kwargs) -> None:
        """Centralized error handling dengan fail-fast principle.
        
        Args:
            error_msg: Error message
            exc_info: Include exception info
            **kwargs: Additional context
            
        Raises:
            RuntimeError: Always raises untuk fail-fast behavior
        """
        self._error_count += 1
        self._last_error = error_msg
        
        # Log error dengan context
        context = f" | Context: {kwargs}" if kwargs else ""
        self.logger.error(f"❌ {error_msg}{context}", exc_info=exc_info)
        
        # Fail-fast: raise exception
        raise RuntimeError(f"Handler Error [{self.full_module_name}]: {error_msg}")
    
    def reset_error_state(self) -> None:
        """Reset error state."""
        self._error_count = 0
        self._last_error = None
        self.logger.debug(f"🔄 Reset error state for {self.full_module_name}")
    
    def get_component(self, name: str, default: Any = None) -> Any:
        """Get component from manager."""
        return self._component_manager.get_component(name, default)
    
    def add_component(self, name: str, component: Any, shared: bool = False) -> None:
        """Add component to manager."""
        if not self._component_manager.add_component(name, component, shared):
            self.handle_error(f"Failed to add component '{name}'")
    
    def remove_component(self, name: str) -> bool:
        """Remove component from manager."""
        return self._component_manager.remove_component(name)
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        return self._component_manager.get_stats()
    
    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """Initialize handler (to be implemented by subclasses)."""
        pass
    
    def cleanup(self) -> None:
        """Cleanup handler resources."""
        self._component_manager.cleanup()
        self.logger.debug(f"🧹 Cleaned up {self.__class__.__name__}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        if exc_type is not None:
            self.handle_error(f"Exception in context: {exc_val}", exc_info=True)