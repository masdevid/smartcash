"""
File: smartcash/ui/core/initializers/base_initializer.py
Deskripsi: Base initializer dengan fail-fast principle
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from smartcash.ui.utils.ui_logger import get_module_logger

class BaseInitializer(ABC):
    """Base initializer dengan fail-fast principle."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        from smartcash.ui.core.shared.logger import get_enhanced_logger
        self.logger = get_enhanced_logger(f"smartcash.ui.{self.full_module_name}")
        
        self._is_initialized = False
        self._initialization_result = None
        self._error_count = 0
        self._last_error = None
        
        self.logger.debug(f"🚀 Initialized {self.__class__.__name__} for {self.full_module_name}")
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    @property
    def initialization_result(self) -> Optional[Dict[str, Any]]:
        return self._initialization_result
    
    @property
    def error_count(self) -> int:
        return self._error_count
    
    @property
    def last_error(self) -> Optional[str]:
        return self._last_error
    
    def handle_error(self, error_msg: str, exc_info: bool = False, **kwargs) -> None:
        """Handle initialization errors dengan fail-fast principle."""
        self._error_count += 1
        self._last_error = error_msg
        
        # Log error dengan context
        context = f" | Context: {kwargs}" if kwargs else ""
        self.logger.error(f"❌ {error_msg}{context}", exc_info=exc_info)
        
        # Fail-fast: raise exception
        raise RuntimeError(f"Initialization Error [{self.full_module_name}]: {error_msg}")
    
    def pre_initialize_checks(self) -> None:
        """Perform pre-initialization checks dengan fail-fast."""
        # Default implementation - can be overridden
        # Subclasses should raise exceptions if checks fail
        pass
    
    def post_initialize_cleanup(self) -> None:
        """Perform post-initialization cleanup."""
        # Default implementation - can be overridden
        pass
    
    def run_initialization(self) -> Dict[str, Any]:
        """Run full initialization process dengan fail-fast."""
        # Pre-checks
        self.pre_initialize_checks()
        
        # Main initialization
        result = self.initialize()
        
        # Validate result
        if not isinstance(result, dict):
            self.handle_error("Initialize method must return a dictionary")
        
        if not result.get('success', False):
            error_msg = result.get('error', 'Unknown initialization error')
            self.handle_error(f"Initialization failed: {error_msg}")
        
        # Mark as initialized
        self._is_initialized = True
        self._initialization_result = result
        
        # Post-cleanup
        self.post_initialize_cleanup()
        
        self.logger.info(f"✅ Successfully initialized {self.full_module_name}")
        return result
    
    def reset_state(self) -> None:
        """Reset initializer state."""
        self._is_initialized = False
        self._initialization_result = None
        self._error_count = 0
        self._last_error = None
        self.logger.debug(f"🔄 Reset state for {self.full_module_name}")
    
    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """Initialize the module (to be implemented by subclasses)."""
        pass
    
    def get_initialization_stats(self) -> Dict[str, Any]:
        """Get initialization statistics."""
        return {
            'module': self.full_module_name,
            'is_initialized': self._is_initialized,
            'error_count': self._error_count,
            'last_error': self._last_error,
            'has_result': self._initialization_result is not None
        }