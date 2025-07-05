"""
File: smartcash/ui/core/initializers/base_initializer.py
Deskripsi: Base initializer dengan fail-fast principle dan centralized error handling
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar, Callable

from smartcash.ui.core.errors import (
    SmartCashUIError,
    ErrorContext,
    handle_errors,
    safe_component_operation,
    get_error_handler
)

T = TypeVar('T')

class BaseInitializer(ABC):
    """Base initializer dengan fail-fast principle."""
    
    def __init__(self, module_name: str, parent_module: str = None):
        self.module_name = module_name
        self.parent_module = parent_module
        self.full_module_name = f"{parent_module}.{module_name}" if parent_module else module_name
        
        # Initialize error handler and context
        self._error_handler = get_error_handler()
        self._error_context = ErrorContext(
            component=self.__class__.__name__,
            operation="__init__",
            details={
                'module_name': module_name,
                'parent_module': parent_module
            }
        )
        
        # Initialize logger through error handler
        self.logger = self._error_handler.get_logger(f"smartcash.ui.{self.full_module_name}")
        
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
    
    @handle_errors(context_attr='_error_context')
    def handle_error(self, error_msg: str, exc_info: bool = False, **kwargs) -> None:
        """Handle initialization errors dengan fail-fast principle.
        
        Args:
            error_msg: Pesan error
            exc_info: Jika True, log traceback exception
            **kwargs: Konteks tambahan untuk error
        """
        self._error_count += 1
        self._last_error = error_msg
        
        # Update error context
        self._error_context.details.update(kwargs)
        
        # Log error melalui error handler
        error_context = ErrorContext(
            component=self.__class__.__name__,
            operation="handle_error",
            details={
                'error_message': error_msg,
                **kwargs
            }
        )
        
        if exc_info:
            self._error_handler.handle_exception(
                error_msg,
                error_level='ERROR',
                context=error_context,
                exc_info=True
            )
        else:
            self._error_handler.log_error(
                error_msg,
                error_level='ERROR',
                context=error_context
            )
    
    def pre_initialize_checks(self) -> None:
        """Perform pre-initialization checks dengan fail-fast."""
        # Default implementation - can be overridden
        # Subclasses should raise exceptions if checks fail
        pass
    
    def post_initialize_cleanup(self) -> None:
        """Perform post-initialization cleanup."""
        # Default implementation - can be overridden
        pass
    
    @handle_errors(
        error_msg="Failed to initialize component",
        level=ErrorLevel.ERROR,
        reraise=True,
        log_error=True,
        create_ui=True
    )
    def initialize(self, *args, **kwargs) -> Dict[str, Any]:
        """Initialize the module dengan fail-fast principle.
        
        Returns:
            Dict[str, Any]: Hasil inisialisasi
            
        Raises:
            SmartCashUIError: Jika terjadi kesalahan saat inisialisasi
        """
        if self._is_initialized:
            self.logger.warning("Already initialized")
            return self._initialization_result
            
        try:
            # Update context for the operation
            self._error_context.operation = "initialize"
            self._error_context.details.update({
                'initialization_args': args,
                'initialization_kwargs': kwargs
            })
            
            self.logger.info(f"🚀 Starting initialization of {self.module_name}")
            self._initialization_result = self._initialize_impl(*args, **kwargs)
            self._is_initialized = True
            self.logger.info(f"✅ Successfully initialized {self.module_name}")
            
            return self._initialization_result
            
        except SmartCashUIError:
            # Re-raise our custom errors
            raise
            
        except Exception as e:
            error_msg = f"Failed to initialize {self.module_name}"
            self.handle_error(error_msg, exc_info=True, error_details=str(e))
            raise SmartCashUIError(
                error_msg,
                context=self._error_context
            ) from e
    
    def reset_state(self) -> None:
        """Reset initializer state."""
        self._is_initialized = False
        self._initialization_result = None
        self._error_count = 0
        self._last_error = None
        self.logger.debug(f"🔄 Reset state for {self.full_module_name}")
    
    @abstractmethod
    def _initialize_impl(self, *args, **kwargs) -> Dict[str, Any]:
        """Implementation of initialization logic.
        
        Returns:
            Dict[str, Any]: Hasil inisialisasi
            
        Raises:
            Exception: Jika terjadi kesalahan saat inisialisasi
        """
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