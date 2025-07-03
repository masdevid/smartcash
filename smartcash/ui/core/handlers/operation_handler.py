"""
File: smartcash/ui/core/handlers/operation_handler.py
Deskripsi: Operation handler untuk managing long-running operations dengan progress tracking,
cancellation support, dan result handling. Dioptimalkan untuk minimal overhead.
"""

from typing import Dict, Any, Optional, Callable, List, Union, TypeVar, Generic, Tuple
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future

from smartcash.ui.core.handlers.base_handler import BaseHandler

class ProgressLevel(Enum):
    """Level untuk progress tracking."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

T = TypeVar('T')

class OperationStatus(Enum):
    """Status untuk operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OperationResult(Generic[T]):
    """Result container untuk operation."""
    status: OperationStatus
    data: Optional[T] = None
    error: Optional[Exception] = None
    message: str = ""
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class OperationHandler(BaseHandler):
    """Handler untuk long-running operations.
    
    Features:
    - 🚀 Async dan sync operation support
    - 📊 Progress tracking terintegrasi
    - ⏹️ Cancellation support
    - 🔄 Result caching
    - ⚡ Thread/Process pool support
    """
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: Optional[str] = None,
                 max_workers: int = 4,
                 use_process_pool: bool = False):
        """Initialize operation handler."""
        super().__init__(module_name, parent_module)
        
        # Operation state
        self._current_operation: Optional[Future] = None
        self._operation_status = OperationStatus.PENDING
        self._last_result: Optional[OperationResult] = None
        self._cancel_requested = False
        
        # Executor setup
        self._max_workers = max_workers
        self._executor = ProcessPoolExecutor(max_workers) if use_process_pool else ThreadPoolExecutor(max_workers)
        
        self.logger.debug(f"⚡ OperationHandler initialized (workers={max_workers}, process={use_process_pool})")
    
    # === Core Operation Methods ===
    
    def execute_operation(self, 
                         operation_fn: Callable[..., T],
                         *args,
                         operation_name: str = "Operation",
                         show_progress: bool = True,
                         allow_cancel: bool = True,
                         **kwargs) -> OperationResult[T]:
        """Execute operation dengan full tracking.
        
        Args:
            operation_fn: Function untuk dijalankan
            *args: Args untuk function
            operation_name: Nama operation untuk display
            show_progress: Show progress bar
            allow_cancel: Allow cancellation
            **kwargs: Kwargs untuk function
            
        Returns:
            OperationResult dengan data/error
        """
        # Check jika ada operation yang running
        if self._operation_status == OperationStatus.RUNNING:
            return OperationResult(
                status=OperationStatus.FAILED,
                error=RuntimeError("Another operation is already running"),
                message="Operation sedang berjalan"
            )
        
        # Reset state
        self._cancel_requested = False
        self._operation_status = OperationStatus.RUNNING
        start_time = datetime.now()
        
        # Setup UI
        self.clear_outputs()
        self.update_status(f"🔄 {operation_name} dimulai...", 'info')
        if show_progress:
            self.reset_progress()
        
        try:
            # Create wrapped function dengan progress dan cancel support
            def wrapped_fn():
                return self._execute_with_tracking(
                    operation_fn, 
                    args, 
                    kwargs,
                    operation_name,
                    show_progress,
                    allow_cancel
                )
            
            # Submit ke executor
            self._current_operation = self._executor.submit(wrapped_fn)
            
            # Wait for completion
            result = self._current_operation.result()
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Create result
            if self._cancel_requested:
                operation_result = OperationResult(
                    status=OperationStatus.CANCELLED,
                    message=f"{operation_name} dibatalkan",
                    duration=duration
                )
            else:
                operation_result = OperationResult(
                    status=OperationStatus.COMPLETED,
                    data=result,
                    message=f"{operation_name} selesai dalam {duration:.1f} detik",
                    duration=duration
                )
            
            # Update UI
            self.update_status(operation_result.message, 'success' if not self._cancel_requested else 'warning')
            if show_progress:
                self.update_progress(100.0, "Selesai")
            
        except Exception as e:
            # Handle error
            duration = (datetime.now() - start_time).total_seconds()
            operation_result = OperationResult(
                status=OperationStatus.FAILED,
                error=e,
                message=f"{operation_name} gagal: {str(e)}",
                duration=duration
            )
            
            self.handle_error(e, context=operation_name)
        
        finally:
            # Update state
            self._operation_status = operation_result.status
            self._last_result = operation_result
            self._current_operation = None
        
        return operation_result
    
    def _execute_with_tracking(self,
                              operation_fn: Callable,
                              args: tuple,
                              kwargs: dict,
                              operation_name: str,
                              show_progress: bool,
                              allow_cancel: bool) -> Any:
        """Execute dengan progress dan cancel tracking."""
        # Create progress callback
        progress_callback = None
        if show_progress:
            progress_callback = lambda p, msg=None: self.update_progress(p, msg)
        
        # Create cancel check
        cancel_check = None
        if allow_cancel:
            cancel_check = lambda: self._cancel_requested
        
        # Inject callbacks jika function support
        if 'progress_callback' in operation_fn.__code__.co_varnames:
            kwargs['progress_callback'] = progress_callback
        
        if 'cancel_check' in operation_fn.__code__.co_varnames:
            kwargs['cancel_check'] = cancel_check
        
        # Execute
        return operation_fn(*args, **kwargs)
    
    # === Async Support ===
    
    async def execute_async_operation(self,
                                    operation_coro: Callable[..., T],
                                    *args,
                                    operation_name: str = "Async Operation",
                                    **kwargs) -> OperationResult[T]:
        """Execute async operation."""
        loop = asyncio.get_event_loop()
        
        # Wrap coroutine untuk tracking
        async def wrapped_coro():
            return await self._execute_async_with_tracking(
                operation_coro,
                args,
                kwargs,
                operation_name
            )
        
        # Run dalam executor
        return await loop.run_in_executor(
            None,
            lambda: asyncio.run(wrapped_coro())
        )
    
    async def _execute_async_with_tracking(self,
                                         operation_coro: Callable,
                                         args: tuple,
                                         kwargs: dict,
                                         operation_name: str) -> Any:
        """Execute async dengan tracking."""
        # Similar to sync version but dengan await
        return await operation_coro(*args, **kwargs)
    
    # === Batch Operations ===
    
    def execute_batch_operations(self,
                               operations: List[Tuple[Callable, tuple, dict]],
                               operation_name: str = "Batch Operation",
                               parallel: bool = True) -> List[OperationResult]:
        """Execute multiple operations.
        
        Args:
            operations: List of (function, args, kwargs) tuples
            operation_name: Nama untuk display
            parallel: Run parallel atau sequential
            
        Returns:
            List of OperationResult
        """
        results = []
        total = len(operations)
        
        with self.operation_context(operation_name):
            if parallel:
                # Submit semua ke executor
                futures = []
                for i, (fn, args, kwargs) in enumerate(operations):
                    future = self._executor.submit(fn, *args, **kwargs)
                    futures.append(future)
                
                # Collect results dengan progress
                for i, future in enumerate(futures):
                    try:
                        result = future.result()
                        results.append(OperationResult(
                            status=OperationStatus.COMPLETED,
                            data=result
                        ))
                    except Exception as e:
                        results.append(OperationResult(
                            status=OperationStatus.FAILED,
                            error=e
                        ))
                    
                    self.update_progress((i + 1) / total * 100, f"Completed {i + 1}/{total}")
            else:
                # Run sequential
                for i, (fn, args, kwargs) in enumerate(operations):
                    try:
                        result = fn(*args, **kwargs)
                        results.append(OperationResult(
                            status=OperationStatus.COMPLETED,
                            data=result
                        ))
                    except Exception as e:
                        results.append(OperationResult(
                            status=OperationStatus.FAILED,
                            error=e
                        ))
                    
                    self.update_progress((i + 1) / total * 100, f"Completed {i + 1}/{total}")
        
        return results
    
    # === Control Methods ===
    
    def cancel_operation(self) -> bool:
        """Cancel current operation."""
        if self._operation_status != OperationStatus.RUNNING:
            self.logger.warning("⚠️ No operation to cancel")
            return False
        
        self._cancel_requested = True
        
        if self._current_operation and not self._current_operation.done():
            # Try to cancel future
            cancelled = self._current_operation.cancel()
            if cancelled:
                self.logger.info("⏹️ Operation cancelled")
                self.update_status("Operation cancelled", 'warning')
            return cancelled
        
        return False
    
    def is_operation_running(self) -> bool:
        """Check jika ada operation running."""
        return self._operation_status == OperationStatus.RUNNING
    
    def get_last_result(self) -> Optional[OperationResult]:
        """Get last operation result."""
        return self._last_result
    
    def wait_for_operation(self, timeout: Optional[float] = None) -> Optional[OperationResult]:
        """Wait untuk current operation selesai."""
        if self._current_operation and not self._current_operation.done():
            try:
                self._current_operation.result(timeout=timeout)
            except:
                pass
        
        return self._last_result
    
    # === Cleanup ===
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown executor."""
        self._executor.shutdown(wait=wait)
        self.logger.info("⚡ Executor shutdown")
    
    def __del__(self):
        """Cleanup executor saat delete."""
        try:
            self.shutdown(wait=False)
        except:
            pass  # Ignore errors during cleanup
    
    # === Abstract Methods untuk Subclass ===
    
    @abstractmethod
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations. Override di subclass."""
        return {}
    
    def execute_named_operation(self, 
                              name: str, 
                              *args, 
                              **kwargs) -> OperationResult:
        """Execute operation by name."""
        operations = self.get_operations()
        
        if name not in operations:
            return OperationResult(
                status=OperationStatus.FAILED,
                error=ValueError(f"Unknown operation: {name}"),
                message=f"Operation '{name}' tidak ditemukan"
            )
        
        return self.execute_operation(
            operations[name],
            *args,
            operation_name=name,
            **kwargs
        )