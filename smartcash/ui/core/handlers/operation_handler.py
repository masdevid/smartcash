"""
File: smartcash/ui/core/handlers/operation_handler.py
Deskripsi: Operation handler untuk managing long-running operations dengan progress tracking,
cancellation support, dan result handling. Terintegrasi dengan OperationContainer untuk UI.
"""

from typing import Dict, Any, Optional, Callable, List, Union, TypeVar, Generic, Tuple, Type
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed

from smartcash.ui.logger import get_module_logger

from smartcash.common.threadpools import (
    process_in_parallel,
    process_with_stats,
    get_optimal_thread_count,
    optimal_io_workers,
    optimal_cpu_workers
)
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.errors import handle_errors, ErrorLevel
from smartcash.ui.components.operation_container import OperationContainer

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
    - 📊 Progress tracking terintegrasi dengan OperationContainer
    - 💬 Logging terpusat melalui OperationContainer
   - 🎯 Dialog management melalui OperationContainer
    - ⏹️ Cancellation support
    - 🔄 Result caching
    - ⚡ Thread/Process pool support
    """
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: Optional[str] = None,
                 max_workers: int = 4,
                 use_process_pool: bool = False,
                 operation_container: Optional[OperationContainer] = None):
        """Initialize operation handler.
        
        Args:
            module_name: Nama modul untuk logging
            parent_module: Nama modul induk (opsional)
            max_workers: Jumlah maksimal worker threads/processes
            use_process_pool: Gunakan ProcessPoolExecutor jika True, ThreadPoolExecutor jika False
            operation_container: Instance OperationContainer untuk UI (opsional)
        """
        super().__init__(module_name, parent_module)
        
        # Operation state
        self._current_operation: Optional[Future] = None
        self._operation_status = OperationStatus.PENDING
        self._last_result: Optional[OperationResult] = None
        self._cancel_requested = False
        self._operation_container = operation_container
        
        # Executor setup
        self._max_workers = max_workers
        self._executor = ProcessPoolExecutor(max_workers) if use_process_pool else ThreadPoolExecutor(max_workers)
        
        self.logger.debug(f"⚡ OperationHandler initialized (workers={max_workers}, process={use_process_pool}, has_container={operation_container is not None})")
    
    # === ThreadPool Integration ===
    
    def execute_parallel(
        self,
        items: List[Any],
        process_func: Callable[[Any], T],
        max_workers: Optional[int] = None,
        desc: Optional[str] = None,
        show_progress: bool = True,
        use_process_pool: bool = False
    ) -> List[T]:
        """Execute function in parallel on items using thread/process pool.
        
        Args:
            items: List of items to process
            process_func: Function to execute on each item
            max_workers: Max number of workers (default: optimal for I/O)
            desc: Description for progress display
            show_progress: Whether to show progress
            use_process_pool: Use ProcessPool instead of ThreadPool
            
        Returns:
            List of results in the same order as input items
        """
        if not items:
            return []
            
        if max_workers is None:
            max_workers = optimal_cpu_workers() if use_process_pool else optimal_io_workers()
            
        self.logger.debug(f"Starting parallel execution of {len(items)} items with {max_workers} workers")
        
        def wrapped_func(item: Any) -> T:
            if self._cancel_requested:
                raise asyncio.CancelledError("Operation was cancelled")
            return process_func(item)
            
        def progress_callback(completed: int, total: int) -> None:
            if show_progress and desc:
                self._update_progress(
                    message=f"{desc}: {completed}/{total}",
                    current=completed,
                    total=total
                )
        
        try:
            if use_process_pool:
                # For CPU-bound tasks
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(wrapped_func, item) for item in items]
                    return [f.result() for f in as_completed(futures)]
            else:
                # For I/O-bound tasks using threadpools utility
                return process_in_parallel(
                    items=items,
                    process_func=wrapped_func,
                    max_workers=max_workers,
                    desc=desc,
                    show_progress=show_progress
                )
                
        except asyncio.CancelledError:
            self.logger.warning("Parallel execution was cancelled")
            raise
        
    def execute_with_stats(
        self,
        items: List[Any],
        process_func: Callable[[Any], Dict[str, int]],
        max_workers: Optional[int] = None,
        desc: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """Execute function in parallel and collect statistics.
        
        Args:
            items: List of items to process
            process_func: Function that returns statistics dictionary
            max_workers: Max number of workers (default: optimal for I/O)
            desc: Description for progress display
            show_progress: Whether to show progress
            
        Returns:
            Dictionary of accumulated statistics
        """
        if not items:
            return {}
            
        if max_workers is None:
            max_workers = optimal_io_workers()
            
        self.logger.debug(f"Starting parallel execution with stats for {len(items)} items")
        
        def wrapped_func(item: Any) -> Dict[str, int]:
            if self._cancel_requested:
                raise asyncio.CancelledError("Operation was cancelled")
            return process_func(item)
            
        def progress_callback(completed: int, total: int) -> None:
            if show_progress and desc:
                self._update_progress(
                    message=f"{desc}: {completed}/{total}",
                    current=completed,
                    total=total
                )
        
        try:
            return process_with_stats(
                items=items,
                process_func=wrapped_func,
                max_workers=max_workers,
                desc=desc,
                show_progress=False,  # We handle progress updates ourselves
                progress_callback=progress_callback if show_progress else None
            )
        except asyncio.CancelledError:
            self.logger.warning("Parallel execution with stats was cancelled")
            raise

    # === Core Operation Methods ===
    
    @handle_errors(error_msg="Failed to execute operation", level=ErrorLevel.ERROR, reraise=True)
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
        """Execute dengan progress dan cancel tracking.
        
        Args:
            operation_fn: Fungsi yang akan dijalankan
            args: Argument posisi untuk fungsi
            kwargs: Argument keyword untuk fungsi
            operation_name: Nama operasi untuk ditampilkan
            show_progress: Tampilkan progress bar
            allow_cancel: Izinkan pembatalan operasi
            
        Returns:
            Hasil dari operasi
            
        Raises:
            asyncio.CancelledError: Jika operasi dibatalkan
            Exception: Jika terjadi error saat eksekusi
        """
        # Setup progress tracking
        progress_callbacks = {}
        
        def create_progress_callback(level: str = 'primary'):
            """Create progress callback for the specified level."""
            def callback(progress: float, message: str = "", total: Optional[int] = None):
                if self._cancel_requested and allow_cancel:
                    raise asyncio.CancelledError("Operation cancelled by user")
                    
                if show_progress:
                    self._update_progress(
                        message=message or f"Processing {operation_name}...",
                        current=progress if isinstance(progress, (int, float)) else None,
                        total=total if total is not None else 100,
                        level_name=level
                    )
            return callback
        
        # Create progress callbacks for all levels
        progress_callbacks['primary'] = create_progress_callback('primary')
        progress_callbacks['secondary'] = create_progress_callback('secondary')
        progress_callbacks['tertiary'] = create_progress_callback('tertiary')
        
        # Create cancel check
        def cancel_check() -> bool:
            """Check if operation should be cancelled."""
            if not allow_cancel:
                return False
                
            if self._cancel_requested:
                self.log("Operation cancellation requested", 'warning')
                return True
            return False
        
        # Inject callbacks if function supports them
        if hasattr(operation_fn, '__code__'):  # Check if it's a function
            # Check for progress callback parameters
            params = operation_fn.__code__.co_varnames[:operation_fn.__code__.co_argcount]
            
            # Inject progress callbacks
            if 'progress_callback' in params:
                kwargs['progress_callback'] = progress_callbacks['primary']
            
            # Inject secondary progress if supported
            if 'progress_secondary' in params:
                kwargs['progress_secondary'] = progress_callbacks['secondary']
                
            # Inject tertiary progress if supported
            if 'progress_tertiary' in params:
                kwargs['progress_tertiary'] = progress_callbacks['tertiary']
            
            # Inject cancel check if supported
            if 'cancel_check' in params:
                kwargs['cancel_check'] = cancel_check
        
        try:
            # Update initial progress
            if show_progress:
                progress_callbacks['primary'](
                    progress=0,
                    message=f"Starting {operation_name}..."
                )
            
            # Execute the operation
            result = operation_fn(*args, **kwargs)
            
            # Update progress to complete
            if show_progress:
                progress_callbacks['primary'](
                    progress=100,
                    message=f"{operation_name} completed successfully"
                )
                
            return result
            
        except asyncio.CancelledError:
            self.log(f"{operation_name} was cancelled", 'warning')
            raise
            
        except Exception as e:
            self.log(f"Error in {operation_name}: {str(e)}", 'error')
            if show_progress:
                self._update_progress(
                    message=f"Error in {operation_name}: {str(e)}",
                    level='error',
                    level_name='primary'
                )
            raise
    
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
        """Cancel current operation and clear UI components.
        
        Returns:
            bool: True if operation was successfully cancelled, False otherwise
        """
        if self._operation_status != OperationStatus.RUNNING:
            self.logger.warning("⚠️ No operation to cancel")
            return False
        
        self._cancel_requested = True
        
        if self._current_operation and not self._current_operation.done():
            # Try to cancel future
            cancelled = self._current_operation.cancel()
            if cancelled:
                self.logger.info("⏹️ Operation cancelled")
                self._operation_status = OperationStatus.CANCELLED
                
                # Clear UI components
                try:
                    # Clear progress
                    if hasattr(self, '_operation_container') and self._operation_container:
                        self._operation_container.update_progress(0, "Operation cancelled", level='warning')
                        
                    # Clear any dialogs
                    self.clear_dialog()
                    
                    # Log cancellation
                    self.log("Operation was cancelled by user", 'warning')
                    
                    # Update status
                    self.update_status("Operation cancelled", 'warning')
                    
                except Exception as e:
                    self.logger.error(f"Error cleaning up UI after cancellation: {e}")
                    
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
        self.logger.debug("⚡ Executor shutdown")
    
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
    
    def _update_progress(
        self,
        message: str = "",
        current: Optional[int] = None,
        total: Optional[int] = None,
        level: Union[ProgressLevel, str] = ProgressLevel.INFO,
        level_name: str = 'primary'
    ) -> None:
        """Update progress information.
        
        Args:
            message: Progress message
            current: Current progress value
            total: Total value for progress calculation
            level: Progress level (default: INFO)
            level_name: Progress level name ('primary', 'secondary', 'tertiary')
        """
        if self._cancel_requested:
            raise asyncio.CancelledError("Operation was cancelled")
            
        # Convert level to ProgressLevel if it's a string
        if isinstance(level, str):
            level = ProgressLevel(level.lower())
            
        # Log progress
        log_func = getattr(self.logger, level.value, self.logger.info)
        log_func(f"Progress: {message} ({current or '?'}/{total or '?'})")
        
        # Update progress state (can be used by UI components)
        self._progress = {
            'message': message,
            'current': current,
            'total': total,
            'level': level,
            'level_name': level_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update OperationContainer if available
        if hasattr(self, '_operation_container') and self._operation_container:
            try:
                # Update progress bar if we have both current and total values
                if current is not None and total is not None and total > 0:
                    progress_percent = int((current / total) * 100)
                    self._operation_container.update_progress(
                        value=progress_percent,
                        message=message,
                        level=level_name
                    )
                
                # Log the message
                self.log(message, level.value)
            except Exception as e:
                self.logger.error(f"Error updating operation container: {e}")
    
    def log(self, message: str, level: str = 'info') -> None:
        """Log a message, preferring OperationContainer to avoid duplication.
        
        Args:
            message: Message to log
            level: Log level (debug, info, warning, error, critical)
        """
        # Prefer OperationContainer to avoid duplicate logging
        if hasattr(self, '_operation_container') and self._operation_container:
            try:
                # Get module name for namespacing
                namespace = getattr(self, '_module_name', None)
                
                # Try log_message method (standard operation container method)
                if hasattr(self._operation_container, 'log_message'):
                    self._operation_container.log_message(
                        message=message, 
                        level=level,
                        namespace=namespace
                    )
                    return
                
                # Try new-style operation container with log method 
                elif hasattr(self._operation_container, 'log') and callable(getattr(self._operation_container, 'log')):
                    from smartcash.ui.components.log_accordion import LogLevel
                    level_map = {
                        'debug': LogLevel.DEBUG,
                        'info': LogLevel.INFO,
                        'warning': LogLevel.WARNING,
                        'error': LogLevel.ERROR,
                        'critical': LogLevel.ERROR,
                        'success': LogLevel.INFO
                    }
                    log_level = level_map.get(level, LogLevel.INFO)
                    self._operation_container.log(
                        message=message, 
                        level=log_level,
                        namespace=namespace
                    )
                    return
                    
            except Exception as e:
                # If operation container fails, fall back to logger
                self.logger.error(f"Error logging to operation container: {e}")
        
        # Fallback to standard logger if no operation container or on error
        valid_levels = ['debug', 'info', 'warning', 'error', 'critical']
        if level in valid_levels:
            log_func = getattr(self.logger, level)
        else:
            log_func = self.logger.info
        
        # Ensure log_func is callable before calling
        if callable(log_func):
            log_func(message)
        else:
            self.logger.info(message)
    
    def reset_progress(self) -> None:
        """Reset progress tracking to initial state."""
        try:
            # Reset internal progress state
            self._progress = {
                'message': '',
                'current': 0,
                'total': 100,
                'level': ProgressLevel.INFO,
                'level_name': 'primary',
                'timestamp': datetime.now().isoformat()
            }
            
            # Reset progress in operation container if available
            if hasattr(self, '_operation_container') and self._operation_container:
                if hasattr(self._operation_container, 'update_progress'):
                    self._operation_container.update_progress(0, "Ready to start", "primary")
                    
            self.logger.debug("Progress reset to initial state")
            
        except Exception as e:
            self.logger.error(f"Error resetting progress: {e}")
    
    def update_progress(self, progress: float, message: str = "", level: str = "primary") -> None:
        """Update progress information.
        
        Args:
            progress: Progress percentage (0-100)
            message: Progress message
            level: Progress level ('primary', 'secondary', 'tertiary')
        """
        try:
            # Update internal progress state
            self._progress = {
                'message': message,
                'current': int(progress),
                'total': 100,
                'level': ProgressLevel.INFO,
                'level_name': level,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update operation container if available
            if hasattr(self, '_operation_container') and self._operation_container:
                if hasattr(self._operation_container, 'update_progress'):
                    self._operation_container.update_progress(int(progress), message, level)
                    
        except Exception as e:
            self.logger.error(f"Error updating progress: {e}")
    
    def show_dialog(
        self,
        title: str,
        message: str,
        buttons: Optional[Dict[str, str]] = None,
        default_button: Optional[str] = None,
        width: str = "500px"
    ) -> str:
        """Show a dialog using OperationContainer.
        
        Args:
            title: Dialog title
            message: Dialog message (can include HTML)
            buttons: Dictionary of button_id: button_text
            default_button: ID of the default button
            width: Dialog width
            
        Returns:
            ID of the clicked button or empty string if no container available
        """
        if not hasattr(self, '_operation_container') or not self._operation_container:
            self.logger.warning("Cannot show dialog: No OperationContainer available")
            return ""
            
        try:
            return self._operation_container.show_dialog(
                title=title,
                message=message,
                buttons=buttons or {"ok": "OK"},
                default_button=default_button,
                width=width
            )
        except Exception as e:
            self.logger.error(f"Error showing dialog: {e}")
            return ""
    
    def show_info_dialog(self, title: str, message: str, width: str = "500px") -> None:
        """Show an info dialog.
        
        Args:
            title: Dialog title
            message: Dialog message (can include HTML)
            width: Dialog width
        """
        if not hasattr(self, '_operation_container') or not self._operation_container:
            self.logger.warning("Cannot show info dialog: No OperationContainer available")
            return
            
        try:
            self._operation_container.show_info_dialog(
                title=title,
                message=message,
                width=width
            )
        except Exception as e:
            self.logger.error(f"Error showing info dialog: {e}")
    
    def clear_dialog(self) -> None:
        """Clear any currently displayed dialog."""
        if hasattr(self, '_operation_container') and self._operation_container:
            try:
                self._operation_container.clear_dialog()
            except Exception as e:
                self.logger.error(f"Error clearing dialog: {e}")
    
    def disable_all_buttons(self, processing_message: str = "⏳ Processing...") -> Dict[str, Any]:
        """Disable all buttons and store their original state.
        
        Args:
            processing_message: Message to show on buttons during processing
            
        Returns:
            Dictionary containing original button states for restoration
        """
        button_states = {}
        
        try:
            # Common button references that UIModules typically store
            button_refs = [
                'primary_button', 'setup_button', 'install_button', 'uninstall_button',
                'update_button', 'check_button', 'save_button', 'reset_button',
                'download_button', 'process_button', 'action_button'
            ]
            
            # Get buttons from the UIModule if available
            if hasattr(self, 'get_component'):
                for button_ref in button_refs:
                    try:
                        button = self.get_component(button_ref)
                        if button and hasattr(button, 'disabled'):
                            # Store original state
                            button_states[button_ref] = {
                                'disabled': button.disabled,
                                'description': getattr(button, 'description', ''),
                                'button_style': getattr(button, 'button_style', '')
                            }
                            
                            # Disable button and update description
                            button.disabled = True
                            if hasattr(button, 'description'):
                                button.description = processing_message
                                
                    except Exception as e:
                        self.logger.debug(f"Could not disable button {button_ref}: {e}")
            
            self.logger.debug(f"Disabled {len(button_states)} buttons")
            
        except Exception as e:
            self.logger.error(f"Error disabling buttons: {e}")
        
        return button_states
    
    def enable_all_buttons(self, button_states: Dict[str, Any], 
                          success: bool = True, 
                          success_message: str = "✅ Complete",
                          error_message: str = "❌ Error") -> None:
        """Restore buttons to their original state or update based on operation result.
        
        Args:
            button_states: Dictionary of original button states from disable_all_buttons()
            success: Whether the operation was successful
            success_message: Message for successful completion
            error_message: Message for failed operations
        """
        try:
            if hasattr(self, 'get_component'):
                for button_ref, original_state in button_states.items():
                    try:
                        button = self.get_component(button_ref)
                        if button and hasattr(button, 'disabled'):
                            # Restore original disabled state
                            button.disabled = original_state.get('disabled', False)
                            
                            # Update description based on operation result
                            if hasattr(button, 'description'):
                                if success:
                                    button.description = success_message
                                    if hasattr(button, 'button_style'):
                                        button.button_style = 'success'
                                else:
                                    button.description = error_message
                                    if hasattr(button, 'button_style'):
                                        button.button_style = 'danger'
                            
                            # Restore original button style after a delay (for visual feedback)
                            def restore_original_style():
                                try:
                                    if hasattr(button, 'description'):
                                        button.description = original_state.get('description', '')
                                    if hasattr(button, 'button_style'):
                                        button.button_style = original_state.get('button_style', '')
                                except:
                                    pass  # Ignore errors during restoration
                            
                            # Schedule restoration after 2 seconds (visual feedback)
                            import threading
                            threading.Timer(2.0, restore_original_style).start()
                            
                    except Exception as e:
                        self.logger.debug(f"Could not restore button {button_ref}: {e}")
            
            self.logger.debug(f"Restored {len(button_states)} buttons")
            
        except Exception as e:
            self.logger.error(f"Error restoring buttons: {e}")
    
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