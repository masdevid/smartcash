"""
File: smartcash/ui/dataset/preprocessing/handlers/base_handler.py
Deskripsi: Fixed base handler dengan proper progress tracker integration dan API callback compatibility
"""

from abc import ABC, abstractmethod
import traceback
from typing import Dict, Any, Optional, Callable, List, Type, Union
from smartcash.ui.utils import with_error_handling, ErrorHandler
from smartcash.ui.dataset.preprocessing import utils as ui_utils
from smartcash.ui.dataset.preprocessing.utils.ui_utils import update_status_panel_enhanced
from smartcash.common.logger import get_logger

class BasePreprocessingHandler(ABC):
    """Base class untuk handlers dengan proper progress tracker integration"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger_bridge', get_logger(self.__class__.__name__))
        self.error_handler = ErrorHandler()
        self.initializer = None  # Will be set by CommonInitializer
        
        # Ensure we have required components
        if not self.logger:
            raise ValueError("Logger bridge required in UI components")
    
    # === ABSTRACT METHODS ===
    
    @abstractmethod
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup handlers specific untuk handler type ini"""
        pass
    
    # === ERROR HANDLING ===
    
    def _handle_error(self, error: Exception, context: str = "Operation failed") -> None:
        """Handle and propagate errors to CommonInitializer
        
        Args:
            error: The exception that was raised
            context: Context about where the error occurred
        """
        error_msg = f"{context}: {str(error)}"
        
        # Log the error with traceback
        self.logger.error(error_msg, exc_info=True)
        
        # Update UI status
        update_status_panel_enhanced(self.ui_components, f"❌ {context}", 'error')
        
        # If we have access to the initializer, let it handle the error
        if hasattr(self, 'initializer') and self.initializer:
            self.initializer.handle_error(error, context=context)
        else:
            # Otherwise, re-raise to propagate up
            raise error
    
    def setup_with_initializer(self, initializer):
        """Set the initializer for error propagation"""
        self.initializer = initializer
    
    # === FIXED PROGRESS TRACKER ACCESS ===
    
    def get_progress_tracker(self) -> Optional[Any]:
        """Get progress tracker dengan multiple fallback paths"""
        # Primary: direct access
        tracker = self.ui_components.get('progress_tracker')
        if tracker and hasattr(tracker, 'update_overall'):
            return tracker
        
        # Secondary: alias access
        tracker = self.ui_components.get('progress')
        if tracker and hasattr(tracker, 'update_overall'):
            return tracker
        
        # Tertiary: nested access
        if 'progress_components' in self.ui_components:
            tracker = self.ui_components['progress_components'].get('tracker')
            if tracker and hasattr(tracker, 'update_overall'):
                return tracker
        
        return None
    
    def create_progress_callback(self) -> Callable[[str, int, int, str], None]:
        """Create API-compatible progress callback dengan proper tracker integration"""
        def progress_callback(level: str, current: int, total: int, message: str = "") -> None:
            try:
                progress_tracker = self.get_progress_tracker()
                if not progress_tracker or total <= 0:
                    # Fallback ke log only jika tracker tidak ada
                    if current % 20 == 0:  # Log setiap 20%
                        progress_percent = int((current / total) * 100) if total > 0 else 0
                        self.logger.info(f"📊 {message} ({progress_percent}%)")
                    return
                
                progress_percent = max(0, min(100, int((current / total) * 100)))
                
                # Map level ke method yang sesuai
                level_method_map = {
                    'overall': 'update_overall',
                    'current': 'update_current',
                    'step': 'update_step',
                    'batch': 'update_current',  # Map batch ke current
                    'operation': 'update_overall',  # Map operation ke overall
                    'phase': 'update_overall'  # Map phase ke overall
                }
                
                method_name = level_method_map.get(level, 'update_overall')
                
                if hasattr(progress_tracker, method_name):
                    method = getattr(progress_tracker, method_name)
                    method(progress_percent, message)
                    
                    # Show tracker jika belum visible
                    if hasattr(progress_tracker, 'show') and not getattr(progress_tracker, 'is_visible', True):
                        progress_tracker.show()
                
                # Log milestone progress
                if progress_percent % 25 == 0 and progress_percent > 0:  # Log setiap 25%
                    log_message = f"{message} ({progress_percent}%)" if message else f"Progress: {progress_percent}%"
                    self.logger.info(f"📊 {log_message}")
                    
            except Exception as e:
                # Fallback logging jika progress tracker error
                self.logger.debug(f"🔍 Progress callback error: {str(e)}")
                if current % 25 == 0:  # Minimal fallback logging
                    progress_percent = int((current / total) * 100) if total > 0 else 0
                    self.logger.info(f"📊 {message} ({progress_percent}%)")
        
        return progress_callback
    
    # === BUTTON ACCESS ===
    
    def get_button(self, button_id: str) -> Optional[Any]:
        """Get button dengan proper fallback mapping"""
        # Primary: direct access
        if button := self.ui_components.get(button_id):
            return button
        
        # Secondary: action_buttons mapping
        action_buttons = self.ui_components.get('action_buttons', {})
        if button := action_buttons.get(button_id):
            return button
        
        # Button mapping untuk different naming conventions
        button_mappings = {
            'preprocess_btn': ['preprocess_button', 'primary', 'mulai_preprocessing'],
            'check_btn': ['check_button', 'secondary_0', 'check_dataset'],
            'cleanup_btn': ['cleanup_button', 'secondary_1', 'cleanup'],
            'save_button': ['save_btn'],
            'reset_button': ['reset_btn']
        }
        
        # Check aliases
        for alias in button_mappings.get(button_id, []):
            if button := (self.ui_components.get(alias) or action_buttons.get(alias)):
                return button
        
        return None
    
    def setup_button_handler(self, button_id: str, handler_func: Callable, 
                           error_operation: str) -> Optional[Callable]:
        """Setup button handler dengan proper mapping"""
        button = self.get_button(button_id)
        if not button:
            self.logger.debug(f"🔍 Button {button_id} tidak ditemukan")
            return None
        
        # Clear existing handlers
        if hasattr(button, '_click_handlers') and hasattr(button._click_handlers, 'callbacks'):
            button._click_handlers.callbacks.clear()
        
        @button.on_click
        @with_error_handling(
            error_handler=self.error_handler,
            component=self.__class__.__name__,
            operation=error_operation
        )
        def wrapped_handler(_):
            handler_func()
        
        return wrapped_handler
    
    # === OPERATION RESULT PROCESSING ===
    
    def process_operation_result(self, result: Dict[str, Any], operation: str) -> None:
        """Process operation results dengan progress tracker completion"""
        if not result.get('success', False):
            error_msg = result.get('message', f'{operation} failed')
            
            # Set error state pada progress tracker
            progress_tracker = self.get_progress_tracker()
            if progress_tracker and hasattr(progress_tracker, 'error'):
                progress_tracker.error(error_msg)
            
            raise RuntimeError(f"{operation} failed: {error_msg}")
        
        # Complete progress tracker
        progress_tracker = self.get_progress_tracker()
        if progress_tracker and hasattr(progress_tracker, 'complete'):
            success_msg = self._format_success_message(result, operation)
            progress_tracker.complete(success_msg)
        
        # Process berdasarkan operation type
        if operation == 'preprocessing':
            self._process_preprocessing_result(result)
        elif operation == 'cleanup':
            self._process_cleanup_result(result)
        elif operation == 'check':
            self._process_check_result(result)
    
    def _format_success_message(self, result: Dict[str, Any], operation: str) -> str:
        """Format success message berdasarkan operation type"""
        if operation == 'preprocessing':
            stats = result.get('stats', {})
            overview = stats.get('overview', {})
            processed_count = overview.get('total_files', 0)
            return f"Preprocessing completed: {processed_count:,} files processed"
        elif operation == 'cleanup':
            stats = result.get('stats', {})
            files_removed = stats.get('files_removed', 0)
            return f"Cleanup completed: {files_removed:,} files removed"
        elif operation == 'check':
            return "Dataset check completed"
        else:
            return f"{operation.title()} completed successfully"
    
    def _process_preprocessing_result(self, result: Dict[str, Any]) -> None:
        """Process preprocessing results"""
        stats = result.get('stats', {})
        processing_time = result.get('processing_time', 0)
        overview = stats.get('overview', {})
        processed_count = overview.get('total_files', 0)
        success_rate = overview.get('success_rate', '100%')
        
        success_msg = f"✅ Preprocessing berhasil: {processed_count:,} files dalam {processing_time:.1f}s (Success: {success_rate})"
        self.logger.success(success_msg)
    
    def _process_cleanup_result(self, result: Dict[str, Any]) -> None:
        """Process cleanup results"""
        stats = result.get('stats', {})
        files_removed = stats.get('files_removed', 0)
        splits_cleaned = stats.get('splits_cleaned', [])
        
        success_msg = f"✅ Cleanup berhasil: {files_removed:,} files dari {len(splits_cleaned)} splits"
        self.logger.success(success_msg)
    
    def _process_check_result(self, result: Dict[str, Any]) -> None:
        """Process check results"""
        if result.get('service_ready', False):
            file_stats = result.get('file_statistics', {})
            total_files = sum(split_stats.get('raw_images', 0) for split_stats in file_stats.values())
            self.logger.info(f"📊 Dataset: {total_files:,} raw images ditemukan")
        else:
            self.logger.warning("⚠️ Service belum siap atau dataset tidak ditemukan")
    
    # === UTILITY METHODS ===
    
    def extract_config(self) -> Dict[str, Any]:
        """Extract config dengan fallback"""
        config_handler = self.ui_components.get('config_handler')
        if config_handler and hasattr(config_handler, 'extract_config'):
            return config_handler.extract_config(self.ui_components)
        return self.ui_components.get('config', {})
    
    def is_confirmation_pending(self) -> bool:
        """Check pending confirmations"""
        pending_flags = ['_preprocessing_confirmed', '_cleanup_confirmed']
        return any(key in self.ui_components for key in pending_flags)
    
    def show_confirmation_dialog(self, title: str, message: str, 
                               on_confirm: Callable, on_cancel: Callable,
                               confirm_text: str = "Ya", cancel_text: str = "Batal",
                               danger_mode: bool = False) -> None:
        """Generic confirmation dialog"""
        from smartcash.ui.components.dialog.confirmation_dialog import show_confirmation_dialog
        
        show_confirmation_dialog(
            ui_components=self.ui_components,
            title=title,
            message=message,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode
        )
    
    def update_status_panel(self, message: str, status_type: str, force_update: bool = True) -> None:
        """Enhanced status panel update with force refresh
        
        Args:
            message: Message to display
            status_type: Type of status ('success', 'info', 'warning', 'error')
            force_update: Whether to force UI refresh (default: True)
        """
        update_status_panel_enhanced(self.ui_components, message, status_type, force_update)
    
    def handle_operation_cancel(self, operation: str, flag_key: str) -> None:
        """Generic operation cancellation handler"""
        self.ui_components[flag_key] = False
        self.logger.info(f"❌ {operation.title()} dibatalkan")
        
        ui_utils.enable_buttons(self.ui_components)
        self.update_status_panel(f"{operation.title()} dibatalkan", 'warning', force_update=True)
        
        if operation == 'cleanup':
            from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
            clear_dialog_area(self.ui_components)
    
    # === LOGGING SHORTCUTS ===
    
    def log_info(self, message: str) -> None:
        self.logger.info(message)
    
    def log_success(self, message: str) -> None:
        if hasattr(self.logger, 'success'):
            self.logger.success(message)
        else:
            self.logger.info(f"✅ {message}")
    
    def log_warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        self.logger.debug(message)