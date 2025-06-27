"""
File: smartcash/ui/dataset/preprocessing/handlers/base_handler.py
Deskripsi: Fixed base handler dengan proper button access dan suppressed init logs
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List
from smartcash.ui.utils import with_error_handling, ErrorHandler
from smartcash.ui.dataset.preprocessing import utils as ui_utils

class BasePreprocessingHandler(ABC):
    """Base class untuk handlers dengan proper button mapping dan minimal logging"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger_bridge')
        self.error_handler = ErrorHandler()
        
        if not self.logger:
            raise ValueError("Logger bridge required in UI components")
    
    # === ABSTRACT METHODS ===
    
    @abstractmethod
    def setup_handlers(self) -> Dict[str, Any]:
        """Setup handlers specific untuk handler type ini"""
        pass
    
    # === FIXED BUTTON ACCESS ===
    
    def get_button(self, button_id: str) -> Optional[Any]:
        """Get button dengan proper fallback mapping"""
        # Primary: direct access
        if button := self.ui_components.get(button_id):
            return button
        
        # Secondary: action_buttons mapping
        action_buttons = self.ui_components.get('action_buttons', {})
        if button := action_buttons.get(button_id):
            return button
        
        # Tertiary: original create_action_buttons result
        action_components = self.ui_components.get('action_components', {})
        if button := action_components.get(button_id):
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
    
    def clear_button_handlers(self, button: Any) -> None:
        """Clear button handlers dengan safe approach"""
        if not button or not hasattr(button, 'on_click'):
            return
        try:
            if hasattr(button, '_click_handlers'):
                button._click_handlers.callbacks.clear()
        except Exception:
            pass
    
    def setup_button_handler(self, button_id: str, handler_func: Callable, 
                           error_operation: str) -> Optional[Callable]:
        """Setup button handler dengan proper mapping dan minimal logging"""
        button = self.get_button(button_id)
        if not button:
            # SUPPRESSED: hanya log jika debug mode
            if getattr(self.logger, 'level', 20) <= 10:  # DEBUG level
                self.logger.debug(f"🔍 Button {button_id} tidak ditemukan")
            return None
        
        self.clear_button_handlers(button)
        
        @button.on_click
        @with_error_handling(
            error_handler=self.error_handler,
            component=self.__class__.__name__,
            operation=error_operation
        )
        def wrapped_handler(_):
            handler_func()
        
        return wrapped_handler
    
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
    
    def create_progress_callback(self) -> Callable[[str, int, int, str], None]:
        """Create standardized progress callback"""
        def progress_callback(level: str, current: int, total: int, message: str = "") -> None:
            try:
                progress_tracker = self.ui_components.get('progress_tracker')
                if not progress_tracker or total <= 0:
                    return
                
                progress_percent = max(0, min(100, int((current / total) * 100)))
                
                method_map = {
                    'overall': 'update_overall',
                    'current': 'update_current',
                    'step': 'update_step', 
                    'batch': 'update_batch'
                }
                
                method_name = method_map.get(level)
                if method_name and hasattr(progress_tracker, method_name):
                    method = getattr(progress_tracker, method_name)
                    method(progress_percent, message)
                
                # MINIMAL LOGGING: hanya overall dan current
                if level in ['overall', 'current'] and current % 10 == 0:  # Every 10%
                    log_message = f"{message} ({progress_percent}%)" if message else f"Progress: {progress_percent}%"
                    self.logger.info(log_message)
                    
            except Exception as e:
                # SUPPRESSED: minimal error logging
                if getattr(self.logger, 'level', 20) <= 10:
                    self.logger.debug(f"🔍 Progress callback error: {str(e)}")
        
        return progress_callback
    
    def update_status_panel(self, message: str, status_type: str) -> None:
        """Update status panel"""
        status_panel = self.ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'update_status'):
            status_panel.update_status(message, status_type)
    
    def handle_operation_cancel(self, operation: str, flag_key: str) -> None:
        """Generic operation cancellation handler"""
        self.ui_components[flag_key] = False
        self.logger.info(f"❌ {operation.title()} dibatalkan")
        
        ui_utils.enable_buttons(self.ui_components)
        self.update_status_panel(f"{operation.title()} dibatalkan", 'warning')
        
        if operation == 'cleanup':
            from smartcash.ui.components.dialog.confirmation_dialog import clear_dialog_area
            clear_dialog_area(self.ui_components)
    
    def process_operation_result(self, result: Dict[str, Any], operation: str) -> None:
        """Process operation results"""
        if not result.get('success', False):
            error_msg = result.get('message', f'{operation} failed')
            raise RuntimeError(f"{operation} failed: {error_msg}")
        
        # Process berdasarkan operation type
        if operation == 'preprocessing':
            self._process_preprocessing_result(result)
        elif operation == 'cleanup':
            self._process_cleanup_result(result)
        elif operation == 'check':
            self._process_check_result(result)
    
    def _process_preprocessing_result(self, result: Dict[str, Any]) -> None:
        """Process preprocessing results"""
        stats = result.get('stats', {})
        processing_time = result.get('processing_time', 0)
        overview = stats.get('overview', {})
        processed_count = overview.get('total_files', 0)
        success_rate = overview.get('success_rate', '100%')
        
        success_msg = f"✅ Preprocessing berhasil: {processed_count:,} files dalam {processing_time:.1f}s (Success: {success_rate})"
        ui_utils.complete_progress(self.ui_components, success_msg)
        self.logger.success(success_msg)
    
    def _process_cleanup_result(self, result: Dict[str, Any]) -> None:
        """Process cleanup results"""
        stats = result.get('stats', {})
        files_removed = stats.get('files_removed', 0)
        splits_cleaned = stats.get('splits_cleaned', [])
        
        success_msg = f"✅ Cleanup berhasil: {files_removed:,} files dari {len(splits_cleaned)} splits"
        ui_utils.complete_progress(self.ui_components, success_msg)
        self.logger.success(success_msg)
    
    def _process_check_result(self, result: Dict[str, Any]) -> None:
        """Process check results"""
        if result.get('service_ready', False):
            file_stats = result.get('file_statistics', {})
            total_files = sum(split_stats.get('raw_images', 0) for split_stats in file_stats.values())
            self.logger.info(f"📊 Dataset: {total_files:,} raw images ditemukan")
        else:
            self.logger.warning("⚠️ Service belum siap atau dataset tidak ditemukan")
        
        ui_utils.complete_progress(self.ui_components, "Status check selesai")
    
    # === LOGGING SHORTCUTS ===
    
    def log_info(self, message: str) -> None:
        self.logger.info(message)
    
    def log_success(self, message: str) -> None:
        self.logger.success(message)
    
    def log_warning(self, message: str) -> None:
        self.logger.warning(message)
    
    def log_error(self, message: str) -> None:
        self.logger.error(message)
    
    def log_debug(self, message: str) -> None:
        """SUPPRESSED: hanya log jika debug mode aktif"""
        if getattr(self.logger, 'level', 20) <= 10:
            self.logger.debug(message)