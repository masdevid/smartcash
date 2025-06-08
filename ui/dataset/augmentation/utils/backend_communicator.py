"""
File: smartcash/ui/dataset/augmentation/utils/backend_communicator.py
Deskripsi: Enhanced backend communicator dengan granular progress tracking dan service integration
"""

from typing import Dict, Any, Optional, Callable
import time
from smartcash.dataset.augmentor.types import ServiceMessage, UIProgressUpdate

class BackendProgressCommunicator:
    """Enhanced communicator untuk backend-UI integration dengan granular progress"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.current_operation = None
        self.operation_start_time = None
        self.last_update_time = 0
        self.update_throttle = 0.1  # Minimum 100ms between updates
        
        # Setup UI logger reference
        self.ui_logger = ui_components.get('logger')
        self.progress_tracker = ui_components.get('progress_tracker')
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """Start operation dengan UI notification dan progress initialization"""
        self.current_operation = operation_name
        self.operation_start_time = time.time()
        
        # Show progress tracker jika ada
        if self.progress_tracker and hasattr(self.progress_tracker, 'show'):
            self.progress_tracker.show()
        
        # Log operation start
        self.log_info(f"🚀 Memulai {operation_name}")
        
        # Initial progress update
        self.progress('overall', 0, 100, f"Memulai {operation_name}")
    
    def progress(self, level: str, current: int, total: int, message: str = ""):
        """Update progress dengan throttling dan UI integration"""
        current_time = time.time()
        
        # Throttle updates untuk performance
        if current_time - self.last_update_time < self.update_throttle and current < total:
            return
        
        self.last_update_time = current_time
        
        # Calculate percentage
        percentage = min(100, max(0, int((current / max(total, 1)) * 100)))
        
        # Create service message
        service_msg = ServiceMessage(
            operation=self.current_operation or "operation",
            level=level,
            progress=percentage,
            total=100,
            message=message,
            timestamp=current_time
        )
        
        # Update UI progress tracker
        self._update_progress_tracker(service_msg)
        
        # Log milestone updates
        if percentage in [0, 25, 50, 75, 100] or level == 'overall':
            level_emoji = self._get_level_emoji(level)
            self.log_info(f"{level_emoji} {level.title()}: {percentage}% - {message}")
    
    def complete_operation(self, operation_name: str, result_message: str = ""):
        """Complete operation dengan duration tracking dan UI updates"""
        if not self.operation_start_time:
            return
        
        duration = time.time() - self.operation_start_time
        final_message = result_message or f"{operation_name} selesai"
        
        # Complete progress tracker
        if self.progress_tracker and hasattr(self.progress_tracker, 'complete'):
            self.progress_tracker.complete(f"{final_message} (⏱️ {duration:.1f}s)")
        
        # Log completion
        self.log_success(f"✅ {final_message} dalam {duration:.1f}s")
        
        # Reset operation state
        self.current_operation = None
        self.operation_start_time = None
    
    def error_operation(self, operation_name: str, error_message: str):
        """Handle operation error dengan UI feedback"""
        duration = time.time() - self.operation_start_time if self.operation_start_time else 0
        
        # Error progress tracker
        if self.progress_tracker and hasattr(self.progress_tracker, 'error'):
            self.progress_tracker.error(f"❌ {error_message}")
        
        # Log error dengan duration
        error_with_duration = f"❌ {operation_name} gagal: {error_message}"
        if duration > 0:
            error_with_duration += f" (setelah {duration:.1f}s)"
        
        self.log_error(error_with_duration)
        
        # Reset operation state
        self.current_operation = None
        self.operation_start_time = None
    
    def log_info(self, message: str):
        """Log info message dengan UI integration"""
        self._log_to_ui(message, 'info')
    
    def log_success(self, message: str):
        """Log success message dengan UI integration"""
        self._log_to_ui(message, 'success')
    
    def log_warning(self, message: str):
        """Log warning message dengan UI integration"""
        self._log_to_ui(message, 'warning')
    
    def log_error(self, message: str):
        """Log error message dengan UI integration"""
        self._log_to_ui(message, 'error')
    
    def _update_progress_tracker(self, service_msg: ServiceMessage):
        """Update progress tracker berdasarkan service message"""
        if not self.progress_tracker:
            return
        
        try:
            # Map level ke method yang sesuai
            if service_msg.level == 'overall' and hasattr(self.progress_tracker, 'update_overall'):
                self.progress_tracker.update_overall(service_msg.progress, service_msg.message)
            elif service_msg.level == 'step' and hasattr(self.progress_tracker, 'update_step'):
                self.progress_tracker.update_step(service_msg.progress, service_msg.message)
            elif service_msg.level == 'current' and hasattr(self.progress_tracker, 'update_current'):
                self.progress_tracker.update_current(service_msg.progress, service_msg.message)
            elif hasattr(self.progress_tracker, 'update'):
                # Fallback ke generic update
                self.progress_tracker.update(service_msg.progress, service_msg.message)
        except Exception:
            # Silent fail untuk compatibility
            pass
    
    def _log_to_ui(self, message: str, level: str):
        """Log ke UI dengan fallback chain"""
        try:
            # Priority 1: UI Logger
            if self.ui_logger and hasattr(self.ui_logger, level):
                getattr(self.ui_logger, level)(message)
                return
            
            # Priority 2: Log widget
            log_widget = self.ui_components.get('log_output') or self.ui_components.get('status')
            if log_widget and hasattr(log_widget, 'clear_output'):
                from IPython.display import display, HTML
                
                color_map = {
                    'info': '#007bff', 'success': '#28a745', 
                    'warning': '#ffc107', 'error': '#dc3545'
                }
                color = color_map.get(level, '#007bff')
                
                html = f'<div style="color: {color}; margin: 2px 0; padding: 4px; font-family: monospace;">{message}</div>'
                
                with log_widget:
                    display(HTML(html))
                return
                
        except Exception:
            pass
        
        # Fallback print
        emoji_map = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
        print(f"{emoji_map.get(level, 'ℹ️')} {message}")
    
    def _get_level_emoji(self, level: str) -> str:
        """Get emoji untuk progress level"""
        level_emojis = {
            'overall': '🎯',
            'step': '📊',
            'current': '⚡',
            'operation': '🔄'
        }
        return level_emojis.get(level, '📈')
    
    def get_operation_status(self) -> Dict[str, Any]:
        """Get current operation status"""
        return {
            'operation': self.current_operation,
            'duration': time.time() - self.operation_start_time if self.operation_start_time else 0,
            'is_running': self.current_operation is not None
        }

class ServiceIntegrator:
    """Service integrator untuk orchestrasi backend-UI communication"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.communicator = BackendProgressCommunicator(ui_components)
        self.active_operations = {}
    
    def create_service_config(self) -> Dict[str, Any]:
        """Create service config dari UI components"""
        from smartcash.ui.dataset.augmentation.handlers.config_extractor import extract_augmentation_config
        
        # Extract config dari UI
        ui_config = extract_augmentation_config(self.ui_components)
        
        # Add communicator reference
        service_config = {
            'ui_components': self.ui_components,
            'communicator': self.communicator,
            'config': ui_config
        }
        
        return service_config
    
    def create_augmentation_service(self):
        """Create augmentation service dengan UI integration"""
        try:
            from smartcash.dataset.augmentor.service import create_service_from_ui
            return create_service_from_ui(self.ui_components)
        except ImportError:
            self.communicator.log_error("❌ Backend service tidak tersedia")
            return None
    
    def execute_with_progress(self, operation_name: str, service_method: Callable, *args, **kwargs):
        """Execute service method dengan progress tracking"""
        if operation_name in self.active_operations:
            self.communicator.log_warning(f"⚠️ Operation {operation_name} sudah berjalan")
            return None
        
        self.active_operations[operation_name] = True
        
        try:
            # Start operation tracking
            self.communicator.start_operation(operation_name)
            
            # Create progress callback
            def progress_callback(step: str, current: int, total: int, message: str):
                self.communicator.progress(step, current, total, message)
            
            # Execute dengan progress callback
            if 'progress_callback' in kwargs:
                kwargs['progress_callback'] = progress_callback
            else:
                # Try to add progress callback jika method support
                import inspect
                sig = inspect.signature(service_method)
                if 'progress_callback' in sig.parameters:
                    kwargs['progress_callback'] = progress_callback
            
            result = service_method(*args, **kwargs)
            
            # Handle result
            if result and result.get('status') == 'success':
                self.communicator.complete_operation(operation_name, 
                    result.get('message', f"{operation_name} berhasil"))
            else:
                error_msg = result.get('message', 'Unknown error') if result else 'Operation failed'
                self.communicator.error_operation(operation_name, error_msg)
            
            return result
            
        except Exception as e:
            self.communicator.error_operation(operation_name, str(e))
            return {'status': 'error', 'message': str(e)}
            
        finally:
            self.active_operations.pop(operation_name, None)
    
    def get_communicator(self) -> BackendProgressCommunicator:
        """Get communicator instance"""
        return self.communicator

# Factory functions
def create_backend_communicator(ui_components: Dict[str, Any]) -> BackendProgressCommunicator:
    """Factory untuk backend communicator"""
    return BackendProgressCommunicator(ui_components)

def create_service_integrator(ui_components: Dict[str, Any]) -> ServiceIntegrator:
    """Factory untuk service integrator"""
    return ServiceIntegrator(ui_components)

# One-liner utilities
create_communicator = lambda ui_components: create_backend_communicator(ui_components)
create_integrator = lambda ui_components: create_service_integrator(ui_components)
execute_with_progress = lambda integrator, operation, method, *args, **kwargs: integrator.execute_with_progress(operation, method, *args, **kwargs)