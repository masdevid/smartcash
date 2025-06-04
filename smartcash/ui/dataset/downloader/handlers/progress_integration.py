"""
File: smartcash/ui/dataset/downloader/handlers/progress_integration.py
Deskripsi: Integrasi progress tracker dengan backend download service menggunakan shared components
"""

from typing import Dict, Any, Callable, Optional
from smartcash.common.logger import get_logger
from smartcash.ui.utils.fallback_utils import try_operation_safe

class DownloadProgressIntegrator:
    """Integrator untuk progress tracking antara UI dan backend service."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = get_logger('downloader.progress')
        self.tracker = ui_components.get('tracker')
        self.step_mapping = {
            'validate': ('validate', 'overall', '🔍 Validating parameters'),
            'connect': ('connect', 'overall', '🌐 Connecting to Roboflow'),
            'metadata': ('connect', 'step', '📋 Getting dataset metadata'),
            'download': ('download', 'current', '📥 Downloading dataset'),
            'extract': ('extract', 'step', '📦 Extracting files'),
            'organize': ('organize', 'step', '📁 Organizing dataset'),
            'verify': ('organize', 'current', '✅ Verifying results')
        }
    
    def get_callback(self) -> Callable[[str, int, int, str], None]:
        """Get unified progress callback untuk backend services."""
        return self._unified_progress_callback
    
    def _unified_progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Unified callback yang handle semua progress updates dengan one-liner routing."""
        try:
            # Map step ke progress type dan tracker method
            step_info = self.step_mapping.get(step, (step, 'overall', message))
            progress_type, tracker_method, default_message = step_info
            
            # Calculate progress percentage
            progress_pct = min(100, max(0, int((current / max(total, 1)) * 100)))
            final_message = message or default_message
            
            # Route ke appropriate progress method dengan fallback chain
            self._route_progress_update(progress_type, progress_pct, final_message) or \
            self._fallback_progress_update(progress_pct, final_message) or \
            self._basic_log_progress(step, progress_pct, final_message)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Progress callback error: {str(e)}")
    
    def _route_progress_update(self, progress_type: str, progress: int, message: str) -> bool:
        """Route progress ke appropriate tracker method dengan one-liner."""
        return (
            # Try modern tracker methods
            try_operation_safe(lambda: self.ui_components.get('update_progress')(progress_type, progress, message)) or
            try_operation_safe(lambda: getattr(self.tracker, 'update')(progress_type, progress, message)) if self.tracker else False or
            # Try classic progress bar
            try_operation_safe(lambda: setattr(self.ui_components.get('progress_bar'), 'value', progress)) if 'progress_bar' in self.ui_components else False
        )
    
    def _fallback_progress_update(self, progress: int, message: str) -> bool:
        """Fallback progress update methods dengan one-liner."""
        return (
            # Try status panel update
            try_operation_safe(lambda: self._update_status_panel(progress, message)) or
            # Try progress message widget
            try_operation_safe(lambda: setattr(self.ui_components.get('progress_message'), 'value', f"{message} ({progress}%)")) if 'progress_message' in self.ui_components else False
        )
    
    def _basic_log_progress(self, step: str, progress: int, message: str) -> bool:
        """Basic log progress sebagai final fallback."""
        # Only log at significant milestones untuk avoid spam
        progress in [0, 25, 50, 75, 100] and self.logger.info(f"📈 {step}: {progress}% - {message}")
        return True
    
    def _update_status_panel(self, progress: int, message: str) -> None:
        """Update status panel dengan progress info."""
        status_panel = self.ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'value'):
            color = '#28a745' if progress == 100 else '#007bff'
            status_html = f"""
            <div style='padding:8px; background-color:rgba(0,123,255,0.1); 
                       color:{color}; border-radius:4px; border-left:4px solid {color};'>
                📈 {message} ({progress}%)
            </div>
            """
            status_panel.value = status_html
    
    def start_download(self, operation: str = 'download') -> None:
        """Start download dengan progress tracking initialization."""
        self._show_progress_container(operation)
        self.logger.info(f"🚀 Starting {operation} with progress tracking")
    
    def complete_download(self, message: str = "Download completed successfully!") -> None:
        """Complete download dengan success state."""
        # Complete progress dengan one-liner fallback chain
        (try_operation_safe(lambda: self.ui_components.get('complete_operation')(message)) or
         try_operation_safe(lambda: self.tracker.complete(message)) if self.tracker else None or
         self._update_status_panel(100, message))
        
        self.logger.success(f"🎉 {message}")
    
    def error_download(self, error_message: str) -> None:
        """Handle download error dengan error state."""
        # Error progress dengan one-liner fallback chain
        (try_operation_safe(lambda: self.ui_components.get('error_operation')(f"❌ {error_message}")) or
         try_operation_safe(lambda: self.tracker.error(error_message)) if self.tracker else None or
         self._update_error_status(error_message))
        
        self.logger.error(f"💥 Download error: {error_message}")
    
    def _show_progress_container(self, operation: str) -> None:
        """Show progress container untuk operation."""
        (try_operation_safe(lambda: self.ui_components.get('show_for_operation')(operation)) or
         try_operation_safe(lambda: self.tracker.show(operation)) if self.tracker else None or
         try_operation_safe(lambda: setattr(self.ui_components.get('progress_container', type('', (), {'layout': type('', (), {})()})()), 'layout.display', 'block')))
    
    def _update_error_status(self, error_message: str) -> None:
        """Update status dengan error styling."""
        status_panel = self.ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'value'):
            status_html = f"""
            <div style='padding:8px; background-color:#f8d7da; color:#721c24; 
                       border-radius:4px; border-left:4px solid #dc3545;'>
                ❌ {error_message}
            </div>
            """
            status_panel.value = status_html

def create_progress_integrator(ui_components: Dict[str, Any]) -> DownloadProgressIntegrator:
    """Factory untuk create progress integrator."""
    return DownloadProgressIntegrator(ui_components)

def setup_backend_integration(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup integrasi dengan backend download service."""
    logger = get_logger('downloader.backend_integration')
    
    try:
        # Create progress integrator
        progress_integrator = create_progress_integrator(ui_components)
        ui_components['progress_integrator'] = progress_integrator
        
        # Setup download service integration
        def create_download_service_with_progress(config: Dict[str, Any]):
            """Create download service dengan progress callback integration."""
            from smartcash.dataset.downloader.download_service import DownloadService
            
            service = DownloadService(config, logger)
            service.set_progress_callback(progress_integrator.get_callback())
            return service
        
        ui_components['create_download_service'] = create_download_service_with_progress
        
        logger.info("✅ Backend integration setup completed")
        return {'backend_integration': True, 'progress_integrator': progress_integrator}
        
    except Exception as e:
        logger.error(f"❌ Backend integration error: {str(e)}")
        return {'backend_integration': False, 'error': str(e)}