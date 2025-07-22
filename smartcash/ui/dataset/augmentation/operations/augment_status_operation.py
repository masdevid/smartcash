"""
File: smartcash/ui/dataset/augmentation/operations/augment_status_operation.py
Description: Operation for checking and reporting the status of augmentation processes.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, TYPE_CHECKING

from .augmentation_base_operation import AugmentationBaseOperation, OperationPhase

if TYPE_CHECKING:
    from smartcash.ui.dataset.augmentation.augmentation_uimodule import AugmentationUIModule

class AugmentStatusOperation(AugmentationBaseOperation):
    """
    Operation for checking and reporting the status of augmentation processes.
    
    This class handles status checking for both local and remote augmentation processes,
    providing detailed progress information and statistics.
    """
    
    def __init__(
        self,
        ui_module: 'AugmentationUIModule',
        config: Dict[str, Any],
        callbacks: Optional[Dict[str, Callable]] = None
    ) -> None:
        """
        Initialize the status operation.
        
        Args:
            ui_module: Reference to the parent UI module
            config: Configuration dictionary for status checking
            callbacks: Optional callbacks for operation events
        """
        super().__init__(ui_module, config, callbacks)
        self._last_update = None
        self._status_data = {}
    
    def _format_running_status(self, status_data: Dict[str, Any]) -> str:
        """Format running status details."""
        details = []
        
        # Add progress information
        if 'progress' in status_data:
            progress = status_data['progress']
            if isinstance(progress, (int, float)):
                details.append(f"Proses: {progress:.1f}%")
        
        # Add processed counts
        if 'processed' in status_data and 'total' in status_data:
            details.append(
                f"Diproses: {status_data['processed']}/{status_data['total']} gambar"
            )
        
        # Add current operation
        if 'current_operation' in status_data:
            details.append(f"Operasi: {status_data['current_operation']}")
        
        # Add elapsed time if available
        if 'start_time' in status_data:
            try:
                start_time = datetime.fromisoformat(status_data['start_time'])
                elapsed = datetime.now() - start_time
                details.append(f"Berjalan selama: {self._format_timedelta(elapsed)}")
            except (ValueError, TypeError):
                pass
        
        return " | ".join(details) if details else "Sedang berjalan..."
    
    def execute(self) -> Dict[str, Any]:
        """
        Execute the status check operation.
        
        Returns:
            Dictionary containing status information
        """
        self.log_operation_start("Checking Augmentation Status")
        self.log('Memeriksa status augmentasi...', 'info')
        
        try:
            # Get status from backend
            status_api = self.get_backend_api('status')
            if not status_api:
                return self._handle_error("Backend status API not available")
            
            # Get status data
            self._last_update = datetime.now()
            status_data = status_api()
            
            if not status_data:
                return self._handle_error("No status response from backend")
            
            # Store status data
            self._status_data = status_data
            
            # Handle actual backend status format
            if 'service_ready' in status_data and status_data['service_ready']:
                self.log('✅ Service augmentasi siap', 'info')
                # Log dataset status info
                if 'paths' in status_data:
                    self.log(f"📁 Data paths configured: {len(status_data['paths'])} paths", 'info')
            elif 'error' in status_data:
                self.log(f"❌ Error: {status_data['error']}", 'error')
            
            # Log completion
            self.log_info("Pemeriksaan status selesai")
            
            # Return status data
            return {
                'status': 'success',
                'success': True,
                'message': 'Status check completed successfully',
                'data': status_data,
                'timestamp': self._last_update.isoformat(),
                'formatted': self._format_status(status_data)
            }
            
        except Exception as e:
            error_msg = f"Gagal memeriksa status: {str(e)}"
            self.log_error(error_msg)
            return self._handle_error(error_msg)
            
        finally:
            self.log_operation_complete("Status Check")
    
    def get_last_status(self) -> Dict[str, Any]:
        """
        Get the last retrieved status data.
        
        Returns:
            Dictionary containing the last status data
        """
        return {
            'data': self._status_data,
            'timestamp': self._last_update.isoformat() if self._last_update else None,
            'formatted': self._format_status(self._status_data)
        }
    
    def _format_status(self, status_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Format status data for display.
        
        Args:
            status_data: Raw status data from backend
            
        Returns:
            Formatted status information
        """
        if not status_data:
            return {
                'status': 'unknown',
                'message': 'Status tidak tersedia',
                'details': ''
            }
        
        status = status_data.get('status', 'unknown')
        
        if status == 'running':
            return {
                'status': 'running',
                'message': 'Augmentasi sedang berjalan',
                'details': self._format_running_status(status_data)
            }
        elif status == 'completed':
            return {
                'status': 'completed',
                'message': 'Augmentasi selesai',
                'details': self._format_completed_status(status_data)
            }
        elif status == 'failed':
            return {
                'status': 'error',
                'message': 'Augmentasi gagal',
                'details': status_data.get('message', 'Terjadi kesalahan')
            }
        else:
            return {
                'status': 'unknown',
                'message': 'Status tidak dikenali',
                'details': str(status_data)
            }
    
    def _format_running_status(self, status_data: Dict[str, Any]) -> str:
        """Format running status details."""
        details = []
        
        # Add progress information
        if 'progress' in status_data:
            progress = status_data['progress']
            if isinstance(progress, (int, float)):
                details.append(f"Proses: {progress:.1f}%")
        
        # Add processed counts
        if 'processed' in status_data and 'total' in status_data:
            details.append(
                f"Diproses: {status_data['processed']}/{status_data['total']} gambar"
            )
        
        # Add current operation
        if 'current_operation' in status_data:
            details.append(f"Operasi: {status_data['current_operation']}")
        
        # Add elapsed time if available
        if 'start_time' in status_data:
            try:
                start_time = datetime.fromisoformat(status_data['start_time'])
                elapsed = datetime.now() - start_time
                details.append(f"Berjalan selama: {self._format_timedelta(elapsed)}")
            except (ValueError, TypeError):
                pass
        
        return " | ".join(details) if details else "Sedang berjalan..."
    
    def _format_completed_status(self, status_data: Dict[str, Any]) -> str:
        """Format completed status details."""
        details = []
        
        # Add statistics
        if 'processed' in status_data:
            details.append(f"Total diproses: {status_data['processed']}")
        
        if 'generated' in status_data:
            details.append(f"Total dihasilkan: {status_data['generated']}")
        
        # Add timing information
        if 'start_time' in status_data and 'end_time' in status_data:
            try:
                start = datetime.fromisoformat(status_data['start_time'])
                end = datetime.fromisoformat(status_data['end_time'])
                duration = end - start
                details.append(f"Durasi: {self._format_timedelta(duration)}")
            except (ValueError, TypeError):
                pass
        
        return " | ".join(details) if details else "Selesai"
    
    @staticmethod
    def _format_timedelta(delta: timedelta) -> str:
        """Format a timedelta as a human-readable string."""
        total_seconds = int(delta.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}j {minutes}m {seconds}d"
        elif minutes > 0:
            return f"{minutes}m {seconds}d"
        else:
            return f"{seconds}d"


# Factory function has been moved to augment_factory.py
