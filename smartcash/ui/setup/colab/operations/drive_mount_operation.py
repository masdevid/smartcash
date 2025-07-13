"""
File: smartcash/ui/setup/colab/operations/drive_mount_operation.py
Description: Mount Google Drive with verification
"""

import os
import sys
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from ..utils.env_detector import get_runtime_type, _is_google_colab


class DriveMountOperation(OperationHandler):
    """Mount Google Drive with verification."""
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize drive mount operation.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
        """
        super().__init__(
            module_name='drive_mount_operation',
            parent_module='colab',
            **kwargs
        )
        self.config = config
    
    def initialize(self) -> None:
        """Initialize the drive mount operation."""
        self.logger.info("🚀 Initializing drive mount operation")
        # No specific initialization needed for drive mount operation
        self.logger.info("✅ Drive mount operation initialization complete")
    
    def get_operations(self) -> Dict[str, Callable]:
        """Get available operations."""
        return {
            'mount_drive': self.execute_mount_drive
        }
    
    def execute_mount_drive(self, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Mount Google Drive with verification.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Get detailed environment information
            from ..utils.env_detector import detect_environment_info, _is_google_colab
            
            # Check if running in Colab first
            is_colab = _is_google_colab()
            runtime_info = get_runtime_type()
            
            # If we're in Colab but config doesn't reflect that, update it
            if is_colab and self.config.get('environment', {}).get('type') != 'colab':
                if 'environment' not in self.config:
                    self.config['environment'] = {}
                self.config['environment']['type'] = 'colab'
                self.logger.info("Updated environment type to 'colab' based on runtime detection")
            
            # If we're not in Colab, return error with debug info
            if not is_colab:
                error_msg = [
                    "Google Drive mounting is only available in Colab environment. ",
                    f"Runtime type: {runtime_info.get('type')}",
                    f"GPU status: {runtime_info.get('gpu')}",
                    "\nEnvironment details:",
                    f"- Python executable: {sys.executable}",
                    f"- Current working directory: {os.getcwd()}",
                    f"- /content exists: {os.path.exists('/content')}",
                    f"- /content/drive exists: {os.path.exists('/content/drive')}",
                    "\nEnvironment variables:",
                    *[f"- {k}: {v}" for k, v in os.environ.items() 
                      if 'COLAB' in k or 'JUPYTER' in k or 'IPYTHON' in k]
                ]
                
                # Add debug info if available
                if 'debug' in runtime_info:
                    error_msg.extend([
                        "\nDebug information:",
                        f"Python path: {runtime_info['debug'].get('python_executable')}",
                        "Files checked:",
                        *[f"- {k}: {v}" for k, v in runtime_info['debug'].get('files_checked', {}).items()]
                    ])
                
                return {
                    'success': False,
                    'error': '\n'.join(error_msg),
                    'environment_info': {
                        'detected_type': runtime_info.get('type'),
                        'is_colab': is_colab,
                        'has_gpu': runtime_info.get('gpu') == 'available',
                        'debug': runtime_info.get('debug', {})
                    }
                }
            
            if progress_callback:
                progress_callback(10, "🔍 Checking Drive mount status...")
            
            mount_path = '/content/drive'
            
            # Check if already mounted
            if os.path.exists(mount_path) and os.path.exists(os.path.join(mount_path, 'MyDrive')):
                self.log("Google Drive already mounted", 'info')
                if progress_callback:
                    progress_callback(100, "✅ Google Drive already mounted")
                return {
                    'success': True,
                    'already_mounted': True,
                    'mount_path': mount_path,
                    'message': 'Google Drive was already mounted'
                }
            
            if progress_callback:
                progress_callback(30, "📁 Mounting Google Drive...")
            
            # Import and mount drive
            try:
                from google.colab import drive
                drive.mount(mount_path)
                self.log("Google Drive mount initiated", 'info')
                
                if progress_callback:
                    progress_callback(70, "🔍 Verifying mount...")
                
                # Verify mount
                if os.path.exists(os.path.join(mount_path, 'MyDrive')):
                    # Check write access
                    write_access = self._test_write_access(mount_path)
                    
                    if progress_callback:
                        progress_callback(100, "✅ Google Drive mounted successfully")
                    
                    return {
                        'success': True,
                        'mount_path': mount_path,
                        'write_access': write_access,
                        'message': 'Google Drive mounted successfully'
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Drive mount verification failed'
                    }
                    
            except ImportError:
                return {
                    'success': False,
                    'error': 'Google Colab drive module not available'
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Drive mount failed: {str(e)}'
                }
                
        except Exception as e:
            self.log(f"Drive mount operation failed: {str(e)}", 'error')
            return {
                'success': False,
                'error': f'Drive mount operation failed: {str(e)}'
            }
    
    def _test_write_access(self, mount_path: str) -> bool:
        """Test write access to mounted drive.
        
        Args:
            mount_path: Path to mounted drive
            
        Returns:
            True if write access is available, False otherwise
        """
        try:
            test_file = os.path.join(mount_path, 'MyDrive', '.smartcash_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            self.log("Drive write access verified", 'info')
            return True
        except Exception:
            self.log("Drive write access limited", 'warning')
            return False