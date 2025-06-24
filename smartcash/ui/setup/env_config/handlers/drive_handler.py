"""
File: smartcash/ui/setup/env_config/handlers/drive_handler.py
Deskripsi: Handler untuk mounting dan managing Google Drive
"""

import os
from typing import Dict, Any

class DriveHandler:
    """💾 Handler untuk Google Drive operations"""
    
    def __init__(self, logger=None):
        self.logger = logger or self._create_dummy_logger()
        self.mount_path = '/content/drive'
    
    def mount_drive(self) -> Dict[str, Any]:
        """🔗 Mount Google Drive dengan auto-detection"""
        try:
            # Check if already mounted
            if self._is_drive_mounted():
                self.logger.info("✅ Google Drive sudah mounted")
                return {
                    'success': True,
                    'mount_path': self.mount_path,
                    'already_mounted': True
                }
            
            # Attempt to mount
            self.logger.info("🔗 Mounting Google Drive...")
            
            # Import and mount
            from google.colab import drive
            drive.mount(self.mount_path, force_remount=False)
            
            # Verify mount
            if self._is_drive_mounted():
                self.logger.success(f"✅ Drive mounted successfully at {self.mount_path}")
                return {
                    'success': True,
                    'mount_path': self.mount_path,
                    'already_mounted': False
                }
            else:
                raise Exception("Mount verification failed")
                
        except ImportError:
            self.logger.warning("⚠️ Not running in Google Colab, skipping drive mount")
            return {
                'success': False,
                'mount_path': 'N/A',
                'reason': 'Not in Colab environment'
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to mount drive: {str(e)}")
            return {
                'success': False,
                'mount_path': 'N/A',
                'error': str(e)
            }
    
    def _is_drive_mounted(self) -> bool:
        """🔍 Check apakah drive sudah mounted"""
        try:
            mydrive_path = os.path.join(self.mount_path, 'MyDrive')
            return os.path.exists(mydrive_path) and os.path.isdir(mydrive_path)
        except:
            return False
    
    def get_drive_path(self, relative_path: str = "") -> str:
        """📁 Get full drive path"""
        if not relative_path:
            return os.path.join(self.mount_path, 'MyDrive')
        return os.path.join(self.mount_path, 'MyDrive', relative_path.lstrip('/'))
    
    def _create_dummy_logger(self):
        """📝 Create dummy logger fallback"""
        class DummyLogger:
            def info(self, msg): print(f"ℹ️ {msg}")
            def warning(self, msg): print(f"⚠️ {msg}")
            def error(self, msg): print(f"❌ {msg}")
            def success(self, msg): print(f"✅ {msg}")
        return DummyLogger()