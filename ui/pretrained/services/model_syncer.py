# File: smartcash/ui/pretrained/services/model_syncer.py
"""
File: smartcash/ui/pretrained/services/model_syncer.py
Deskripsi: Complete service untuk syncing models dengan progress tracking
"""

import os
import shutil
from typing import List, Optional, Callable, Dict, Any, TypeVar, Type, Union
from functools import wraps

from smartcash.ui.utils.error_utils import (
    with_error_handling,
    log_errors,
    create_error_context,
    ErrorContext
)

# Type alias for logger bridge type
LoggerBridge = Callable[[str, str], None]

# Type variable for class methods
T = TypeVar('T', bound='PretrainedModelSyncer')

class PretrainedModelSyncer:
    """🔄 Service untuk syncing pretrained models dengan progress tracking"""
    
    def __init__(self, logger_bridge: Optional[LoggerBridge] = None):
        """Initialize the model syncer with an optional logger bridge
        
        Args:
            logger_bridge: Optional logger bridge for UI logging
        """
        self._logger_bridge = logger_bridge
        self._model_extensions = ['.pt', '.bin', '.pth', '.ckpt']
    
    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message using the logger bridge if available
        
        Args:
            message: The message to log
            level: Log level (debug, info, warning, error)
            **kwargs: Additional logging parameters
        """
        if not self._logger_bridge:
            return
            
        log_func = getattr(self._logger_bridge, level.lower(), None)
        if callable(log_func):
            log_func(message, **kwargs)
    
    @with_error_handling(
        component="pretrained",
        operation="sync_to_drive",
        fallback_value=False
    )
    @log_errors(level="error")
    def sync_to_drive(self, source_dir: str, drive_dir: str,
                     progress_callback: Optional[Callable[[int, str], None]] = None,
                     status_callback: Optional[Callable[[str], None]] = None,
                     **kwargs) -> bool:
        """📤 Sync models dari local ke drive dengan progress tracking
        
        Args:
            source_dir: Source directory (local models)
            drive_dir: Drive directory target
            progress_callback: Callback untuk update progress (progress_pct, message)
            status_callback: Callback untuk update status message
            **kwargs: Additional arguments for error context
            
        Returns:
            True jika sync berhasil, False jika gagal
        """
        context = create_error_context(
            component="pretrained",
            operation="sync_to_drive",
            details={
                "source_dir": source_dir,
                "drive_dir": drive_dir
            }
        )
        
        status_msg = "📁 Preparing drive directory..."
        if status_callback:
            status_callback(status_msg)
        self._log(status_msg, "info")
        
        # Create drive directory jika belum ada
        os.makedirs(drive_dir, exist_ok=True)
        
        # Get model files
        model_files = self._get_model_files(source_dir)
            
        if not model_files:
            status_msg = "⚠️ No model files found to sync"
            if status_callback:
                status_callback(status_msg)
            self._log(status_msg, "warning")
            
            if progress_callback:
                progress_callback(100, "No files to sync")
            return True
            
        total_files = len(model_files)
        
        # Sync each file
        for idx, model_file in enumerate(model_files, 1):
            src_path = os.path.join(source_dir, model_file)
            dst_path = os.path.join(drive_dir, model_file)
            
            # Update status
            status_msg = f"🔄 Syncing {model_file} to drive..."
            if status_callback:
                status_callback(status_msg)
            self._log(status_msg, "info")
            
            # Calculate progress
            progress_pct = int((idx / total_files) * 100)
            if progress_callback:
                progress_callback(progress_pct, f"Syncing {model_file}")
            
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(src_path, dst_path)
            
            # Log success
            self._log(f"✅ Synced {model_file} to drive", "info")
        
        completion_msg = "✅ Sync to drive completed"
        if status_callback:
            status_callback(completion_msg)
        self._log(completion_msg, "info")
            
        if progress_callback:
            progress_callback(100, "Sync completed")
            
        return True
    
    @with_error_handling(
        component="pretrained",
        operation="sync_from_drive",
        fallback_value=False
    )
    @log_errors(level="error")
    def sync_from_drive(self, drive_dir: str, target_dir: str,
                       progress_callback: Optional[Callable[[int, str], None]] = None,
                       status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """📥 Sync models dari drive ke local dengan progress tracking
        
        Args:
            drive_dir: Drive directory source
            target_dir: Target directory (local models)
            progress_callback: Callback untuk update progress (progress_pct, message)
            status_callback: Callback untuk update status message
            
        Returns:
            True jika sync berhasil, False jika gagal
        """
        try:
            if status_callback:
                status_callback("📁 Checking drive directory...")
            
            if not os.path.exists(drive_dir):
                error_msg = f"❌ Drive directory not found: {drive_dir}"
                if status_callback:
                    status_callback(error_msg)
                return False
            
            # Create target directory jika belum ada
            os.makedirs(target_dir, exist_ok=True)
            
            # Get model files dari drive
            model_files = self._get_model_files(drive_dir)
            
            if not model_files:
                if status_callback:
                    status_callback("⚠️ No model files found in drive")
                if progress_callback:
                    progress_callback(100, "No files to sync")
                return True
            
            if status_callback:
                status_callback(f"📥 Syncing {len(model_files)} files from drive...")
            
            # Copy files dengan progress tracking
            for i, model_file in enumerate(model_files):
                drive_path = os.path.join(drive_dir, model_file)
                target_path = os.path.join(target_dir, model_file)
                
                # Update progress
                progress = int((i / len(model_files)) * 90) + 5  # 5-95%
                if progress_callback:
                    progress_callback(progress, f"Syncing {model_file}...")
                
                # Check jika file sudah ada di local dengan ukuran sama
                if self._files_are_identical(drive_path, target_path):
                    self._log(f"⏭️ Skipped {model_file} (already synced)", "info")
                    continue
                
                # Copy from drive
                shutil.copy2(drive_path, target_path)
                self._log(f"📥 Synced {model_file} from drive", "info")
            
            if status_callback:
                status_callback(f"✅ Successfully synced {len(model_files)} files from drive")
            if progress_callback:
                progress_callback(100, "Sync from drive completed")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Drive sync failed: {str(e)}"
            self._log(error_msg, "error")
            if status_callback:
                status_callback(error_msg)
            return False
    
    @with_error_handling(
        component="pretrained",
        operation="bi_directional_sync",
        fallback_value=False
    )
    @log_errors(level="error")
    def bi_directional_sync(self, local_dir: str, drive_dir: str,
                          progress_callback: Optional[Callable[[int, str], None]] = None,
                          status_callback: Optional[Callable[[str], None]] = None,
                          **kwargs) -> bool:
        """🔄 Bi-directional sync antara local dan drive
        
        Args:
            local_dir: Local directory
            drive_dir: Drive directory
            progress_callback: Callback untuk update progress
            status_callback: Callback untuk update status
            
        Returns:
            True jika sync berhasil
        """
        try:
            if status_callback:
                status_callback("🔍 Analyzing sync requirements...")
            
            local_files = set(self._get_model_files(local_dir))
            drive_files = set(self._get_model_files(drive_dir)) if os.path.exists(drive_dir) else set()
            
            # Files to sync to drive (local only)
            to_drive = local_files - drive_files
            # Files to sync from drive (drive only)
            from_drive = drive_files - local_files
            
            total_operations = len(to_drive) + len(from_drive)
            
            if total_operations == 0:
                if status_callback:
                    status_callback("✅ Models already in sync")
                if progress_callback:
                    progress_callback(100, "Already synchronized")
                return True
            
            current_op = 0
            
            # Sync to drive
            if to_drive:
                if status_callback:
                    status_callback(f"📤 Syncing {len(to_drive)} files to drive...")
                
                os.makedirs(drive_dir, exist_ok=True)
                for file in to_drive:
                    source_path = os.path.join(local_dir, file)
                    drive_path = os.path.join(drive_dir, file)
                    shutil.copy2(source_path, drive_path)
                    
                    current_op += 1
                    progress = int((current_op / total_operations) * 90) + 5
                    if progress_callback:
                        progress_callback(progress, f"Synced {file} to drive")
            
            # Sync from drive
            if from_drive:
                if status_callback:
                    status_callback(f"📥 Syncing {len(from_drive)} files from drive...")
                
                os.makedirs(local_dir, exist_ok=True)
                for file in from_drive:
                    drive_path = os.path.join(drive_dir, file)
                    local_path = os.path.join(local_dir, file)
                    shutil.copy2(drive_path, local_path)
                    
                    current_op += 1
                    progress = int((current_op / total_operations) * 90) + 5
                    if progress_callback:
                        progress_callback(progress, f"Synced {file} from drive")
            
            if status_callback:
                status_callback("✅ Bi-directional sync completed")
            if progress_callback:
                progress_callback(100, "Synchronization completed")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Bi-directional sync failed: {str(e)}"
            self._log(error_msg, "error")
            if status_callback:
                status_callback(error_msg)
            return False
    
    @with_error_handling(
        component="pretrained",
        operation="_get_model_files",
        fallback_factory=lambda: []
    )
    @log_errors(level="error")
    def _get_model_files(self, directory: str) -> List[str]:
        """Get list of model files in directory
        
        Args:
            directory: Directory to scan for model files
            
        Returns:
            List of model file paths relative to directory
            
        Note:
            This method is wrapped with error handling decorators that will return
            an empty list in case of any errors during execution.
        """
        if not directory:
            self._log("No directory specified for model files search", "warning")
            return []
            
        if not os.path.exists(directory):
            self._log(f"Directory not found: {directory}", "warning")
            return []
            
        if not os.path.isdir(directory):
            self._log(f"Path is not a directory: {directory}", "warning")
            return []
            
        try:
            # Get all files with model extensions in the directory and subdirectories
            model_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self._model_extensions):
                        # Get relative path from the specified directory
                        rel_path = os.path.relpath(os.path.join(root, file), directory)
                        model_files.append(rel_path)
            
            self._log(
                f"Found {len(model_files)} model files in {directory}",
                "debug"
            )
            return model_files
            
        except Exception as e:
            # This error will be caught by the decorator, but we log it here for context
            self._log(
                f"Error scanning directory {directory}: {str(e)}",
                "error",
                exc_info=True
            )
            # The decorator will return an empty list due to fallback_factory
            return []
    
    def _files_are_identical(self, file1: str, file2: str) -> bool:
        """🔍 Check apakah dua file identical (size dan timestamp)"""
        try:
            if not (os.path.exists(file1) and os.path.exists(file2)):
                return False
            
            stat1 = os.stat(file1)
            stat2 = os.stat(file2)
            
            # Compare size dan modification time
            return (stat1.st_size == stat2.st_size and 
                   abs(stat1.st_mtime - stat2.st_mtime) < 2)  # 2 second tolerance
            
        except Exception:
            return False
    
    @with_error_handling(
        component="pretrained",
        operation="get_sync_status",
        fallback_factory=lambda: {
            'error': 'Error getting sync status',
            'sync_needed': False,
            'local_files_count': 0,
            'drive_files_count': 0,
            'only_in_local': [],
            'only_in_drive': [],
            'in_both': [],
            'local_dir': '',
            'drive_dir': ''
        }
    )
    @log_errors(level="error")
    def get_sync_status(self, local_dir: str, drive_dir: str) -> Dict[str, Any]:
        """Get sync status between local and drive directories
        
        Args:
            local_dir: Local directory path
            drive_dir: Drive directory path
            
        Returns:
            Dictionary containing:
                - local_files_count: Number of files in local directory
                - drive_files_count: Number of files in drive directory
                - only_in_local: List of files only in local directory
                - only_in_drive: List of files only in drive directory
                - in_both: List of files in both directories
                - sync_needed: Boolean indicating if sync is needed
                - local_dir: The local directory path that was checked
                - drive_dir: The drive directory path that was checked
                - error: Error message if any occurred
        """
        # Input validation
        if not local_dir or not drive_dir:
            error_msg = "Local and drive directories must be specified"
            self._log(error_msg, "error")
            return {
                'error': error_msg,
                'sync_needed': False,
                'local_dir': local_dir,
                'drive_dir': drive_dir
            }
        
        # Check if directories exist (but don't fail if they don't)
        local_exists = os.path.exists(local_dir) and os.path.isdir(local_dir)
        drive_exists = os.path.exists(drive_dir) and os.path.isdir(drive_dir)
        
        if not local_exists:
            self._log(f"Local directory not found or not a directory: {local_dir}", "warning")
        if not drive_exists:
            self._log(f"Drive directory not found or not a directory: {drive_dir}", "warning")
        
        # Get file lists with error handling
        local_files = set()
        drive_files = set()
        
        if local_exists:
            local_files = set(self._get_model_files(local_dir))
        if drive_exists:
            drive_files = set(self._get_model_files(drive_dir))
        
        # Find differences
        only_in_local = sorted(list(local_files - drive_files))
        only_in_drive = sorted(list(drive_files - local_files))
        in_both = sorted(list(local_files & drive_files))
        
        # Prepare status dictionary
        status = {
            'local_files_count': len(local_files),
            'drive_files_count': len(drive_files),
            'only_in_local': only_in_local,
            'only_in_drive': only_in_drive,
            'in_both': in_both,
            'sync_needed': len(only_in_local) > 0 or len(only_in_drive) > 0,
            'local_dir': local_dir,
            'drive_dir': drive_dir,
            'local_exists': local_exists,
            'drive_exists': drive_exists
        }
        
        # Log detailed status
        self._log(
            f"Sync status - "
            f"Local: {len(local_files)} files ({'exists' if local_exists else 'missing'}), "
            f"Drive: {len(drive_files)} files ({'exists' if drive_exists else 'missing'}), "
            f"Only local: {len(only_in_local)}, "
            f"Only drive: {len(only_in_drive)}, "
            f"In both: {len(in_both)}",
            "info"
        )
        
        # Log detailed file differences if debug level is enabled
        if self._logger_bridge and hasattr(self._logger_bridge, 'debug'):
            if only_in_local:
                self._log("Files only in local: " + ", ".join(only_in_local[:5] + 
                        (['...'] if len(only_in_local) > 5 else [])) + 
                        f" ({len(only_in_local)} total)", "debug")
            if only_in_drive:
                self._log("Files only in drive: " + ", ".join(only_in_drive[:5] + 
                        (['...'] if len(only_in_drive) > 5 else [])) + 
                        f" ({len(only_in_drive)} total)", "debug")
        
        return status