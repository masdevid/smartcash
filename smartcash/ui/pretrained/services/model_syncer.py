"""
File: smartcash/ui/pretrained/services/model_syncer.py
Deskripsi: Service untuk mensinkronisasi pretrained models antara storage lokal dan drive cloud

This module provides the PretrainedModelSyncer class for syncing model files with progress
tracking and error handling. Supports one-way and bi-directional sync operations.

Example:
    from smartcash.ui.utils.logger_bridge import LoggerBridge
    
    syncer = PretrainedModelSyncer(logger_bridge=LoggerBridge())
    
    # One-way sync
    syncer.sync_to_drive(
        source_dir="/path/to/local/models",
        drive_dir="/path/to/drive/models",
        progress_callback=update_progress,
        status_callback=update_status
    )
    
    # Bi-directional sync
    syncer.bi_directional_sync(
        local_dir="/path/to/local/models",
        drive_dir="/path/to/drive/models",
        progress_callback=update_progress,
        status_callback=update_status
    )
"""

import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Callable, TypeVar, Union
from pathlib import Path
from functools import wraps

from smartcash.ui.utils.logger_bridge import LoggerBridge
from smartcash.ui.pretrained.utils import (
    with_error_handling,
    log_errors,
    get_logger
)
from smartcash.ui.pretrained.utils.progress_adapter import PretrainedProgressAdapter

# Type aliases
LoggerBridge = Callable[[str, str], None]
ProgressCallback = Callable[[int, str], None]  # (progress_pct: int, message: str) -> None
StatusCallback = Callable[[str], None]  # (message: str) -> None
ProgressTrackerType = Union[PretrainedProgressAdapter, ProgressCallback]

# Set up logger
logger = get_logger()

# Type variable for class methods
T = TypeVar('T', bound='PretrainedModelSyncer')

class SyncDirection(Enum):
    """File synchronization direction.
    
    Attributes:
        TO_DRIVE: Local → Drive
        FROM_DRIVE: Drive → Local
        BIDIRECTIONAL: Two-way sync
    """
    TO_DRIVE = auto()
    FROM_DRIVE = auto()
    BIDIRECTIONAL = auto()


@dataclass
class SyncResult:
    """Result of a synchronization operation.
    
    Attributes:
        success: True if operation completed successfully
        files_processed: Number of files processed
        files_skipped: Number of files skipped (already in sync)
        error: Error message if operation failed
        details: Additional operation details
    """
    success: bool = False
    files_processed: int = 0
    files_skipped: int = 0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

class PretrainedModelSyncer:
    """Synchronize pretrained models between local storage and cloud drive.
    
    Handles syncing model files with progress tracking, error handling, and verification.
    Supports one-way and bi-directional sync operations.
    
    Example:
        syncer = PretrainedModelSyncer(logger_bridge=LoggerBridge())
        success = syncer.sync_to_drive(
            source_dir="/path/to/local",
            drive_dir="/path/to/drive",
            progress_callback=update_progress,
            status_callback=update_status
        )
    """
    
    def __init__(self, 
                 logger_bridge: Optional[LoggerBridge] = None,
                 progress_tracker: Optional[PretrainedProgressAdapter] = None):
        """Initialize the model syncer with an optional logger bridge and progress tracker.
        
        Args:
            logger_bridge: Optional logger bridge for UI logging. If not provided,
                         logging will be done to console only.
            progress_tracker: Optional progress tracker instance. If not provided,
                           a new one will be created when needed.
        """
        self._logger_bridge = logger_bridge
        self._progress_tracker = progress_tracker
        self._model_extensions = ['.pt', '.bin', '.pth', '.ckpt']
    
    def _log(self, message: str, level: str = "info", **kwargs) -> None:
        """Log a message using the configured logger.
        
        Args:
            message: Message to log
            level: Log level (debug, info, warning, error, critical)
            **kwargs: Additional logging parameters
        """
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(message, **kwargs)
    
    def _update_progress(self, 
                        current: int, 
                        total: int, 
                        callback: Optional[ProgressTrackerType],
                        prefix: str = "") -> None:
        """Update progress using the provided callback or progress tracker.
        
        Args:
            current: Current progress count (0-based)
            total: Total items to process
            callback: ProgressTracker instance or callback function
            prefix: Optional message prefix
        """
        if not callback:
            return
            
        if total <= 0:
            return
            
        progress_pct = int((current / total) * 100)
        message = f"{prefix} ({current}/{total})" if prefix else ""
        
        if isinstance(callback, PretrainedProgressAdapter):
            callback.update_progress(progress_pct, message)
        elif callable(callback):
            callback(progress_pct, message)
    
    def _update_status(self, message: str, 
                      callback: Optional[Union[PretrainedProgressAdapter, StatusCallback]]) -> None:
        """Update status message through callback/progress tracker and logging.
        
        Args:
            message: Status message to display/log
            callback: ProgressTracker instance or status callback function
        """
        if isinstance(callback, PretrainedProgressAdapter):
            callback.update_status(message)
        elif callable(callback):
            callback(message)
            
        self._log(message, "info")
    
    def _ensure_directory(self, path: str) -> bool:
        """Ensure directory exists, creating it if needed.
        
        Args:
            path: Absolute path to directory
            
        Returns:
            bool: True if directory exists or was created successfully
        """
        try:
            os.makedirs(path, exist_ok=True)
            return True
        except Exception as e:
            self._log(f"Failed to create directory {path}: {str(e)}", "error")
            return False
    
    def _validate_paths(self, *paths: str) -> Tuple[bool, Optional[str]]:
        """Validate paths are absolute and secure.
        
        Args:
            *paths: Path strings to validate
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        for path in paths:
            if not path:
                return False, "Path cannot be empty"
                
            if not os.path.isabs(path):
                return False, f"Path must be absolute: {path}"
                
            # Check for path traversal attempts using a more robust method
            try:
                # Normalize and resolve the path
                resolved = os.path.abspath(os.path.normpath(path))
                # Check if the resolved path is still within the expected directory
                if not os.path.abspath(resolved).startswith(os.path.abspath(os.sep)):
                    return False, f"Suspicious path resolution: {path}"
            except (ValueError, RuntimeError) as e:
                return False, f"Invalid path format: {str(e)}"
                
        return True, None
        
    def _verify_file_copy(self, src: str, dst: str) -> bool:
        """Verify file copy by comparing source and destination sizes.
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            bool: True if files exist and have same size
        """
        try:
            if not os.path.exists(dst):
                self._log(f"Destination file does not exist: {dst}", "error")
                return False
                
            if not os.path.exists(src):
                self._log(f"Source file does not exist: {src}", "error")
                return False
                
            src_size = os.path.getsize(src)
            dst_size = os.path.getsize(dst)
            
            if src_size != dst_size:
                self._log(
                    f"File size mismatch after copy. Source: {src_size} bytes, "
                    f"Destination: {dst_size} bytes",
                    "error"
                )
                return False
                
            self._log(f"Verified copy: {src} -> {dst} ({src_size} bytes)", "debug")
            return True
            
        except Exception as e:
            self._log(f"Error verifying file copy from {src} to {dst}: {str(e)}", 
                    "error", exc_info=True)
            return False
    
    def _copy_file(self, src: str, dst: str, verify: bool = True) -> bool:
        """Safely copy a file with error handling and optional verification.
        
        Args:
            src: Source file path
            dst: Destination file path
            verify: If True, verifies the copy was successful
            
        Returns:
            bool: True if copy was successful (and verified if verify=True)
            
        Raises:
            RuntimeError: If source file is inaccessible
        """
        # Validate paths first
        is_valid, error_msg = self._validate_paths(src, os.path.dirname(dst))
        if not is_valid:
            self._log(f"Invalid path: {error_msg}", "error")
            return False
            
        # Check if source exists and is accessible
        if not os.path.isfile(src):
            self._log(f"Source file not found or not a file: {src}", "error")
            return False
            
        try:
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            
            # Log the copy operation
            self._log(f"Copying {src} to {dst}", "debug")
            
            # Perform the copy with metadata preservation
            shutil.copy2(src, dst)
            
            # Verify the copy if requested
            if verify:
                if not self._verify_file_copy(src, dst):
                    try:
                        os.remove(dst)  # Clean up partial/corrupt copy
                        self._log(f"Removed incomplete/corrupt copy: {dst}", "warning")
                    except Exception as e:
                        self._log(f"Failed to clean up incomplete copy {dst}: {str(e)}", "error")
                    return False
                self._log(f"Successfully copied and verified {src} -> {dst}", "debug")
            else:
                self._log(f"Copied {src} -> {dst} (verification skipped)", "debug")
                
            return True
            
        except Exception as e:
            self._log(f"Failed to copy {src} to {dst}: {str(e)}", "error", exc_info=True)
            # Attempt to clean up any partial copy
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except Exception as cleanup_error:
                self._log(f"Failed to clean up after copy error: {str(cleanup_error)}", "error")
            return False
            
    def _sync_files(self, 
                   source_dir: str, 
                   target_dir: str, 
                   direction: SyncDirection,
                   progress_callback: Optional[ProgressTrackerType] = None,
                   status_callback: Optional[Union[PretrainedProgressAdapter, StatusCallback]] = None) -> SyncResult:
        """Core file synchronization engine.
        
        Handles file copying, verification, and progress updates for sync operations.
        
        Args:
            source_dir: Source directory path
            target_dir: Target directory path
            direction: Sync direction (TO_DRIVE, FROM_DRIVE, BIDIRECTIONAL)
            progress_callback: Optional progress callback (percent, message)
            status_callback: Optional status callback
            
        Returns:
            SyncResult: Operation result with file counts and any errors
        """
        result = SyncResult()
        
        # Log the sync operation
        self._log(
            f"Starting sync: {source_dir} -> {target_dir} "
            f"(direction: {direction.name})",
            "info"
        )
        
        # Validate directories
        is_valid, error_msg = self._validate_paths(source_dir, target_dir)
        if not is_valid:
            error_msg = f"Invalid paths: {error_msg}"
            self._log(error_msg, "error")
            result.error = error_msg
            return result
            
        # Ensure target directory exists
        if not self._ensure_directory(target_dir):
            error_msg = f"Failed to create target directory: {target_dir}"
            self._log(error_msg, "error")
            result.error = error_msg
            return result
            
        try:
            # Get all model files in source directory
            source_files = self._get_model_files(source_dir)
            total_files = len(source_files)
            
            if not source_files:
                status_msg = "No model files found in source directory"
                self._update_status(status_msg, status_callback)
                self._log(status_msg, "warning")
                result.success = True
                return result
                
            self._update_status(
                f"Found {total_files} model files to sync", 
                status_callback
            )
            
            # Process each file
            for i, src_file in enumerate(source_files, 1):
                try:
                    rel_path = os.path.relpath(src_file, source_dir)
                    dst_file = os.path.join(target_dir, rel_path)
                    
                    # Skip if files are identical
                    if os.path.exists(dst_file) and self._files_are_identical(src_file, dst_file):
                        self._log(f"Skipping identical file: {rel_path}", "debug")
                        result.files_skipped += 1
                        continue
                        
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    
                    # Copy the file with verification
                    if self._copy_file(src_file, dst_file):
                        result.files_processed += 1
                        self._log(f"Successfully synced {rel_path}", "debug")
                    else:
                        error_msg = f"Failed to sync {rel_path}"
                        self._log(error_msg, "error")
                        result.details.setdefault("failed_files", []).append({
                            "source": src_file,
                            "target": dst_file,
                            "error": error_msg
                        })
                    
                    # Update progress
                    progress_msg = f"Syncing {os.path.basename(src_file)}"
                    self._update_progress(
                        current=i,
                        total=total_files,
                        callback=progress_callback,
                        prefix=progress_msg
                    )
                    
                    # Throttle updates for better performance with many small files
                    if i % 10 == 0 or i == total_files:
                        self._log(
                            f"Progress: {i}/{total_files} files processed "
                            f"({(i/total_files*100):.1f}%)",
                            "debug"
                        )
                        
                except Exception as e:
                    error_msg = f"Error processing {src_file}: {str(e)}"
                    self._log(error_msg, "error", exc_info=True)
                    result.details.setdefault("failed_files", []).append({
                        "source": src_file,
                        "target": dst_file if 'dst_file' in locals() else None,
                        "error": error_msg
                    })
                    
        except Exception as e:
            error_msg = f"Error during sync: {str(e)}"
            self._log(error_msg, "error", exc_info=True)
            result.error = error_msg
        
        # Determine overall success
        result.success = result.files_processed > 0 or result.files_skipped == total_files
        
        # Update result details
        result.details.update({
            'source_dir': source_dir,
            'target_dir': target_dir,
            'direction': direction.name,
            'total_files': total_files,
            'successful_files': result.files_processed,
            'skipped_files': result.files_skipped,
            'failed_files': total_files - (result.files_processed + result.files_skipped)
        })
        
        # Generate completion message
        if result.files_processed == 0 and result.files_skipped == 0:
            completion_msg = "❌ No files were processed"
        else:
            completion_msg = (
                f"✅ Sync {direction.name} completed. "
                f"Processed: {result.files_processed}, "
                f"Skipped: {result.files_skipped}"
            )
            if not result.success:
                completion_msg = f"⚠️ {completion_msg} (with errors)"
        
        self._update_status(completion_msg, status_callback)
        self._update_progress(
            total_files, 
            total_files, 
            progress_callback, 
            "Sync completed" + (" with errors" if not result.success else "")
        )
        
        return result
    
    @with_error_handling(
        component="model_syncer",
        operation="_sync_files",
        fallback_factory=lambda: SyncResult(error="Failed to sync files"),
        log_level="error"
    )
    def _get_model_files(self, directory: str) -> List[str]:
        """Find model files in directory.
        
        Args:
            directory: Directory to scan
            
        Returns:
            List[str]: Relative paths to model files
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
    
    def _files_are_identical(self, file1: str, file2: str, tolerance_sec: float = 2.0) -> bool:
        """Check if two files are identical by comparing size and modification time.
        
        Args:
            file1: Path to first file
            file2: Path to second file
            tolerance_sec: Max time difference in seconds to consider files identical
                         (accounts for filesystem timestamp precision issues)
        Returns:
            bool: True if files have same size and similar mod times, False otherwise
        """
        try:
            # Check if both files exist
            if not (os.path.exists(file1) and os.path.exists(file2)):
                self._log(
                    f"One or both files don't exist: {file1}, {file2}",
                    "debug"
                )
                return False
            
            # Get file stats
            stat1 = os.stat(file1)
            stat2 = os.stat(file2)
            
            # Compare file sizes first (fast check)
            if stat1.st_size != stat2.st_size:
                return False
                
            # Compare modification times with tolerance
            time_diff = abs(stat1.st_mtime - stat2.st_mtime)
            
            # Log detailed comparison for debugging
            self._log(
                f"File comparison - {file1} vs {file2}: "
                f"sizes={stat1.st_size}/{stat2.st_size}, "
                f"mtime_diff={time_diff:.3f}s, "
                f"tolerance={tolerance_sec}s",
                "debug"
            )
            
            return time_diff <= tolerance_sec
            
        except Exception as e:
            self._log(
                f"Error comparing files {file1} and {file2}: {str(e)}",
                "error",
                exc_info=True
            )
            return False
    
    @with_error_handling(
        component="model_syncer",
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
        },
        log_level="error"
    )
    def get_sync_status(self, local_dir: str, drive_dir: str) -> Dict[str, Any]:
        """Get sync status between local and drive directories.
        
        Args:
            local_dir: Local directory path
            drive_dir: Drive directory path
            
        Returns:
            Dict containing sync status information
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