# File: smartcash/ui/pretrained/services/model_downloader.py
"""
File: smartcash/ui/pretrained/services/model_downloader.py
Deskripsi: Complete service untuk downloading pretrained models dengan progress tracking
"""

import os
import requests
import hashlib
import logging
import time
from typing import Callable, Optional, Dict, Any, TYPE_CHECKING, Union
from concurrent.futures import ThreadPoolExecutor

from smartcash.ui.types import ProgressTrackerType, StatusCallback
from smartcash.ui.pretrained.utils import (
    with_error_handling,
    log_errors,
    get_logger
)
from smartcash.ui.pretrained.utils.progress_adapter import PretrainedProgressAdapter

# Set up logger
logger = get_logger()

if TYPE_CHECKING:
    from smartcash.ui.utils.ui_logger import UILogger as LoggerBridge
else:
    LoggerBridge = Any  # For runtime type hints

class PretrainedModelDownloader:
    """🚀 Service untuk downloading pretrained models dengan progress tracking"""
    
    def __init__(self, 
                 logger_bridge: Optional[LoggerBridge] = None,
                 progress_tracker: Optional[PretrainedProgressAdapter] = None):
        """Initialize downloader with optional logger bridge and progress tracker
        
        Args:
            logger_bridge: LoggerBridge instance for centralized logging
            progress_tracker: Optional progress tracker instance
        """
        self._logger_bridge = logger_bridge
        self._progress_tracker = progress_tracker
        self._download_urls = {
            'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
        }
        self._expected_sizes = {
            'yolov5s': 14_400_000  # ~14.4MB
        }
        self._last_downloaded_files = {}  # Track last downloaded files by model type
        self._progress_callback = None
        self._status_callback = None
        
    def set_logger_bridge(self, logger_bridge: LoggerBridge) -> None:
        """Set the logger bridge for this downloader
        
        Args:
            logger_bridge: LoggerBridge instance for centralized logging
        """
        self._logger_bridge = logger_bridge
        
    def set_progress_callbacks(self, 
                            progress_cb: Optional[ProgressTrackerType] = None,
                            status_cb: Optional[StatusCallback] = None) -> None:
        """Set progress and status callbacks or progress tracker for download operations
        
        Args:
            progress_cb: ProgressTracker instance or callback for progress updates (progress_pct, message)
            status_cb: Callback for status updates (message) - optional if using ProgressTracker
        """
        if isinstance(progress_cb, PretrainedProgressAdapter):
            self._progress_tracker = progress_cb
            # If status callback not provided, use the one from progress tracker
            if status_cb is None:
                status_cb = progress_cb.get_status_callback()
        
        self._progress_callback = progress_cb
        self._status_callback = status_cb or (lambda _: None)
        
    def _safe_callback(self, callback: Optional[Callable], *args, **kwargs) -> None:
        """Safely execute a callback with error handling
        
        Args:
            callback: The callback function to execute
            *args: Positional arguments to pass to the callback
            **kwargs: Keyword arguments to pass to the callback
        """
        if callback is not None:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self._log_error(f"Error in callback: {str(e)}", exc_info=True)
    
    def _update_progress(self, progress: int, message: str, 
                        callback: Optional[ProgressTrackerType] = None) -> None:
        """Update progress using provided or registered callback or progress tracker
        
        Args:
            progress: Progress percentage (0-100)
            message: Progress message
            callback: ProgressTracker instance or callback to use (falls back to registered callback if None)
        """
        target_callback = callback or self._progress_tracker or self._progress_callback
        
        if target_callback is None:
            return
            
        if isinstance(target_callback, PretrainedProgressAdapter):
            target_callback.update_progress(progress, message)
        elif callable(target_callback):
            self._safe_callback(target_callback, progress, message)
            
        self._log_debug(f"Progress: {progress}% - {message}")
    
    def _update_status(self, message: str, 
                      callback: Optional[Union[PretrainedProgressAdapter, StatusCallback]] = None) -> None:
        """Update status message using provided or registered callback or progress tracker
        
        Args:
            message: Status message
            callback: ProgressTracker instance or callback to use (falls back to registered callback if None)
        """
        target_callback = callback or self._progress_tracker or self._status_callback
        
        if target_callback is None:
            return
            
        if isinstance(target_callback, PretrainedProgressAdapter):
            target_callback.update_status(message)
        elif callable(target_callback):
            self._safe_callback(target_callback, message)
            
        self._log_info(message)
        
    def _log(self, level: str, message: str, **kwargs) -> None:
        """Generic log method that handles different log levels
        
        Args:
            level: Log level ('debug', 'info', 'warning', 'error')
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        if self._logger_bridge and hasattr(self._logger_bridge, level):
            getattr(self._logger_bridge, level)(message, **kwargs)
    
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message using logger_bridge if available
        
        Args:
            message: The message to log
            **kwargs: Additional arguments to pass to the logger
        """
        self._log('debug', message, **kwargs)
            
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message using logger_bridge if available
        
        Args:
            message: The message to log
        """
        logger.info(message)
            
    def _log_warning(self, message: str) -> None:
        """Log warning message using logger_bridge if available
        
        Args:
            message: The message to log
        """
        logger.warning(f"⚠️ {message}")
            
    def _log_error(self, message: str, exc_info: bool = False) -> None:
        """Log error message using logger_bridge if available
        
        Args:
            message: The message to log
            exc_info: Whether to include exception info
        """
        if exc_info:
            logger.exception(f"❌ {message}")
        else:
            logger.error(f"❌ {message}")
    
    def _cleanup_file(self, file_path: str, status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """🧹 Hapus file dengan error handling dan status update
        
        Args:
            file_path: Path ke file yang akan dihapus
            status_callback: Callback untuk update status
            
        Returns:
            bool: True jika berhasil, False jika gagal
        """
        if not file_path or not os.path.exists(file_path):
            return False
            
        try:
            filename = os.path.basename(file_path)
            self._update_status(f"🧹 Membersihkan file: {filename}", status_callback)
            
            os.remove(file_path)
            success_msg = f"✅ Berhasil menghapus file: {filename}"
            self._log_info(success_msg)
            return True
            
        except Exception as e:
            error_msg = f"⚠️ Gagal menghapus file {file_path}: {str(e)}"
            self._log_error(error_msg, exc_info=True)
            self._update_status(error_msg, status_callback)
            return False
    
    @with_error_handling(
        component="model_downloader",
        operation="download_yolov5s",
        fallback_value=False,
        log_level="error"
    )
    def download_yolov5s(self, models_dir: str, 
                        progress_callback: Optional[Callable[[int, str], None]] = None,
                        status_callback: Optional[Callable[[str], None]] = None,
                        model_url: Optional[str] = None,
                        use_registered_callbacks: bool = True) -> bool:
        """📥 Download YOLOv5s model dengan progress tracking dan auto-cleanup
        
        Args:
            models_dir: Directory untuk menyimpan model
            progress_callback: Callback untuk update progress (progress_pct, message)
            status_callback: Callback untuk update status message
            model_url: URL kustom untuk download model (opsional)
            use_registered_callbacks: Gunakan callback yang sudah terdaftar jika True
            
        Returns:
            bool: True jika download berhasil, False jika gagal
        """
        model_type = 'yolov5s'
        url = model_url or self._download_urls.get(model_type)
        if not url:
            raise ValueError(f"No download URL available for model type: {model_type}")
            
        output_path = os.path.join(models_dir, os.path.basename(url))
        
        # Use provided callbacks or fall back to registered ones if use_registered_callbacks is True
        current_progress_cb = progress_callback or (self._progress_callback if use_registered_callbacks else None)
        current_status_cb = status_callback or (self._status_callback if use_registered_callbacks else None)
        
        try:
            # Check if we have a previous download with different URL
            if model_type in self._last_downloaded_files:
                last_url, last_path = self._last_downloaded_files[model_type]
                if last_url != url and os.path.exists(last_path) and os.path.exists(output_path):
                    self._cleanup_file(last_path, current_status_cb)
            
            # Update last downloaded file info
            self._last_downloaded_files[model_type] = (url, output_path)
            
            # Update status
            status_msg = "📁 Mempersiapkan direktori download..."
            if current_status_cb:
                current_status_cb(status_msg)
            logger.info(f"Created directory: {models_dir}")
            
            # Create directory jika belum ada
            os.makedirs(models_dir, exist_ok=True)
            
            # Use the provided URL or fall back to default
            url = model_url or self._download_urls.get(model_type)
            if not url:
                raise ValueError(f"No download URL available for model type: {model_type}")
            
            # Generate output path
            output_path = os.path.join(models_dir, os.path.basename(url))
            
            # Register callbacks if using the registered callbacks system
            if use_registered_callbacks and self._progress_tracker:
                self._progress_tracker.register_callbacks(progress_callback, status_callback)
            
            # Start download
            status_msg = f"📥 Mengunduh {model_type} dari GitHub..."
            if status_callback:
                status_callback(status_msg)
            logger.info(status_msg)
            
            return self._download_with_progress(
                url, 
                output_path, 
                model_type, 
                progress_callback=progress_callback,
                status_callback=status_callback
            )
            
        except Exception as e:
            error_msg = f"Download failed: {str(e)}"
            self._log_error(error_msg, exc_info=True)
            if status_callback:
                status_callback(f"❌ {error_msg}")
            return False
    
    @with_error_handling(
        component="model_downloader",
        operation="_download_with_progress",
        fallback_value=False,
        log_level="error"
    )
    def _download_with_progress(
        self,
        url: str,
        output_path: str,
        model_type: str,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ) -> bool:
        """🔄 Internal download dengan detailed progress tracking
        
        Args:
            url: URL to download from
            output_path: Local path to save the downloaded file
            model_type: Type of model being downloaded
            progress_callback: Optional callback for progress updates
            status_callback: Optional callback for status updates
            
        Returns:
            bool: True if download and validation succeeded, False otherwise
        """
        logger.info(f"Starting download from {url} to {output_path}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        # Update progress and status
        self._update_progress(5, "Memulai unduhan...", progress_callback)
        self._update_status("🔍 Menghubungkan ke server...", status_callback)
        
        try:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress every 1MB or on last chunk
                        if total_size > 0 and (downloaded % (1024 * 1024) == 0 or downloaded == total_size):
                            progress = min(95, int((downloaded / total_size) * 90) + 5)  # 5-95%
                            size_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            message = f"Unduh {size_mb:.1f}/{total_mb:.1f} MB"
                            self._update_progress(progress, message, progress_callback)
            
            # Validate downloaded file
            self._update_status("🔍 Memvalidasi model...", status_callback)
            self._update_progress(95, "Memvalidasi file...", progress_callback)
            
            if self._validate_model_file(output_path, model_type):
                success_msg = f"✅ {model_type} berhasil diunduh"
                self._update_status(success_msg, status_callback)
                self._update_progress(100, "Unduhan selesai", progress_callback)
                logger.info(success_msg)
                return True
            
            # If validation fails
            self._handle_validation_failure(output_path, status_callback)
            return False
            
        except Exception as e:
            # Clean up on any error during download
            self._cleanup_download(output_path, f"❌ Gagal mengunduh: {str(e)}", status_callback)
            raise  # Re-raise to be handled by the decorator
    
    def _handle_validation_failure(self, file_path: str, status_callback: Optional[Callable[[str], None]] = None) -> None:
        """Handle validation failure by cleaning up and updating status"""
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"Removed invalid download: {file_path}")
            except Exception as e:
                logger.error(f"Failed to remove invalid download: {str(e)}", exc_info=True)
        
        error_msg = "❌ Validasi file unduhan gagal"
        self._update_status(error_msg, status_callback)
        logger.error(error_msg)
    
    def _cleanup_download(self, file_path: str, error_msg: str, 
                         status_callback: Optional[Callable[[str], None]] = None) -> None:
        """Clean up failed download and update status"""
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Gagal membersihkan file: {str(e)}", exc_info=True)
        
        self._update_status(error_msg, status_callback)
        logger.error(error_msg)
            
    def _update_progress_with_callback(self, progress: int, message: str, 
                                     callback: Optional[Callable[[int, str], None]] = None) -> None:
        """Update progress using provided callback and/or registered callback
        
        Args:
            progress: Progress percentage (0-100)
            message: Progress message
            callback: Optional callback to use (falls back to registered callback if None)
        """
        self._update_progress(progress, message, callback)
    
    def _update_status_with_callback(self, message: str, 
                                   callback: Optional[Callable[[str], None]] = None) -> None:
        """Update status using provided callback and/or registered callback
        
        Args:
            message: Status message
            callback: Optional callback to use (falls back to registered callback if None)
        """
        self._update_status(message, callback)
            
    def _handle_download_error(self, error_msg: str, output_path: str, 
                            status_callback: Optional[Callable[[str], None]] = None) -> None:
        """Handle download errors consistently
        
        Args:
            error_msg: Error message to log and display
            output_path: Path to the output file for cleanup
            status_callback: Optional callback for status updates
        """
        logger.error(error_msg, exc_info=True)
        self._update_status_with_callback(f"❌ {error_msg}", status_callback)
        
        # Cleanup partial file
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
                logger.debug(f"Membersihkan unduhan yang gagal: {output_path}")
            except Exception as cleanup_error:
                error_msg = f"Gagal membersihkan unduhan: {str(cleanup_error)}"
                logger.error(error_msg, exc_info=True)
                
    @with_error_handling(
        component="model_downloader",
        operation="_validate_model_file",
        fallback_value=False,
        log_level="warning"
    )
    def _validate_model_file(self, file_path: str, model_type: str) -> bool:
        """Validate the downloaded model file
        
        Args:
            file_path: Path to the model file
            model_type: Type of model being validated
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        if not os.path.isfile(file_path):
            self._log_warning(f"File not found: {file_path}")
            return False
            
        try:
            file_size = os.path.getsize(file_path)
            expected_size = self._expected_sizes.get(model_type, 0)
            
            # Check file is not empty
            if file_size < 1024:  # < 1KB
                self._log_warning(f"File too small: {file_size} bytes")
                return False
                
            # Check file size is within expected range (allow 20% variance)
            if expected_size > 0:
                min_expected = expected_size * 0.8
                max_expected = expected_size * 1.2
                if not (min_expected <= file_size <= max_expected):
                    self._log_warning(f"Unexpected file size: {file_size} (expected ~{expected_size})")
                    return False
            
            # Basic file extension check
            if not file_path.lower().endswith('.pt'):
                self._log_warning(f"Invalid file extension: {file_path}")
                return False
                
            # TODO: Add more validation (e.g., file signature, checksum)
            
            success_msg = f"Model file validated: {os.path.basename(file_path)} ({file_size} bytes)"
            self._log_info(success_msg)
            return True
            
        except Exception as e:
            error_msg = f"Error validating file {file_path}: {str(e)}"
            self._log_error(error_msg, exc_info=True)
            return False
    
    def get_download_info(self, model_type: str = 'yolov5s') -> Dict[str, Any]:
        """📊 Get download information untuk model
        
        Args:
            model_type: Type of model to get info for (default: 'yolov5s')
            
        Returns:
            Dict containing model download information
        """
        try:
            return {
                'model_type': model_type,
                'url': self._download_urls.get(model_type, ''),
                'expected_size_mb': self._expected_sizes.get(model_type, 0) / (1024 * 1024),
                'available': model_type in self._download_urls
            }
        except Exception as e:
            error_msg = f"Failed to get download info for {model_type}: {str(e)}"
            self._log_error(error_msg, exc_info=True)
            return {
                'model_type': model_type,
                'url': '',
                'expected_size_mb': 0,
                'available': False,
                'error': error_msg
            }