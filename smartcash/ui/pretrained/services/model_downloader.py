# File: smartcash/ui/pretrained/services/model_downloader.py
"""
File: smartcash/ui/pretrained/services/model_downloader.py
Deskripsi: Complete service untuk downloading pretrained models dengan progress tracking
"""

import os
import requests
import hashlib
from typing import Callable, Optional, Dict, Any, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

if TYPE_CHECKING:
    from smartcash.common.logger import LoggerBridge
else:
    LoggerBridge = Any  # For runtime type hints

class PretrainedModelDownloader:
    """🚀 Service untuk downloading pretrained models dengan progress tracking"""
    
    def __init__(self, logger_bridge: Optional[LoggerBridge] = None):
        """Initialize downloader with optional logger bridge
        
        Args:
            logger_bridge: LoggerBridge instance for centralized logging
        """
        self._logger_bridge = logger_bridge
        self._download_urls = {
            'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
        }
        self._expected_sizes = {
            'yolov5s': 14_400_000  # ~14.4MB
        }
        self._last_downloaded_files = {}  # Track last downloaded files by model type
        
    def set_logger_bridge(self, logger_bridge: LoggerBridge) -> None:
        """Set the logger bridge for this downloader
        
        Args:
            logger_bridge: LoggerBridge instance for centralized logging
        """
        self._logger_bridge = logger_bridge
        
    def _log_debug(self, message: str, **kwargs) -> None:
        """Log debug message using logger_bridge if available"""
        if self._logger_bridge and hasattr(self._logger_bridge, 'debug'):
            self._logger_bridge.debug(message, **kwargs)
            
    def _log_info(self, message: str, **kwargs) -> None:
        """Log info message using logger_bridge if available"""
        if self._logger_bridge and hasattr(self._logger_bridge, 'info'):
            self._logger_bridge.info(message, **kwargs)
            
    def _log_warning(self, message: str, **kwargs) -> None:
        """Log warning message using logger_bridge if available"""
        if self._logger_bridge and hasattr(self._logger_bridge, 'warning'):
            self._logger_bridge.warning(message, **kwargs)
            
    def _log_error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """Log error message using logger_bridge if available"""
        if self._logger_bridge and hasattr(self._logger_bridge, 'error'):
            self._logger_bridge.error(message, exc_info=exc_info, **kwargs)
    
    def _cleanup_old_file(self, file_path: str, status_callback: Optional[Callable[[str], None]] = None) -> None:
        """🧹 Hapus file lama jika berbeda dengan yang baru
        
        Args:
            file_path: Path ke file yang akan dihapus
            status_callback: Callback untuk update status
        """
        if not file_path or not os.path.exists(file_path):
            return
            
        try:
            filename = os.path.basename(file_path)
            status_msg = f"🧹 Membersihkan file lama: {filename}"
            if status_callback:
                status_callback(status_msg)
                
            os.remove(file_path)
            success_msg = f"✅ Berhasil menghapus file lama: {file_path}"
            self._log_info(success_msg)
            
        except Exception as e:
            error_msg = f"⚠️ Gagal menghapus file lama {file_path}: {str(e)}"
            self._log_error(error_msg, exc_info=True)
            if status_callback:
                status_callback(error_msg)
    
    def download_yolov5s(self, models_dir: str, 
                        progress_callback: Optional[Callable[[int, str], None]] = None,
                        status_callback: Optional[Callable[[str], None]] = None,
                        model_url: Optional[str] = None) -> bool:
        """📥 Download YOLOv5s model dengan progress tracking dan auto-cleanup
        
        Args:
            models_dir: Directory untuk menyimpan model
            progress_callback: Callback untuk update progress (progress_pct, message)
            status_callback: Callback untuk update status message
            model_url: URL kustom untuk download model (opsional)
            
        Returns:
            True jika download berhasil, False jika gagal
        """
        try:
            model_type = 'yolov5s'
            url = model_url or self._download_urls[model_type]
            output_path = os.path.join(models_dir, f'{model_type}.pt')
            
            # Check if we have a previous download with different URL
            if model_type in self._last_downloaded_files:
                last_url, last_path = self._last_downloaded_files[model_type]
                if last_url != url and os.path.exists(last_path) and os.path.exists(output_path):
                    self._cleanup_old_file(last_path, status_callback)
            
            # Update last downloaded file info
            self._last_downloaded_files[model_type] = (url, output_path)
            
            # Update status
            if status_callback:
                status_callback("📁 Preparing download directory...")
            
            # Create directory jika belum ada
            os.makedirs(models_dir, exist_ok=True)
            
            # Check if model already exists
            if os.path.exists(output_path) and self._validate_model_file(output_path, model_type):
                if status_callback:
                    status_callback("✅ Model already exists and valid")
                if progress_callback:
                    progress_callback(100, "Model already available")
                return True
            
            # Start download
            if status_callback:
                status_callback(f"📥 Downloading {model_type} from GitHub...")
            
            return self._download_with_progress(url, output_path, model_type, 
                                              progress_callback, status_callback)
            
        except Exception as e:
            error_msg = f"❌ Download failed: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
            return False
    
    def _download_with_progress(self, url: str, output_path: str, model_type: str,
                               progress_callback: Optional[Callable[[int, str], None]] = None,
                               status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """🔄 Internal download dengan detailed progress tracking"""
        try:
            # Start download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            if progress_callback:
                progress_callback(5, "Download started...")
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        if progress_callback and total_size > 0:
                            progress = int((downloaded / total_size) * 85) + 5  # 5-90%
                            size_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            message = f"Downloaded {size_mb:.1f}/{total_mb:.1f} MB"
                            progress_callback(progress, message)
            
            # Validate downloaded file
            if status_callback:
                status_callback("🔍 Validating downloaded model...")
            if progress_callback:
                progress_callback(95, "Validating file...")
            
            if self._validate_model_file(output_path, model_type):
                if status_callback:
                    status_callback(f"✅ {model_type} downloaded successfully")
                if progress_callback:
                    progress_callback(100, "Download completed")
                return True
            else:
                # Remove invalid file
                if os.path.exists(output_path):
                    os.remove(output_path)
                error_msg = "❌ Downloaded file validation failed"
                if status_callback:
                    status_callback(error_msg)
                return False
            
        except requests.exceptions.RequestException as e:
            error_msg = f"❌ Network error: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
            return False
        except Exception as e:
            error_msg = f"❌ Download error: {str(e)}"
            self.logger.error(error_msg)
            if status_callback:
                status_callback(error_msg)
            # Cleanup partial file
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
    
    def _validate_model_file(self, file_path: str, model_type: str) -> bool:
        """✅ Validate downloaded model file"""
        try:
            if not os.path.isfile(file_path):
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            expected_size = self._expected_sizes.get(model_type, 0)
            
            if expected_size > 0 and file_size < expected_size * 0.8:  # Allow 20% variance
                self.logger.warning(f"⚠️ File size too small: {file_size} < {expected_size * 0.8}")
                return False
            
            # Check file is not empty
            if file_size < 1024:  # < 1KB
                self.logger.warning(f"⚠️ File too small: {file_size} bytes")
                return False
            
            # Basic file extension check
            if not file_path.endswith('.pt'):
                self.logger.warning(f"⚠️ Invalid file extension: {file_path}")
                return False
            
            self.logger.info(f"✅ Model file validated: {os.path.basename(file_path)} ({file_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Validation error: {str(e)}")
            return False
    
    def get_download_info(self, model_type: str = 'yolov5s') -> Dict[str, Any]:
        """📊 Get download information untuk model"""
        return {
            'model_type': model_type,
            'url': self._download_urls.get(model_type, ''),
            'expected_size_mb': self._expected_sizes.get(model_type, 0) / (1024 * 1024),
            'available': model_type in self._download_urls
        }