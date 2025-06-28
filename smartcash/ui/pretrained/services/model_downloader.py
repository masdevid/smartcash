# File: smartcash/ui/pretrained/services/model_downloader.py
"""
File: smartcash/ui/pretrained/services/model_downloader.py
Deskripsi: Complete service untuk downloading pretrained models dengan progress tracking
"""

import os
import requests
import hashlib
from typing import Callable, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class PretrainedModelDownloader:
    """🚀 Service untuk downloading pretrained models dengan progress tracking"""
    
    def __init__(self):
        self.logger = logger
        self._download_urls = {
            'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt'
        }
        self._expected_sizes = {
            'yolov5s': 14_400_000  # ~14.4MB
        }
    
    def download_yolov5s(self, models_dir: str, 
                        progress_callback: Optional[Callable[[int, str], None]] = None,
                        status_callback: Optional[Callable[[str], None]] = None) -> bool:
        """📥 Download YOLOv5s model dengan progress tracking
        
        Args:
            models_dir: Directory untuk menyimpan model
            progress_callback: Callback untuk update progress (progress_pct, message)
            status_callback: Callback untuk update status message
            
        Returns:
            True jika download berhasil, False jika gagal
        """
        try:
            model_type = 'yolov5s'
            url = self._download_urls[model_type]
            output_path = os.path.join(models_dir, f'{model_type}.pt')
            
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