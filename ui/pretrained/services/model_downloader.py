"""
File: smartcash/ui/pretrained/services/model_downloader.py  
Deskripsi: Service untuk downloading pretrained models
"""

import os
import requests
from typing import Callable, Optional
from smartcash.common.logger import get_logger

class PretrainedModelDownloader:
    """Service untuk downloading pretrained models dengan progress tracking"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def download_model(self, url: str, output_path: str, 
                      progress_callback: Optional[Callable[[int, str], None]] = None) -> bool:
        """
        Download model dengan progress tracking.
        
        Args:
            url: Download URL
            output_path: Output file path
            progress_callback: Progress callback function(progress, message)
            
        Returns:
            bool: Download success status
        """
        try:
            # Create directory jika belum ada
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Start download
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress
                        if progress_callback and total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            size_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            message = f"Downloaded {size_mb:.1f}/{total_mb:.1f} MB"
                            progress_callback(progress, message)
            
            self.logger.info(f"✅ Downloaded: {os.path.basename(output_path)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Download failed: {str(e)}")
            # Cleanup partial file
            if os.path.exists(output_path):
                os.remove(output_path)
            return False