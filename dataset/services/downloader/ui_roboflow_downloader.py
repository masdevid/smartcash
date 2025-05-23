"""
File: smartcash/dataset/services/downloader/ui_roboflow_downloader.py
Deskripsi: Modified Roboflow downloader dengan UI progress callback (no tqdm)
"""

import os
import time
import requests
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.dataset.services.downloader.ui_progress_mixin import UIProgressMixin

class UIRoboflowDownloader(UIProgressMixin):
    """Roboflow downloader dengan UI progress callback."""
    
    def __init__(self, logger=None, timeout: int = 30, chunk_size: int = 8192):
        super().__init__()
        self.logger = logger or get_logger()
        self.timeout = timeout
        self.chunk_size = chunk_size
    
    def download_and_extract(self, download_url: str, output_path: Path, show_progress: bool = False) -> bool:
        """Download dan extract dengan progress callback (no tqdm)."""
        zip_path = output_path / "dataset.zip"
        
        try:
            # Step 1: Download
            self._notify_progress("download", 0, 100, "Memulai download...")
            
            if not self._download_file(download_url, zip_path):
                return False
            
            # Step 2: Extract
            self._notify_progress("extract", 0, 100, "Mengekstrak dataset...")
            
            if not self._extract_zip(zip_path, output_path, remove_zip=True):
                return False
            
            self._notify_progress("extract", 100, 100, "Ekstraksi selesai")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Download/extract gagal: {str(e)}")
            return False
    
    def _download_file(self, url: str, output_path: Path) -> bool:
        """Download file dengan progress callback."""
        try:
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Progress callback setiap 5%
                        if total_size > 0 and downloaded % (total_size // 20) < self.chunk_size:
                            progress = int((downloaded / total_size) * 100)
                            self._notify_progress("download", progress, 100, f"Download: {progress}%")
            
            self._notify_progress("download", 100, 100, "Download selesai")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Download error: {str(e)}")
            return False
    
    def _extract_zip(self, zip_path: Path, output_dir: Path, remove_zip: bool = True) -> bool:
        """Extract ZIP dengan progress callback."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                total_files = len(files)
                
                for i, file in enumerate(files):
                    zip_ref.extract(file, output_dir)
                    
                    # Progress callback setiap 10 file atau 10%
                    if i % max(1, total_files // 10) == 0:
                        progress = int((i / total_files) * 100)
                        self._notify_progress("extract", progress, 100, f"Ekstrak: {progress}%")
            
            if remove_zip:
                zip_path.unlink()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Extract error: {str(e)}")
            return False
