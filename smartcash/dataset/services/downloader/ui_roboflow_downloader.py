"""
File: smartcash/dataset/services/downloader/ui_roboflow_downloader.py
Deskripsi: Enhanced Roboflow downloader dengan accurate progress tracking dan robust error handling
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
    """Enhanced Roboflow downloader dengan accurate progress tracking dan comprehensive error handling."""
    
    def __init__(self, logger=None, timeout: int = 30, chunk_size: int = 8192, retry_limit: int = 3):
        super().__init__()
        self.logger = logger or get_logger()
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.retry_limit = retry_limit
    
    def download_and_extract(self, download_url: str, output_path: Path, show_progress: bool = False) -> bool:
        """Enhanced download dan extract dengan comprehensive error handling dan retry logic."""
        zip_path = output_path / "dataset.zip"
        
        try:
            # Ensure output directory exists
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Download dengan retry logic
            self._notify_step_start("download", "Memulai download dataset...")
            
            if not self._download_file_with_retry(download_url, zip_path):
                return False
            
            # Verify downloaded file
            if not self._verify_downloaded_file(zip_path):
                return False
            
            # Step 2: Extract dengan validation
            self._notify_step_start("extract", "Mengekstrak dataset...")
            
            if not self._extract_zip_with_validation(zip_path, output_path, remove_zip=True):
                return False
            
            # Step 3: Post-extraction validation
            if not self._validate_extraction_results(output_path):
                return False
            
            self._notify_step_complete("extract", "Download dan ekstraksi berhasil")
            
            if self.logger:
                self.logger.success("✅ Dataset berhasil didownload dan diekstraksi")
            
            return True
            
        except Exception as e:
            error_msg = f"Download/extract gagal: {str(e)}"
            self.logger.error(f"❌ {error_msg}")
            self._notify_step_error("download", error_msg)
            
            # Cleanup pada error
            if zip_path.exists():
                zip_path.unlink()
            
            return False
    
    def _download_file_with_retry(self, url: str, output_path: Path) -> bool:
        """Download file dengan retry logic dan detailed progress tracking."""
        
        for attempt in range(1, self.retry_limit + 1):
            try:
                if self.logger and attempt > 1:
                    self.logger.info(f"🔄 Download attempt {attempt}/{self.retry_limit}")
                
                response = requests.get(url, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                last_progress_update = 0
                
                if self.logger:
                    size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
                    self.logger.info(f"📥 Downloading dataset ({size_mb:.2f} MB)...")
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Smart progress updates - setiap 5% atau setiap 1MB
                            if total_size > 0:
                                progress = int((downloaded / total_size) * 100)
                                
                                # Update setiap 5% atau setiap 1MB yang didownload
                                if (progress - last_progress_update >= 5 or 
                                    downloaded - (last_progress_update * total_size / 100) >= 1024 * 1024):
                                    
                                    downloaded_mb = downloaded / (1024 * 1024)
                                    total_mb = total_size / (1024 * 1024)
                                    
                                    self._notify_progress("download", progress, 100, 
                                                        f"Downloaded: {downloaded_mb:.1f}/{total_mb:.1f} MB ({progress}%)")
                                    last_progress_update = progress
                
                self._notify_progress("download", 100, 100, "Download selesai")
                return True
                
            except requests.RequestException as e:
                if attempt < self.retry_limit:
                    wait_time = attempt * 2  # Exponential backoff
                    if self.logger:
                        self.logger.warning(f"⚠️ Download error (attempt {attempt}): {str(e)}")
                        self.logger.info(f"🔄 Retry dalam {wait_time} detik...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"❌ Download gagal setelah {self.retry_limit} attempts: {str(e)}")
                    return False
            except Exception as e:
                self.logger.error(f"❌ Unexpected download error: {str(e)}")
                return False
        
        return False
    
    def _verify_downloaded_file(self, zip_path: Path) -> bool:
        """Verify downloaded ZIP file integrity."""
        try:
            # Check file exists dan tidak kosong
            if not zip_path.exists():
                self.logger.error("❌ File ZIP tidak ditemukan setelah download")
                return False
            
            file_size = zip_path.stat().st_size
            if file_size < 1000:  # Minimum 1KB
                self.logger.error(f"❌ File ZIP terlalu kecil: {file_size} bytes")
                return False
            
            # Verify ZIP file integrity
            if not zipfile.is_zipfile(zip_path):
                self.logger.error("❌ File yang didownload bukan ZIP yang valid")
                return False
            
            # Test ZIP dapat dibuka
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.infolist()
                if not file_list:
                    self.logger.error("❌ ZIP file kosong")
                    return False
            
            if self.logger:
                size_mb = file_size / (1024 * 1024)
                self.logger.info(f"✅ ZIP file valid ({size_mb:.2f} MB)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error verifying ZIP file: {str(e)}")
            return False
    
    def _extract_zip_with_validation(self, zip_path: Path, output_dir: Path, remove_zip: bool = True) -> bool:
        """Extract ZIP dengan validation dan progress tracking yang akurat."""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                total_files = len(files)
                extracted_count = 0
                last_progress_update = 0
                
                if self.logger:
                    self.logger.info(f"📦 Extracting {total_files} files...")
                
                for i, file_info in enumerate(files):
                    try:
                        # Extract file
                        zip_ref.extract(file_info, output_dir)
                        extracted_count += 1
                        
                        # Smart progress updates
                        progress = int((i + 1) / total_files * 100)
                        
                        # Update setiap 10% atau setiap 50 files
                        if (progress - last_progress_update >= 10 or 
                            i - (last_progress_update * total_files / 100) >= 50):
                            
                            self._notify_progress("extract", progress, 100, 
                                                f"Extracted: {extracted_count}/{total_files} files ({progress}%)")
                            last_progress_update = progress
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error extracting {file_info.filename}: {str(e)}")
                        # Continue dengan file lainnya
                        continue
                
                self._notify_progress("extract", 100, 100, f"Extraction complete: {extracted_count}/{total_files} files")
                
                # Log hasil extraction
                if self.logger:
                    success_rate = (extracted_count / total_files) * 100 if total_files > 0 else 0
                    self.logger.info(f"📊 Extraction: {extracted_count}/{total_files} files ({success_rate:.1f}%)")
                
                # Remove ZIP jika diminta dan extraction berhasil
                if remove_zip and extracted_count > 0:
                    zip_path.unlink()
                    if self.logger:
                        self.logger.info("🗑️ ZIP file removed after extraction")
                
                # Consider successful jika minimal 90% files ter-extract
                return success_rate >= 90
                
        except zipfile.BadZipFile:
            self.logger.error("❌ ZIP file corrupted atau invalid")
            return False
        except Exception as e:
            self.logger.error(f"❌ Extraction error: {str(e)}")
            return False
    
    def _validate_extraction_results(self, output_dir: Path) -> bool:
        """Validate hasil extraction untuk memastikan dataset structure yang benar."""
        try:
            # Check basic structure
            expected_dirs = ['train', 'valid', 'test']
            found_dirs = []
            
            for dir_name in expected_dirs:
                dir_path = output_dir / dir_name
                if dir_path.exists() and dir_path.is_dir():
                    found_dirs.append(dir_name)
                    
                    # Check untuk images dan labels subdirectories
                    images_dir = dir_path / 'images'
                    labels_dir = dir_path / 'labels'
                    
                    if images_dir.exists() and labels_dir.exists():
                        # Count files
                        image_count = len(list(images_dir.glob('*.*')))
                        label_count = len(list(labels_dir.glob('*.txt')))
                        
                        if self.logger:
                            self.logger.info(f"📁 {dir_name}: {image_count} images, {label_count} labels")
            
            # Minimal harus ada train directory dengan isi
            if 'train' not in found_dirs:
                self.logger.error("❌ Train directory tidak ditemukan")
                return False
            
            # Check untuk file penting lainnya
            data_yaml = output_dir / 'data.yaml'
            if data_yaml.exists():
                if self.logger:
                    self.logger.info("✅ Dataset configuration file found")
            
            if self.logger:
                self.logger.info(f"✅ Dataset structure valid: {', '.join(found_dirs)} directories found")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error validating extraction: {str(e)}")
            return False
    
    def get_download_info(self) -> Dict[str, any]:
        """Get informasi tentang downloader untuk debugging."""
        return {
            'timeout': self.timeout,
            'chunk_size': self.chunk_size,
            'retry_limit': self.retry_limit,
            'has_progress_callback': self._progress_callback is not None,
            'current_step': self._current_step
        }