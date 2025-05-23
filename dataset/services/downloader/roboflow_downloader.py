"""
File: smartcash/dataset/services/downloader/roboflow_downloader.py
Deskripsi: Updated Roboflow downloader dengan progress callback support (no tqdm)
"""

import os, json, time, zipfile, requests
from pathlib import Path
from typing import Dict, Optional, Union, Callable

from smartcash.common.logger import get_logger

class RoboflowDownloader:
    """Roboflow downloader dengan progress callback support."""
    
    def __init__(self, logger=None, timeout: int = 30, chunk_size: int = 8192, retry_limit: int = 3):
        self.logger = logger or get_logger()
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.retry_limit = retry_limit
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set callback untuk progress updates."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def get_roboflow_metadata(self, workspace: str, project: str, version: str, api_key: str, format: str, temp_dir: Union[str, Path]) -> Dict:
        """Get metadata dengan progress callback."""
        metadata_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{format}?api_key={api_key}"
        
        self.logger.info("📋 Mendapatkan metadata dataset")
        self._notify_progress("metadata", 10, 100, "Menghubungi server Roboflow")
        
        try:
            response = requests.get(metadata_url, timeout=self.timeout)
            response.raise_for_status()
            metadata = response.json()
            
            if 'export' not in metadata or 'link' not in metadata['export']:
                raise ValueError("Format metadata tidak valid")
            
            # Save metadata
            temp_path = Path(temp_dir)
            temp_path.mkdir(parents=True, exist_ok=True)
            metadata_path = temp_path / f"{workspace}_{project}_{version}_metadata.json"
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._notify_progress("metadata", 100, 100, "Metadata berhasil diperoleh")
            
            # Log dataset info
            if 'project' in metadata and 'classes' in metadata['project']:
                classes = metadata['project']['classes']
                total_images = metadata.get('version', {}).get('images', 0)
                self.logger.info(f"📊 Dataset: {len(classes)} kelas, {total_images} gambar")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error metadata: {str(e)}")
            raise
    
    def process_roboflow_download(self, download_url: str, output_path: Union[str, Path], show_progress: bool = False) -> bool:
        """Download dan extract dengan progress callback (no tqdm)."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        zip_path = output_path / "dataset.zip"
        
        try:
            self.logger.info("📥 Memulai download dataset")
            self._notify_progress("download", 0, 100, "Memulai download")
            
            # Download
            if not self._download_with_callback(download_url, zip_path):
                return False
            
            # Verify ZIP
            file_size = zip_path.stat().st_size
            if file_size < 1000:
                self.logger.error(f"❌ File ZIP tidak valid: {file_size} bytes")
                return False
            
            # Extract
            self.logger.info(f"📦 Mengekstrak dataset ({file_size / (1024*1024):.2f} MB)")
            self._notify_progress("extract", 0, 100, "Mengekstrak dataset")
            
            if not self._extract_with_callback(zip_path, output_path, remove_zip=True):
                return False
            
            # Verify results
            extracted_files = list(output_path.glob('**/*'))
            if not extracted_files:
                self.logger.error("❌ Tidak ada file diekstrak")
                return False
            
            self.logger.success(f"✅ Download selesai: {len(extracted_files)} file")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Process gagal: {str(e)}")
            if zip_path.exists():
                zip_path.unlink()
            return False
    
    def _download_with_callback(self, url: str, output_path: Path) -> bool:
        """Download dengan progress callback."""
        for attempt in range(1, self.retry_limit + 1):
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
                            if total_size > 0 and downloaded % max(1, total_size // 20) < self.chunk_size:
                                progress = int((downloaded / total_size) * 100)
                                self._notify_progress("download", progress, 100, f"Download: {progress}%")
                
                self._notify_progress("download", 100, 100, "Download selesai")
                return True
                
            except Exception as e:
                if attempt < self.retry_limit:
                    wait_time = attempt
                    self.logger.warning(f"⚠️ Download error (attempt {attempt}): {str(e)}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"❌ Download gagal: {str(e)}")
                    return False
        return False
    
    def _extract_with_callback(self, zip_path: Path, output_dir: Path, remove_zip: bool = True) -> bool:
        """Extract dengan progress callback."""
        try:
            if not zipfile.is_zipfile(zip_path):
                self.logger.error(f"❌ File bukan ZIP: {zip_path}")
                return False
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                total_files = len(files)
                
                if not files:
                    self.logger.error("❌ ZIP kosong")
                    return False
                
                # Extract dengan progress
                for i, file in enumerate(files):
                    zip_ref.extract(file, output_dir)
                    
                    # Progress callback setiap 10%
                    if i % max(1, total_files // 10) == 0:
                        progress = int((i / total_files) * 100)
                        self._notify_progress("extract", progress, 100, f"Ekstrak: {progress}%")
                
                self._notify_progress("extract", 100, 100, "Ekstraksi selesai")
            
            if remove_zip:
                zip_path.unlink()
                self.logger.info("🗑️ ZIP file dihapus")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Extract gagal: {str(e)}")
            return False