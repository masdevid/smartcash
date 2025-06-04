"""
File: smartcash/dataset/downloader/roboflow_client.py
Deskripsi: Fixed Roboflow client dengan proper constructor dan response handling
"""

import requests
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from smartcash.common.logger import get_logger

class RoboflowClient:
    """Client untuk API Roboflow dengan progress callback support."""
    
    def __init__(self, api_key: str, timeout: int = 30, retry_count: int = 3, logger=None):
        self.api_key = api_key
        self.timeout = timeout
        self.retry_count = retry_count
        self.logger = logger or get_logger()
        self.base_url = "https://api.roboflow.com"
        self._progress_callback: Optional[Callable] = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set callback untuk progress updates."""
        self._progress_callback = callback
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress dengan error handling."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def get_dataset_metadata(self, workspace: str, project: str, version: str, format: str = "yolov5pytorch") -> Dict[str, Any]:
        """Get metadata dataset dari Roboflow API dengan proper response handling."""
        url = f"{self.base_url}/{workspace}/{project}/{version}/{format}"
        params = {'api_key': self.api_key}
        
        self._notify_progress("metadata", 0, 100, "Mengambil metadata dataset")
        
        for attempt in range(1, self.retry_count + 1):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                metadata = response.json()
                
                # Validate response structure
                if 'export' not in metadata or 'link' not in metadata['export']:
                    raise ValueError("Response metadata tidak lengkap")
                
                self._notify_progress("metadata", 100, 100, "Metadata berhasil diperoleh")
                
                # Log dataset info
                if 'project' in metadata:
                    classes_count = len(metadata['project'].get('classes', []))
                    images_count = metadata.get('version', {}).get('images', 0)
                    self.logger.info(f"📊 Dataset: {classes_count} kelas, {images_count} gambar")
                
                return {
                    'status': 'success',
                    'data': metadata,
                    'download_url': metadata['export']['link'],
                    'size_mb': metadata.get('export', {}).get('size', 0),
                    'message': 'Metadata berhasil diperoleh'
                }
                
            except requests.HTTPError as e:
                error_msg = self._handle_http_error(e.response.status_code, workspace, project, version)
                if attempt == self.retry_count:
                    return {'status': 'error', 'message': error_msg}
                
            except requests.RequestException as e:
                if attempt < self.retry_count:
                    wait_time = attempt * 2
                    self.logger.warning(f"⚠️ Attempt {attempt} gagal, retry dalam {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    return {'status': 'error', 'message': f"Koneksi gagal: {str(e)}"}
                    
            except Exception as e:
                return {'status': 'error', 'message': f"Error metadata: {str(e)}"}
        
        return {'status': 'error', 'message': 'Gagal mendapatkan metadata setelah beberapa percobaan'}
    
    def download_dataset(self, download_url: str, output_path: Path, chunk_size: int = 8192) -> Dict[str, Any]:
        """Download dataset dengan progress tracking."""
        try:
            self._notify_progress("download", 0, 100, "Memulai download dataset")
            
            response = requests.get(download_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            last_progress = 0
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress setiap 5% atau 1MB
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            if progress - last_progress >= 5 or downloaded - (last_progress * total_size / 100) >= 1024 * 1024:
                                size_mb = downloaded / (1024 * 1024)
                                total_mb = total_size / (1024 * 1024)
                                self._notify_progress("download", progress, 100, f"Download: {size_mb:.1f}/{total_mb:.1f} MB")
                                last_progress = progress
            
            self._notify_progress("download", 100, 100, "Download selesai")
            
            return {
                'status': 'success',
                'file_path': str(output_path),
                'size_bytes': downloaded,
                'size_mb': downloaded / (1024 * 1024),
                'message': 'Download berhasil'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Download gagal: {str(e)}"}
    
    def validate_credentials(self, workspace: str, project: str) -> Dict[str, Any]:
        """Validate API credentials dan akses ke workspace/project."""
        url = f"{self.base_url}/{workspace}"
        params = {'api_key': self.api_key}
        
        try:
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                return {'valid': True, 'message': 'Kredensial valid'}
            elif response.status_code in [401, 403]:
                return {'valid': False, 'message': 'API key tidak valid atau tidak memiliki akses'}
            elif response.status_code == 404:
                return {'valid': False, 'message': f'Workspace "{workspace}" tidak ditemukan'}
            else:
                return {'valid': False, 'message': f'Error API: {response.status_code}'}
                
        except requests.RequestException as e:
            return {'valid': False, 'message': f'Koneksi gagal: {str(e)}'}
    
    def _handle_http_error(self, status_code: int, workspace: str, project: str, version: str) -> str:
        """Handle HTTP error dengan pesan yang informatif."""
        error_messages = {
            400: "Request tidak valid - periksa parameter",
            401: "API key tidak valid",
            403: "Tidak memiliki akses ke dataset",
            404: f"Dataset {workspace}/{project}:{version} tidak ditemukan",
            429: "Terlalu banyak request - coba lagi nanti",
            500: "Server error - coba lagi nanti"
        }
        
        return error_messages.get(status_code, f"HTTP Error {status_code}")

# Factory function
def create_roboflow_client(api_key: str, logger=None) -> RoboflowClient:
    """Factory untuk create RoboflowClient dengan default settings."""
    return RoboflowClient(api_key, timeout=30, retry_count=3, logger=logger)