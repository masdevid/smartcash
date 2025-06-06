"""
File: smartcash/dataset/downloader/roboflow_client.py
Deskripsi: Optimized Roboflow client dengan one-liner methods dan enhanced performance
"""

import requests
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from smartcash.common.logger import get_logger

class RoboflowClient:
    """Optimized Roboflow client dengan one-liner methods dan caching."""
    
    def __init__(self, api_key: str, timeout: int = 30, retry_count: int = 3, logger=None):
        self.api_key, self.timeout, self.retry_count, self.logger = api_key, timeout, retry_count, logger or get_logger()
        self.base_url, self._progress_callback, self._metadata_cache = "https://api.roboflow.com", None, {}
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set callback dengan one-liner assignment"""
        self._progress_callback = callback
    
    def get_dataset_metadata(self, workspace: str, project: str, version: str, format: str = "yolov5pytorch") -> Dict[str, Any]:
        """Get metadata dengan one-liner caching dan optimized retry"""
        cache_key = f"{workspace}/{project}:{version}:{format}"
        
        # One-liner cache check
        if cache_key in self._metadata_cache:
            cached_result = self._metadata_cache[cache_key]
            self._notify_progress("metadata", 100, 100, "✅ Metadata dari cache")
            return cached_result
        
        url, params = f"{self.base_url}/{workspace}/{project}/{version}/{format}", {'api_key': self.api_key}
        self._notify_progress("metadata", 0, 100, "🌐 Mengambil metadata...")
        
        # Optimized retry loop dengan one-liner error handling
        for attempt in range(1, self.retry_count + 1):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                metadata = response.json()
                
                # One-liner validation dan cache storage
                ('export' not in metadata or 'link' not in metadata['export']) and self._raise_error("Response tidak lengkap")
                
                result = {
                    'status': 'success', 'data': metadata, 'download_url': metadata['export']['link'],
                    'size_mb': metadata.get('export', {}).get('size', 0), 'message': 'Metadata berhasil'
                }
                
                # One-liner caching dan logging
                self._metadata_cache[cache_key] = result
                self._notify_progress("metadata", 100, 100, "✅ Metadata diperoleh")
                'project' in metadata and self.logger.info(f"📊 Dataset: {len(metadata['project'].get('classes', []))} kelas, {metadata.get('version', {}).get('images', 0)} gambar")
                
                return result
                
            except requests.HTTPError as e:
                error_msg = self._handle_http_error(e.response.status_code, workspace, project, version)
                attempt == self.retry_count and self._return_error_result(error_msg)
                
            except requests.RequestException as e:
                if attempt < self.retry_count:
                    wait_time = attempt * 2
                    self.logger.warning(f"⚠️ Retry {attempt} dalam {wait_time}s: {str(e)}")
                    time.sleep(wait_time)
                else:
                    return self._return_error_result(f"Koneksi gagal: {str(e)}")
                    
            except Exception as e:
                return self._return_error_result(f"Error metadata: {str(e)}")
        
        return self._return_error_result('Gagal setelah retry')
    
    def download_dataset(self, download_url: str, output_path: Path, chunk_size: int = 8192) -> Dict[str, Any]:
        """Download dengan optimized progress tracking dan one-liner updates"""
        try:
            self._notify_progress("download", 0, 100, "📥 Memulai download...")
            
            response = requests.get(download_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            total_size, downloaded, last_progress = int(response.headers.get('content-length', 0)), 0, 0
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # One-liner progress update dengan threshold
                        total_size > 0 and (
                            progress := int((downloaded / total_size) * 100),
                            (progress - last_progress >= 5 or downloaded - (last_progress * total_size / 100) >= 1048576) and (
                                self._notify_progress("download", progress, 100, f"📥 {downloaded/1048576:.1f}/{total_size/1048576:.1f} MB"),
                                setattr(self, '_temp_last_progress', progress)  # Update last_progress
                            )
                        ) and (last_progress := progress)
            
            self._notify_progress("download", 100, 100, "✅ Download selesai")
            return {
                'status': 'success', 'file_path': str(output_path), 'size_bytes': downloaded,
                'size_mb': downloaded / 1048576, 'message': 'Download berhasil'
            }
            
        except Exception as e:
            return {'status': 'error', 'message': f"Download gagal: {str(e)}"}
    
    def validate_credentials(self, workspace: str, project: str) -> Dict[str, Any]:
        """One-liner credential validation dengan optimized error mapping"""
        try:
            response = requests.get(f"{self.base_url}/{workspace}", params={'api_key': self.api_key}, timeout=10)
            return {200: {'valid': True, 'message': 'Kredensial valid'},
                    401: {'valid': False, 'message': 'API key tidak valid'}, 
                    403: {'valid': False, 'message': 'Tidak memiliki akses'},
                    404: {'valid': False, 'message': f'Workspace "{workspace}" tidak ditemukan'}}.get(
                    response.status_code, {'valid': False, 'message': f'Error API: {response.status_code}'})
        except requests.RequestException as e:
            return {'valid': False, 'message': f'Koneksi gagal: {str(e)}'}
    
    def _handle_http_error(self, status_code: int, workspace: str, project: str, version: str) -> str:
        """One-liner HTTP error mapping"""
        return {400: "Request tidak valid", 401: "API key tidak valid", 403: "Tidak memiliki akses",
                404: f"Dataset {workspace}/{project}:{version} tidak ditemukan", 429: "Terlalu banyak request",
                500: "Server error"}.get(status_code, f"HTTP Error {status_code}")
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """One-liner progress notification dengan safe execution"""
        self._progress_callback and (lambda: self._progress_callback(step, current, total, message))() if True else None
    
    def _return_error_result(self, message: str) -> Dict[str, Any]:
        """One-liner error result creation"""
        return {'status': 'error', 'message': message}
    
    def _raise_error(self, message: str) -> None:
        """One-liner error raising"""
        raise ValueError(message)
    
    def clear_cache(self) -> None:
        """One-liner cache clearing"""
        self._metadata_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """One-liner cache info"""
        return {'cached_datasets': len(self._metadata_cache), 'cache_keys': list(self._metadata_cache.keys())}

# One-liner factory
def create_roboflow_client(api_key: str, logger=None) -> RoboflowClient:
    """Factory untuk optimized RoboflowClient"""
    return RoboflowClient(api_key, timeout=30, retry_count=3, logger=logger)