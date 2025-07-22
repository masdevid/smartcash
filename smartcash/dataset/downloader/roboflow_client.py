"""
File: smartcash/dataset/downloader/roboflow_client.py
Deskripsi: Simplified Roboflow client menggunakan base components
"""

from typing import Dict, Any
from pathlib import Path
from smartcash.dataset.downloader.base import BaseDownloaderComponent, RequestsHelper


class RoboflowClient(BaseDownloaderComponent):
    """Simplified Roboflow client dengan shared components"""
    
    def __init__(self, api_key: str, logger=None):
        super().__init__(logger)
        self.api_key = api_key
        self.base_url = "https://api.roboflow.com"
        self._metadata_cache = {}
    
    def get_dataset_metadata(self, workspace: str, project: str, version: str, 
                           format: str = "yolov5pytorch") -> Dict[str, Any]:
        """Get dataset metadata dengan caching"""
        cache_key = f"{workspace}/{project}:{version}:{format}"
        
        if cache_key in self._metadata_cache:
            self._notify_progress("metadata", 100, 100, "✅ Metadata dari cache")
            return self._metadata_cache[cache_key]
        
        url = f"{self.base_url}/{workspace}/{project}/{version}/{format}"
        params = {'api_key': self.api_key}
        
        self._notify_progress("metadata", 0, 100, "🌐 Mengambil metadata...")
        
        try:
            response = RequestsHelper.get_with_retry(url, params, timeout=30, retry_count=3)
            metadata = response.json()
            
            if 'export' not in metadata or 'link' not in metadata['export']:
                return self._create_error_result("Response tidak lengkap")
            
            result = self._create_success_result(
                data=metadata,
                download_url=metadata['export']['link'],
                size_mb=metadata.get('export', {}).get('size', 0)
            )
            
            self._metadata_cache[cache_key] = result
            self._notify_progress("metadata", 100, 100, "✅ Metadata diperoleh")
            
            return result
            
        except Exception as e:
            return self._create_error_result(f"Metadata failed: {str(e)}")
    
    def download_dataset(self, download_url: str, output_path: Path) -> Dict[str, Any]:
        """Download dataset menggunakan shared helper"""
        self._notify_progress("download", 0, 100, "📥 Memulai download...")
        
        result = RequestsHelper.download_with_progress(
            download_url, output_path, self._notify_progress
        )
        
        if result['status'] == 'success':
            self._notify_progress("download", 100, 100, "✅ Download selesai")
        
        return result
    
    def validate_credentials(self, workspace: str) -> Dict[str, Any]:
        """Validate API credentials"""
        try:
            url = f"{self.base_url}/{workspace}"
            params = {'api_key': self.api_key}
            
            response = RequestsHelper.get_with_retry(url, params, timeout=10, retry_count=1)
            return {'valid': True, 'message': 'Kredensial valid'}
            
        except Exception as e:
            return {'valid': False, 'message': f'Kredensial tidak valid: {str(e)}'}


def create_roboflow_client(api_key: str, logger=None) -> RoboflowClient:
    """Factory untuk RoboflowClient"""
    return RoboflowClient(api_key, logger)