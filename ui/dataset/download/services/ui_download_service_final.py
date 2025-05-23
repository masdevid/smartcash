"""
File: smartcash/ui/dataset/download/services/ui_download_service_final.py
Deskripsi: Final UI download service dengan progress callback integration
"""

import time
from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.services.downloader.ui_roboflow_downloader import UIRoboflowDownloader
from smartcash.ui.dataset.download.services.progress_bridge import ProgressBridge

class UIDownloadServiceFinal:
    """Final download service dengan progress integration."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.progress_bridge = ProgressBridge(
            observer_manager=ui_components.get('observer_manager'),
            namespace="download"
        )
        
        # Create downloader dengan progress callback
        self.downloader = UIRoboflowDownloader(logger=self.logger)
        self.downloader.set_progress_callback(self._progress_callback)
    
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download dengan observer progress tracking."""
        start_time = time.time()
        
        try:
            self.progress_bridge.notify_start("Mempersiapkan download", 4)
            
            # Validate
            self._validate_params(params)
            self.progress_bridge.notify_progress(10, 100, "Parameter valid", "validate", 1, 4)
            
            # Get metadata
            metadata = self._get_metadata(params)
            self.progress_bridge.notify_progress(25, 100, "Metadata diperoleh", "metadata", 2, 4)
            
            # Download & extract
            output_path = Path(params['output_dir'])
            download_url = metadata['export']['link']
            
            success = self.downloader.download_and_extract(download_url, output_path)
            
            if not success:
                raise Exception("Download/extract gagal")
            
            # Finalize
            self.progress_bridge.notify_progress(95, 100, "Memverifikasi hasil", "verify", 4, 4)
            stats = self._get_stats(output_path)
            
            duration = time.time() - start_time
            self.progress_bridge.notify_complete(f"Download selesai: {stats.get('total_images', 0)} gambar", duration)
            
            return {
                'status': 'success',
                'output_dir': str(output_path),
                'stats': stats,
                'duration': duration
            }
            
        except Exception as e:
            self.progress_bridge.notify_error(f"Download gagal: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Callback dari downloader ke progress bridge."""
        # Map step ke overall progress ranges
        step_ranges = {
            'download': (25, 75),  # 25-75% untuk download
            'extract': (75, 95),   # 75-95% untuk extract
        }
        
        if step in step_ranges:
            start_pct, end_pct = step_ranges[step]
            progress_pct = start_pct + (current / total) * (end_pct - start_pct)
            
            step_mapping = {
                'download': ('Mendownload dataset', 3, 4),
                'extract': ('Mengekstrak dataset', 4, 4)
            }
            
            step_info = step_mapping.get(step, (message, 3, 4))
            step_name, current_step, total_steps = step_info
            
            self.progress_bridge.notify_progress(
                int(progress_pct), 100, message,
                step_name, current_step, total_steps
            )
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validasi parameter download."""
        required = ['workspace', 'project', 'version', 'api_key', 'output_dir']
        missing = [p for p in required if not params.get(p)]
        
        if missing:
            raise ValueError(f"Parameter tidak lengkap: {', '.join(missing)}")
        
        Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def _get_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get dataset metadata dari Roboflow."""
        import requests
        
        metadata_url = (f"https://api.roboflow.com/{params['workspace']}"
                       f"/{params['project']}/{params['version']}"
                       f"/yolov5pytorch?api_key={params['api_key']}")
        
        try:
            response = requests.get(metadata_url, timeout=30)
            response.raise_for_status()
            metadata = response.json()
            
            if 'export' not in metadata or 'link' not in metadata['export']:
                raise ValueError("Invalid metadata format")
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Gagal mendapatkan metadata: {str(e)}")
    
    def _get_stats(self, output_path: Path) -> Dict[str, Any]:
        """Get basic stats dari dataset yang didownload."""
        stats = {'total_images': 0, 'total_labels': 0}
        
        try:
            # Count files in common locations
            for split in ['train', 'valid', 'test']:
                images_dir = output_path / split / 'images'
                if images_dir.exists():
                    img_count = len(list(images_dir.glob('*.*')))
                    stats['total_images'] += img_count
                    stats[f'{split}_images'] = img_count
        except Exception:
            pass
        
        return stats