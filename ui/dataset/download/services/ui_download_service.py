"""
File: smartcash/ui/dataset/download/services/ui_download_service.py
Deskripsi: Updated download service dengan progress bridge (no tqdm)
"""

import time
from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.ui.dataset.download.services.progress_bridge import ProgressBridge

class UIDownloadService:
    """Download service dengan progress bridge ke UI."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self.progress_bridge = ProgressBridge(
            observer_manager=ui_components.get('observer_manager'),
            namespace="download"
        )
    
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download dengan progress tracking via observer."""
        start_time = time.time()
        
        try:
            self.progress_bridge.notify_start("Mempersiapkan download dataset", 5)
            
            # Validasi
            self._validate_params(params)
            self.progress_bridge.notify_progress(10, 100, "Validasi parameter berhasil")
            
            # Create service
            download_service = DownloadService(
                output_dir=params['output_dir'],
                config={'data': {'roboflow': params}},
                logger=self.logger
            )
            
            # Set progress callback ke service
            download_service._progress_callback = self._progress_callback
            
            self.progress_bridge.notify_progress(20, 100, "Menghubungi server Roboflow", "connect", 1, 5)
            
            # Execute download
            result = download_service.download_from_roboflow(
                api_key=params['api_key'],
                workspace=params['workspace'],
                project=params['project'],
                version=params['version'],
                output_dir=params['output_dir'],
                show_progress=False,  # Disable tqdm
                verify_integrity=True
            )
            
            duration = time.time() - start_time
            self.progress_bridge.notify_complete(f"Download selesai: {result.get('stats', {}).get('total_images', 0)} gambar", duration)
            
            return result
            
        except Exception as e:
            self.progress_bridge.notify_error(f"Download gagal: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validasi parameter."""
        required = ['workspace', 'project', 'version', 'api_key', 'output_dir']
        missing = [p for p in required if not params.get(p)]
        
        if missing:
            raise ValueError(f"Parameter tidak lengkap: {', '.join(missing)}")
        
        Path(params['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def _progress_callback(self, step: str, progress: int, total: int, message: str) -> None:
        """Callback untuk progress dari service layer."""
        step_mapping = {
            'metadata': ('Mendapatkan metadata', 2, 5),
            'download': ('Mendownload dataset', 3, 5), 
            'extract': ('Mengekstrak dataset', 4, 5),
            'verify': ('Memverifikasi dataset', 5, 5)
        }
        
        step_info = step_mapping.get(step, (message, 3, 5))
        step_message, current_step, total_steps = step_info
        
        # Convert ke percentage untuk overall progress
        base_progress = {
            'metadata': 30,
            'download': 50, 
            'extract': 75,
            'verify': 90
        }.get(step, 50)
        
        # Calculate actual progress
        step_progress = base_progress + (progress / total) * 15 if total > 0 else base_progress
        
        self.progress_bridge.notify_progress(
            int(step_progress), 100, message,
            step_message, current_step, total_steps
        )