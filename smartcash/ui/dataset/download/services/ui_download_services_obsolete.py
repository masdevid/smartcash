"""
File: smartcash/ui/dataset/download/services/ui_download_service.py
Deskripsi: Wrapper service untuk download yang terintegrasi langsung dengan UI tanpa threading
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
from smartcash.dataset.services.downloader.download_service import DownloadService
from smartcash.common.exceptions import DatasetError

class UIDownloadService:
    """Service wrapper untuk download dengan integrasi UI langsung."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi UI download service.
        
        Args:
            ui_components: Dictionary komponen UI untuk progress tracking
        """
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        self._download_service = None
    
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Download dataset dengan progress tracking ke UI.
        
        Args:
            params: Parameter download (workspace, project, version, api_key, output_dir)
            
        Returns:
            Dictionary hasil download
        """
        start_time = time.time()
        
        try:
            self._update_progress(0, "Mempersiapkan download...")
            
            # Validasi parameter
            self._validate_params(params)
            
            # Create download service
            self._download_service = DownloadService(
                output_dir=params['output_dir'],
                config={
                    'data': {
                        'roboflow': {
                            'workspace': params['workspace'],
                            'project': params['project'],
                            'version': params['version'],
                            'api_key': params['api_key']
                        }
                    }
                },
                logger=self.logger
            )
            
            self._update_progress(10, "Menghubungi server Roboflow...")
            
            # Execute download dengan callback progress
            result = self._execute_download_with_progress(params)
            
            # Final progress
            self._update_progress(100, "Download selesai")
            
            duration = time.time() - start_time
            result['duration'] = duration
            
            if self.logger:
                self.logger.success(f"✅ Download selesai dalam {duration:.1f}s")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Download gagal: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Validasi parameter download."""
        required = ['workspace', 'project', 'version', 'api_key', 'output_dir']
        missing = [p for p in required if not params.get(p)]
        
        if missing:
            raise DatasetError(f"Parameter tidak lengkap: {', '.join(missing)}")
        
        # Buat output directory jika belum ada
        output_path = Path(params['output_dir'])
        output_path.mkdir(parents=True, exist_ok=True)
    
    def _execute_download_with_progress(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute download dengan progress manual tracking."""
        
        # Step 1: Get metadata
        self._update_step_progress(1, 5, 20, "Mendapatkan metadata dataset...")
        
        try:
            # Direct call tanpa observer complexity
            result = self._download_service.download_from_roboflow(
                api_key=params['api_key'],
                workspace=params['workspace'],
                project=params['project'],
                version=params['version'],
                output_dir=params['output_dir'],
                show_progress=False,  # Kita handle progress sendiri
                verify_integrity=True
            )
            
            # Step 2: Download progress simulation
            self._update_step_progress(2, 5, 40, "Mendownload dataset...")
            time.sleep(0.5)  # Small delay untuk UI feedback
            
            # Step 3: Extract progress simulation  
            self._update_step_progress(3, 5, 65, "Mengekstrak dataset...")
            time.sleep(0.3)
            
            # Step 4: Verify progress simulation
            self._update_step_progress(4, 5, 85, "Memverifikasi dataset...")
            time.sleep(0.2)
            
            # Step 5: Finalize
            self._update_step_progress(5, 5, 100, "Menyelesaikan download...")
            
            return result
            
        except Exception as e:
            raise DatasetError(f"Download gagal: {str(e)}")
    
    def _update_progress(self, value: int, message: str) -> None:
        """Update overall progress."""
        from smartcash.ui.dataset.download.utils.progress_updater import update_progress
        update_progress(self.ui_components, value, message)
    
    def _update_step_progress(self, step: int, total_steps: int, progress: int, message: str) -> None:
        """Update step progress."""
        from smartcash.ui.dataset.download.utils.progress_updater import update_step_progress
        update_step_progress(self.ui_components, step, total_steps, progress, message)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._download_service:
            self._download_service.cleanup()
            self._download_service = None