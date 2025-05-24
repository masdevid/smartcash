"""
File: smartcash/ui/dataset/download/services/ui_download_service.py
Deskripsi: Download service tanpa step validasi dataset setelah download
"""

import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.services.downloader.ui_roboflow_downloader import UIRoboflowDownloader
from smartcash.ui.dataset.download.services.progress_bridge import ProgressBridge
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager

class UIDownloadService:
    """Download service tanpa validasi dataset otomatis - gunakan check button."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        
        # Setup log suppression untuk backend services
        self._setup_backend_log_suppression()
        
        # Environment manager untuk path resolution
        self.env_manager = get_environment_manager()
        self.paths = get_paths_for_environment(
            is_colab=self.env_manager.is_colab,
            is_drive_mounted=self.env_manager.is_drive_mounted
        )
        
        # Enhanced progress bridge dengan dual tracking
        self.progress_bridge = ProgressBridge(
            observer_manager=ui_components.get('observer_manager'),
            namespace="download"
        )
        self.progress_bridge.set_ui_components_reference(ui_components)
        
        # Create downloader dengan progress callback
        self.downloader = UIRoboflowDownloader(logger=self.logger)
        self.downloader.set_progress_callback(self._downloader_progress_callback)
        
        # Dataset organizer untuk memindahkan ke struktur final
        self.organizer = DatasetOrganizer(logger=self.logger)
        self.organizer.set_progress_callback(self._organizer_progress_callback)
    
    def _setup_backend_log_suppression(self):
        """Suppress logs dari backend services."""
        backend_loggers = [
            'requests', 'urllib3', 'http.client', 'requests.packages.urllib3',
            'smartcash.dataset.services', 'smartcash.common', 'tensorflow', 
            'torch', 'PIL', 'matplotlib', 'zipfile'
        ]
        
        for logger_name in backend_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL)
            logger.propagate = False
            logger.handlers.clear()
        
        if not hasattr(self, '_original_stdout'):
            self._original_stdout = sys.stdout
    
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download tanpa step validasi otomatis."""
        start_time = time.time()
        
        # Define steps tanpa validasi
        steps = self._define_process_steps(params)
        self.progress_bridge.define_steps(steps)
        
        try:
            # 🚀 Initialize
            self.progress_bridge.notify_start("Memulai download dan organisasi dataset")
            
            # Step 1: Validate parameters
            self.progress_bridge.notify_step_start("validate", "Memvalidasi parameter download")
            self._validate_params(params)
            self.progress_bridge.notify_step_complete("Parameter berhasil divalidasi")
            
            # Step 2: Get metadata
            self.progress_bridge.notify_step_start("metadata", "Mendapatkan informasi dataset dari Roboflow")
            metadata = self._get_metadata(params)
            self.progress_bridge.notify_step_complete("Metadata dataset berhasil didapatkan")
            
            # Step 3: Download ke downloads folder
            self.progress_bridge.notify_step_start("download", "Mendownload dataset dari Roboflow")
            download_path = Path(self.paths['downloads']) / f"{params['workspace']}_{params['project']}_{params['version']}"
            download_path.mkdir(parents=True, exist_ok=True)
            
            download_url = metadata['export']['link']
            
            # Suppress console output selama download
            with self._suppress_console_output():
                if not self.downloader.download_and_extract(download_url, download_path):
                    raise Exception("Download atau ekstraksi gagal")
            
            self.progress_bridge.notify_step_complete("Dataset berhasil didownload")
            
            # Step 4: Organize dataset (pindah ke struktur final)
            self.progress_bridge.notify_step_start("organize", "Mengorganisir dataset ke struktur final")
            
            with self._suppress_console_output():
                organize_result = self.organizer.organize_dataset(str(download_path), remove_source=True)
            
            if organize_result['status'] != 'success':
                raise Exception(f"Gagal mengorganisir dataset: {organize_result.get('message')}")
            
            self.progress_bridge.notify_step_complete("Dataset berhasil diorganisir")
            
            # 🎉 Complete (tanpa step validasi)
            duration = time.time() - start_time
            total_images = organize_result.get('total_images', 0)
            success_message = f"Download dan organisasi selesai: {total_images} gambar dalam {duration:.1f}s"
            
            self.progress_bridge.notify_complete(success_message, duration)
            
            if self.logger:
                self.logger.success(f"✅ {success_message}")
                self.logger.info("🔍 Gunakan tombol 'Check Dataset' untuk memverifikasi hasil")
                self._log_final_structure(organize_result)
            
            return {
                'status': 'success',
                'output_dir': self.paths['data_root'],
                'download_path': str(download_path),
                'stats': organize_result,
                'duration': duration,
                'drive_storage': self.env_manager.is_drive_mounted,
                'organized': True,
                'verification_note': 'Gunakan Check Dataset untuk verifikasi hasil'
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.progress_bridge.notify_error(f"Proses gagal: {error_msg}")
            
            if self.logger:
                self.logger.error(f"❌ Download error setelah {duration:.1f}s: {error_msg}")
            
            return {'status': 'error', 'message': error_msg, 'duration': duration}
    
    def _suppress_console_output(self):
        """Context manager untuk suppress console output dari backend."""
        class SuppressOutput:
            def __enter__(self):
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr
                sys.stdout = open('/dev/null', 'w') if hasattr(sys.stdout, 'fileno') else sys.stdout
                sys.stderr = open('/dev/null', 'w') if hasattr(sys.stderr, 'fileno') else sys.stderr
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                try:
                    if sys.stdout != self._original_stdout and hasattr(sys.stdout, 'close'):
                        sys.stdout.close()
                    if sys.stderr != self._original_stderr and hasattr(sys.stderr, 'close'):
                        sys.stderr.close()
                except:
                    pass
                sys.stdout = self._original_stdout
                sys.stderr = self._original_stderr
        
        return SuppressOutput()
    
    def _define_process_steps(self, params: Dict[str, Any]) -> list:
        """Define steps tanpa validasi dataset."""
        steps = [
            {'name': 'validate', 'weight': 10, 'description': 'Validasi Parameter'},
            {'name': 'metadata', 'weight': 15, 'description': 'Ambil Metadata'},
            {'name': 'download', 'weight': 50, 'description': 'Download Dataset'},
            {'name': 'organize', 'weight': 25, 'description': 'Organisir Dataset'}
        ]
        
        return steps
    
    def _downloader_progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Callback dari downloader untuk update step progress."""
        if step in ['download', 'extract']:
            step_progress = int((current / total) * 100) if total > 0 else 0
            self.progress_bridge.notify_step_progress(step_progress, message)
    
    def _organizer_progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Callback dari organizer untuk update step progress."""
        if step == 'organize':
            self.progress_bridge.notify_step_progress(current, message)
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Enhanced parameter validation."""
        required = ['workspace', 'project', 'version', 'api_key']
        missing = [p for p in required if not params.get(p)]
        
        if missing:
            raise ValueError(f"Parameter tidak lengkap: {', '.join(missing)}")
        
        api_key = params['api_key']
        if len(api_key) < 10:
            raise ValueError("API key terlalu pendek, periksa kembali")
        
        downloads_path = Path(self.paths['downloads'])
        downloads_path.mkdir(parents=True, exist_ok=True)
        
        if self.logger:
            self.logger.debug("✅ Parameter validation berhasil")
    
    def _get_metadata(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get dataset metadata dengan enhanced error handling."""
        import requests
        
        metadata_url = (f"https://api.roboflow.com/{params['workspace']}"
                       f"/{params['project']}/{params['version']}"
                       f"/yolov5pytorch?api_key={params['api_key']}")
        
        try:
            if self.logger:
                self.logger.debug(f"🌐 Requesting metadata: {params['workspace']}/{params['project']}:{params['version']}")
            
            response = requests.get(metadata_url, timeout=30)
            response.raise_for_status()
            metadata = response.json()
            
            if 'export' not in metadata:
                raise ValueError("Response tidak mengandung export data")
            
            if 'link' not in metadata['export']:
                raise ValueError("Export link tidak ditemukan dalam response")
            
            if self.logger and 'project' in metadata:
                project_info = metadata['project']
                classes_count = len(project_info.get('classes', []))
                version_info = metadata.get('version', {})
                images_count = version_info.get('images', 0)
                
                self.logger.info(f"📊 Dataset info: {classes_count} kelas, {images_count} gambar")
            
            return metadata
            
        except requests.RequestException as e:
            if "404" in str(e):
                raise Exception(f"Dataset tidak ditemukan: {params['workspace']}/{params['project']}:{params['version']}")
            elif "401" in str(e) or "403" in str(e):
                raise Exception("API key tidak valid atau tidak memiliki akses ke dataset")
            else:
                raise Exception(f"Error koneksi ke Roboflow: {str(e)}")
        except Exception as e:
            raise Exception(f"Error mendapatkan metadata: {str(e)}")
    
    def _log_final_structure(self, stats: Dict[str, Any]) -> None:
        """Log struktur final dataset."""
        if not self.logger:
            return
        
        self.logger.info("📁 Struktur dataset final:")
        for split, split_stats in stats.get('splits', {}).items():
            if split_stats.get('images', 0) > 0:
                self.logger.info(f"   • {split}: {split_stats['images']} gambar, {split_stats['labels']} label")
                self.logger.info(f"     Path: {split_stats['path']}")
        
        storage_type = "Google Drive" if self.env_manager.is_drive_mounted else "Local Storage"
        self.logger.info(f"💾 Storage: {storage_type}")