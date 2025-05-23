"""
File: smartcash/ui/dataset/download/services/enhanced_ui_download_service.py
Deskripsi: Enhanced download service dengan dual progress tracking, dataset organization, dan path management yang benar
"""

import time
from pathlib import Path
from typing import Dict, Any
from smartcash.dataset.services.downloader.ui_roboflow_downloader import UIRoboflowDownloader
from smartcash.ui.dataset.download.services.enhanced_progress_bridge import EnhancedProgressBridge
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer
from smartcash.common.constants.paths import get_paths_for_environment
from smartcash.common.environment import get_environment_manager

class EnhancedUIDownloadService:
    """Enhanced download service dengan dual progress tracking dan dataset organization."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
        
        # Environment manager untuk path resolution
        self.env_manager = get_environment_manager()
        self.paths = get_paths_for_environment(
            is_colab=self.env_manager.is_colab,
            is_drive_mounted=self.env_manager.is_drive_mounted
        )
        
        # Enhanced progress bridge dengan dual tracking
        self.progress_bridge = EnhancedProgressBridge(
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
    
    def download_dataset(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Download dengan enhanced dual progress tracking dan dataset organization."""
        start_time = time.time()
        
        # Define steps berdasarkan opsi
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
            if not self.downloader.download_and_extract(download_url, download_path):
                raise Exception("Download atau ekstraksi gagal")
            
            self.progress_bridge.notify_step_complete("Dataset berhasil didownload")
            
            # Step 4: Organize dataset (pindah ke struktur final)
            self.progress_bridge.notify_step_start("organize", "Mengorganisir dataset ke struktur final")
            organize_result = self.organizer.organize_dataset(str(download_path), remove_source=True)
            
            if organize_result['status'] != 'success':
                raise Exception(f"Gagal mengorganisir dataset: {organize_result.get('message')}")
            
            self.progress_bridge.notify_step_complete("Dataset berhasil diorganisir")
            
            # Step 5: Verify hasil final (optional)
            if len(steps) > 4:  # Ada step verify
                self.progress_bridge.notify_step_start("verify", "Memverifikasi hasil akhir")
                final_stats = self._get_final_stats()
                self.progress_bridge.notify_step_complete("Verifikasi selesai")
            else:
                final_stats = organize_result
            
            # 🎉 Complete
            duration = time.time() - start_time
            total_images = final_stats.get('total_images', 0)
            success_message = f"Download dan organisasi selesai: {total_images} gambar dalam {duration:.1f}s"
            
            self.progress_bridge.notify_complete(success_message, duration)
            
            if self.logger:
                self.logger.success(f"✅ {success_message}")
                self._log_final_structure(final_stats)
            
            return {
                'status': 'success',
                'output_dir': self.paths['data_root'],
                'download_path': str(download_path),
                'stats': final_stats,
                'duration': duration,
                'drive_storage': self.env_manager.is_drive_mounted,
                'organized': True
            }
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            self.progress_bridge.notify_error(f"Proses gagal: {error_msg}")
            
            if self.logger:
                self.logger.error(f"❌ Download error setelah {duration:.1f}s: {error_msg}")
            
            return {'status': 'error', 'message': error_msg, 'duration': duration}
    
    def _define_process_steps(self, params: Dict[str, Any]) -> list:
        """Define steps berdasarkan parameter dan opsi."""
        steps = [
            {'name': 'validate', 'weight': 5, 'description': 'Validasi Parameter'},
            {'name': 'metadata', 'weight': 10, 'description': 'Ambil Metadata'},
            {'name': 'download', 'weight': 60, 'description': 'Download Dataset'},
            {'name': 'organize', 'weight': 20, 'description': 'Organisir Dataset'}
        ]
        
        # Tambah step verify jika diminta
        if params.get('validate_dataset', False):
            steps.append({'name': 'verify', 'weight': 5, 'description': 'Verifikasi Dataset'})
        
        return steps
    
    def _downloader_progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Callback dari downloader untuk update step progress."""
        if step in ['download', 'extract']:
            # Convert ke percentage untuk step progress
            step_progress = int((current / total) * 100) if total > 0 else 0
            self.progress_bridge.notify_step_progress(step_progress, message)
    
    def _organizer_progress_callback(self, step: str, current: int, total: int, message: str) -> None:
        """Callback dari organizer untuk update step progress."""
        if step == 'organize':
            step_progress = current  # Organizer sudah mengirim percentage
            self.progress_bridge.notify_step_progress(step_progress, message)
    
    def _validate_params(self, params: Dict[str, Any]) -> None:
        """Enhanced parameter validation."""
        required = ['workspace', 'project', 'version', 'api_key']
        missing = [p for p in required if not params.get(p)]
        
        if missing:
            raise ValueError(f"Parameter tidak lengkap: {', '.join(missing)}")
        
        # Validate API key format
        api_key = params['api_key']
        if len(api_key) < 10:
            raise ValueError("API key terlalu pendek, periksa kembali")
        
        # Ensure directories exist
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
            
            # Enhanced metadata validation
            if 'export' not in metadata:
                raise ValueError("Response tidak mengandung export data")
            
            if 'link' not in metadata['export']:
                raise ValueError("Export link tidak ditemukan dalam response")
            
            # Log dataset info
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
    
    def _get_final_stats(self) -> Dict[str, Any]:
        """Get statistik dataset yang sudah diorganisir."""
        return self.organizer.check_organized_dataset()
    
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