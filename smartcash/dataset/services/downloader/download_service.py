"""
File: smartcash/dataset/services/downloader/download_service.py
Deskripsi: Updated download service dengan dataset organizer integration dan path consistency
"""

import os, time, shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.common.environment import get_environment_manager
from smartcash.common.exceptions import DatasetError
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
from smartcash.dataset.services.downloader.roboflow_downloader import RoboflowDownloader
from smartcash.dataset.services.downloader.download_validator import DownloadValidator
from smartcash.dataset.services.downloader.file_processor import DownloadFileProcessor
from smartcash.dataset.services.organizer.dataset_organizer import DatasetOrganizer

class DownloadService:
    """Download service dengan dataset organizer integration."""
    
    def __init__(self, output_dir: str, config: Dict[str, Any], logger=None, num_workers: int = 4):
        self.config = config
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        self._progress_callback: Optional[Callable] = None
        
        # Environment manager untuk path management
        self.env_manager = get_environment_manager()
        
        # Use downloads folder sebagai temporary location
        from smartcash.common.constants.paths import get_paths_for_environment
        self.paths = get_paths_for_environment(
            is_colab=self.env_manager.is_colab,
            is_drive_mounted=self.env_manager.is_drive_mounted
        )
        
        # Setup download path (temporary) dan final data path
        self.download_dir = Path(self.paths['downloads'])
        self.data_dir = Path(self.paths['data_root'])
        
        # Initialize components
        self.downloader = RoboflowDownloader(logger=logger)
        self.validator = DownloadValidator(logger=logger, num_workers=num_workers)
        self.file_processor = DownloadFileProcessor(
            output_dir=str(self.download_dir), config=config, 
            logger=logger, num_workers=num_workers
        )
        
        # Dataset organizer untuk move ke final structure
        self.organizer = DatasetOrganizer(logger=logger)
        
        self._backup_service = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set callback untuk progress updates ke UI."""
        self._progress_callback = callback
        if hasattr(self.downloader, 'set_progress_callback'):
            self.downloader.set_progress_callback(callback)
        if hasattr(self.organizer, 'set_progress_callback'):
            self.organizer.set_progress_callback(callback)
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass
    
    def download_from_roboflow(self, api_key: Optional[str] = None, workspace: Optional[str] = None, 
                              project: Optional[str] = None, version: Optional[str] = None, 
                              format: str = "yolov5pytorch", output_dir: Optional[str] = None, 
                              show_progress: bool = True, verify_integrity: bool = True, 
                              backup_existing: bool = False, organize_dataset: bool = True) -> Dict[str, Any]:
        """Download dari Roboflow dengan organizer integration."""
        start_time = time.time()
        
        self._notify_progress("start", 0, 100, "Memulai download dataset")
        
        # Parameter handling
        api_key = api_key or self.config.get('data', {}).get('roboflow', {}).get('api_key') or os.environ.get("ROBOFLOW_API_KEY")
        workspace = workspace or self.config.get('data', {}).get('roboflow', {}).get('workspace', 'smartcash-wo2us')
        project = project or self.config.get('data', {}).get('roboflow', {}).get('project', 'rupiah-emisi-2022')
        version = version or self.config.get('data', {}).get('roboflow', {}).get('version', '3')
        
        if not api_key:
            raise DatasetError("🔑 API key tidak tersedia")
        if not all([workspace, project, version]):
            raise DatasetError("📋 Workspace, project, dan version diperlukan")
        
        # Setup temporary download path
        temp_download_path = self.download_dir / f"{workspace}_{project}_{version}"
        
        try:
            self._notify_progress("prepare", 5, 100, "Mempersiapkan download")
            
            # Backup handling untuk final data structure jika ada
            if backup_existing and organize_dataset:
                existing_stats = self.organizer.check_organized_dataset()
                if existing_stats['is_organized']:
                    self._notify_progress("backup", 10, 100, "Membuat backup dataset lama")
                    self._handle_backup_organized_dataset()
            
            # Clean temp directory
            if temp_download_path.exists():
                shutil.rmtree(temp_download_path, ignore_errors=True)
            temp_download_path.mkdir(parents=True, exist_ok=True)
            
            # Get metadata
            self._notify_progress("metadata", 15, 100, "Mendapatkan metadata dataset")
            metadata = self.downloader.get_roboflow_metadata(workspace, project, version, api_key, format, str(self.data_dir))
            
            if 'export' not in metadata or 'link' not in metadata['export']:
                raise DatasetError("❌ Format metadata tidak valid")
            
            download_url = metadata['export']['link']
            file_size_mb = metadata.get('export', {}).get('size', 0)
            
            if file_size_mb > 0:
                self.logger.info(f"📦 Ukuran dataset: {file_size_mb:.2f} MB")
            
            # Download dengan progress callback
            self._notify_progress("download", 20, 100, f"Mendownload dataset ({file_size_mb:.2f} MB)")
            download_success = self.downloader.process_roboflow_download(download_url, temp_download_path, show_progress=False)
            
            if not download_success:
                raise DatasetError("❌ Download gagal")
            
            # Verify
            if verify_integrity:
                self._notify_progress("verify", 70, 100, "Memverifikasi dataset")
                valid = self.validator.verify_download(str(temp_download_path), metadata)
                if not valid:
                    self.logger.warning("⚠️ Verifikasi gagal, melanjutkan proses")
            
            # Organize dataset ke final structure jika diminta
            final_stats = {}
            if organize_dataset:
                self._notify_progress("organize", 80, 100, "Mengorganisir dataset ke struktur final")
                
                organize_result = self.organizer.organize_dataset(str(temp_download_path), remove_source=True)
                if organize_result['status'] != 'success':
                    self.logger.warning(f"⚠️ Gagal mengorganisir dataset: {organize_result.get('message')}")
                    # Fallback ke stats dari temp directory
                    final_stats = self.validator.get_dataset_stats(str(temp_download_path))
                else:
                    final_stats = organize_result
                    final_output_dir = self.paths['data_root']
            else:
                # Tanpa organize, pindah ke output directory
                if output_dir:
                    final_output_path = Path(output_dir)
                else:
                    final_output_path = self.download_dir / f"{workspace}_{project}_{version}_final"
                
                self._notify_progress("finalize", 90, 100, "Memindahkan ke lokasi final")
                self._finalize_download(temp_download_path, final_output_path)
                final_stats = self.validator.get_dataset_stats(str(final_output_path))
                final_output_dir = str(final_output_path)
            
            # Complete
            elapsed_time = time.time() - start_time
            total_images = final_stats.get('total_images', 0)
            
            self._notify_progress("complete", 100, 100, f"Download selesai: {total_images} gambar")
            
            self.logger.success(
                f"✅ Dataset {workspace}/{project}:{version} berhasil didownload ({elapsed_time:.1f}s)\n"
                f"   • Lokasi: {final_output_dir}\n"
                f"   • Ukuran: {file_size_mb:.2f} MB\n"
                f"   • Gambar: {total_images} file\n"
                f"   • Organized: {'Ya' if organize_dataset else 'Tidak'}"
            )
            
            return {
                "status": "success", "workspace": workspace, "project": project, 
                "version": version, "format": format, "output_dir": final_output_dir,
                "stats": final_stats, "duration": elapsed_time, 
                "drive_storage": self.env_manager.is_drive_mounted,
                "organized": organize_dataset
            }
            
        except Exception as e:
            self._notify_progress("error", 0, 100, f"Error: {str(e)}")
            if temp_download_path.exists():
                shutil.rmtree(temp_download_path, ignore_errors=True)
            self.logger.error(f"❌ Error download: {str(e)}")
            raise DatasetError(f"Error download: {str(e)}")
    
    def _handle_backup_organized_dataset(self) -> None:
        """Handle backup untuk organized dataset."""
        if self._backup_service is None:
            from smartcash.dataset.services.downloader.backup_service import BackupService
            self._backup_service = BackupService(self.config, self.logger)
        
        # Backup each split directory
        for split in ['train', 'valid', 'test']:
            split_path = Path(self.paths[split])
            if split_path.exists():
                backup_result = self._backup_service.backup_dataset(split_path, f"dataset_{split}", show_progress=False)
                if backup_result.get("status") == "success":
                    self.logger.info(f"💾 Backup {split} berhasil")
        
        # Cleanup old backups
        self._backup_service.cleanup_old_backups(max_backups=3)
    
    def _finalize_download(self, temp_path: Path, final_path: Path) -> None:
        """Finalize download ke final location."""
        if final_path.exists():
            if final_path.is_symlink():
                final_path.unlink()
            else:
                shutil.rmtree(final_path, ignore_errors=True)
        
        final_path.mkdir(parents=True, exist_ok=True)
        
        # Copy dengan progress tracking
        total_files = sum(1 for _ in temp_path.rglob('*') if _.is_file())
        copied_files = 0
        
        for src_file in temp_path.rglob('*'):
            if src_file.is_file():
                rel_path = src_file.relative_to(temp_path)
                dst_file = final_path / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_file, dst_file)
                
                copied_files += 1
                if copied_files % max(1, total_files // 10) == 0:
                    progress = 90 + int((copied_files / total_files) * 8)
                    self._notify_progress("finalize", progress, 100, f"Menyalin file: {copied_files}/{total_files}")
        
        # Cleanup temp
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset info dengan organized dataset awareness."""
        # Check organized dataset first
        organized_stats = self.organizer.check_organized_dataset()
        
        if organized_stats['is_organized']:
            info = {
                'name': self.config.get('data', {}).get('roboflow', {}).get('project', 'Unnamed Project'),
                'workspace': self.config.get('data', {}).get('roboflow', {}).get('workspace', 'Unnamed Workspace'),
                'version': self.config.get('data', {}).get('roboflow', {}).get('version', 'Unknown Version'),
                'is_available_locally': True,
                'local_stats': organized_stats,
                'data_dir': self.paths['data_root'],
                'drive_storage': self.env_manager.is_drive_mounted,
                'drive_path': str(self.env_manager.drive_path) if self.env_manager.drive_path else None,
                'has_api_key': bool(self.config.get('data', {}).get('roboflow', {}).get('api_key') or os.environ.get("ROBOFLOW_API_KEY")),
                'organized': True
            }
            
            total_images = organized_stats.get('total_images', 0)
            storage_location = "Drive" if self.env_manager.is_drive_mounted else "Local"
            self.logger.info(f"🔍 Dataset (Organized/{storage_location}): {info['name']} v{info['version']} | Total: {total_images} gambar")
            
            return info
        
        # Fallback ke check traditional dataset
        is_available = self.validator.is_dataset_available(self.data_dir)
        local_stats = self.validator.get_local_stats(self.data_dir) if is_available else {}
        
        info = {
            'name': self.config.get('data', {}).get('roboflow', {}).get('project', 'Unnamed Project'),
            'workspace': self.config.get('data', {}).get('roboflow', {}).get('workspace', 'Unnamed Workspace'),
            'version': self.config.get('data', {}).get('roboflow', {}).get('version', 'Unknown Version'),
            'is_available_locally': is_available,
            'local_stats': local_stats,
            'data_dir': str(self.data_dir),
            'drive_storage': self.env_manager.is_drive_mounted,
            'drive_path': str(self.env_manager.drive_path) if self.env_manager.drive_path else None,
            'has_api_key': bool(self.config.get('data', {}).get('roboflow', {}).get('api_key') or os.environ.get("ROBOFLOW_API_KEY")),
            'organized': False
        }
        
        if is_available:
            total_images = sum(local_stats.get(split, 0) for split in DEFAULT_SPLITS)
            storage_location = "Drive" if self.env_manager.is_drive_mounted else "Local"
            self.logger.info(f"🔍 Dataset (Traditional/{storage_location}): {info['name']} v{info['version']} | Total: {total_images} gambar")
        
        return info