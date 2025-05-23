"""
File: smartcash/dataset/services/downloader/download_service.py
Deskripsi: Updated download service dengan Drive integration dan progress callback
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

class DownloadService:
    """Download service dengan Drive integration dan progress callback support."""
    
    def __init__(self, output_dir: str, config: Dict[str, Any], logger=None, num_workers: int = 4):
        self.config = config
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        self._progress_callback: Optional[Callable] = None
        
        # Environment manager untuk Drive detection
        self.env_manager = get_environment_manager()
        
        # Setup paths dengan Drive priority
        self.output_dir, self.data_dir = self._setup_drive_paths(output_dir)
        
        # Initialize components dengan Drive paths
        self.downloader = RoboflowDownloader(logger=logger)
        self.validator = DownloadValidator(logger=logger, num_workers=num_workers)
        self.file_processor = DownloadFileProcessor(
            output_dir=str(self.output_dir), config=config, 
            logger=logger, num_workers=num_workers
        )
        
        self._backup_service = None
    
    def set_progress_callback(self, callback: Callable[[str, int, int, str], None]) -> None:
        """Set callback untuk progress updates ke UI."""
        self._progress_callback = callback
        # Propagate ke components
        if hasattr(self.downloader, 'set_progress_callback'):
            self.downloader.set_progress_callback(callback)
    
    def _notify_progress(self, step: str, current: int, total: int, message: str) -> None:
        """Notify progress via callback."""
        if self._progress_callback:
            try:
                self._progress_callback(step, current, total, message)
            except Exception:
                pass  # Ignore callback errors
    
    def _setup_drive_paths(self, output_dir: str) -> Tuple[Path, Path]:
        """Setup paths dengan Drive priority untuk Colab."""
        if self.env_manager.is_colab and self.env_manager.is_drive_mounted:
            # Drive paths
            drive_base = self.env_manager.drive_path
            drive_output = drive_base / 'downloads' / Path(output_dir).name
            drive_data = drive_base / 'data'
            
            # Buat direktori di Drive
            drive_output.mkdir(parents=True, exist_ok=True)
            drive_data.mkdir(parents=True, exist_ok=True)
            
            # Setup symlinks di Colab ke Drive
            colab_output = Path('/content') / Path(output_dir).name
            colab_data = Path('/content/data')
            
            self._setup_symlink(colab_output, drive_output)
            self._setup_symlink(colab_data, drive_data)
            
            self.logger.info(f"📁 Dataset akan disimpan di Drive: {drive_output}")
            return drive_output, drive_data
        else:
            # Local paths untuk non-Colab atau Drive tidak mounted
            output_path = Path(output_dir)
            data_path = output_path.parent
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"📁 Dataset akan disimpan lokal: {output_path}")
            return output_path, data_path
    
    def _setup_symlink(self, local_path: Path, drive_path: Path) -> None:
        """Setup symlink dari local ke Drive."""
        try:
            if local_path.exists():
                if local_path.is_symlink():
                    if local_path.resolve() == drive_path.resolve():
                        return  # Symlink sudah benar
                    local_path.unlink()
                else:
                    shutil.rmtree(local_path, ignore_errors=True)
            
            local_path.symlink_to(drive_path)
            self.logger.debug(f"🔗 Symlink dibuat: {local_path} -> {drive_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal buat symlink {local_path}: {str(e)}")

    def download_from_roboflow(self, api_key: Optional[str] = None, workspace: Optional[str] = None, 
                              project: Optional[str] = None, version: Optional[str] = None, 
                              format: str = "yolov5pytorch", output_dir: Optional[str] = None, 
                              show_progress: bool = True, verify_integrity: bool = True, 
                              backup_existing: bool = False) -> Dict[str, Any]:
        """Download dari Roboflow dengan Drive storage dan progress callback."""
        start_time = time.time()
        
        # Notify start
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
        
        # Setup output path (Drive integrated)
        if output_dir:
            output_path = self._setup_custom_output(output_dir)
        else:
            output_path = self.output_dir / f"{workspace}_{project}_{version}"
        
        temp_download_path = output_path.with_name(f"{output_path.name}_temp")
        
        try:
            self._notify_progress("prepare", 5, 100, "Mempersiapkan download")
            
            # Backup handling
            if backup_existing and output_path.exists() and any(output_path.iterdir()):
                self._notify_progress("backup", 10, 100, "Membuat backup")
                self._handle_backup(output_path, show_progress)
            
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
                self._notify_progress("verify", 80, 100, "Memverifikasi dataset")
                valid = self.validator.verify_download(str(temp_download_path), metadata)
                if not valid:
                    self.logger.warning("⚠️ Verifikasi gagal, melanjutkan proses")
            
            # Finalize - Move to final location (Drive)
            self._notify_progress("finalize", 90, 100, "Memindahkan ke lokasi final")
            self._finalize_download(temp_download_path, output_path)
            
            # Complete
            stats = self.validator.get_dataset_stats(str(output_path))
            elapsed_time = time.time() - start_time
            
            self._notify_progress("complete", 100, 100, f"Download selesai: {stats.get('total_images', 0)} gambar")
            
            self.logger.success(
                f"✅ Dataset {workspace}/{project}:{version} berhasil didownload ({elapsed_time:.1f}s)\n"
                f"   • Lokasi: {output_path}\n"
                f"   • Ukuran: {file_size_mb:.2f} MB\n"
                f"   • Gambar: {stats.get('total_images', 0)} file"
            )
            
            return {
                "status": "success", "workspace": workspace, "project": project, 
                "version": version, "format": format, "output_dir": str(output_path),
                "stats": stats, "duration": elapsed_time, "drive_storage": self.env_manager.is_drive_mounted
            }
            
        except Exception as e:
            self._notify_progress("error", 0, 100, f"Error: {str(e)}")
            if temp_download_path.exists():
                shutil.rmtree(temp_download_path, ignore_errors=True)
            self.logger.error(f"❌ Error download: {str(e)}")
            raise DatasetError(f"Error download: {str(e)}")
    
    def _setup_custom_output(self, output_dir: str) -> Path:
        """Setup custom output path dengan Drive integration."""
        if self.env_manager.is_colab and self.env_manager.is_drive_mounted:
            # Custom path di Drive
            drive_output = self.env_manager.drive_path / 'downloads' / Path(output_dir).name
            drive_output.mkdir(parents=True, exist_ok=True)
            
            # Setup symlink
            colab_output = Path('/content') / Path(output_dir).name
            self._setup_symlink(colab_output, drive_output)
            
            return drive_output
        else:
            # Local path
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            return output_path
    
    def _finalize_download(self, temp_path: Path, final_path: Path) -> None:
        """Finalize download ke Drive location."""
        # Remove existing
        if final_path.exists():
            if final_path.is_symlink():
                final_path.unlink()
            else:
                shutil.rmtree(final_path, ignore_errors=True)
        
        # Move dengan progress notification
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
    
    def _handle_backup(self, output_path: Path, show_progress: bool) -> None:
        """Handle backup dengan Drive storage."""
        if self._backup_service is None:
            from smartcash.dataset.services.downloader.backup_service import BackupService
            self._backup_service = BackupService(self.config, self.logger)
        
        backup_result = self._backup_service.backup_dataset(output_path, show_progress=show_progress)
        if backup_result.get("status") == "success":
            self._backup_service.cleanup_old_backups(max_backups=3)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset info dengan Drive storage info."""
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
            'has_api_key': bool(self.config.get('data', {}).get('roboflow', {}).get('api_key') or os.environ.get("ROBOFLOW_API_KEY"))
        }
        
        if is_available:
            total_images = sum(local_stats.get(split, 0) for split in DEFAULT_SPLITS)
            storage_location = "Drive" if self.env_manager.is_drive_mounted else "Local"
            self.logger.info(f"🔍 Dataset ({storage_location}): {info['name']} v{info['version']} | Total: {total_images} gambar")
        
        return info