"""
File: smartcash/dataset/services/downloader/download_service.py
Deskripsi: Layanan utama untuk mengelola download dataset dengan integrasi komponen dan notifikasi standar
"""

import os, time, shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetError
from smartcash.dataset.utils.dataset_utils import DatasetUtils
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
from smartcash.dataset.services.downloader.notification_utils import notify_service_event
from smartcash.components.observer import notify, EventTopics

class DownloadService:
    """Layanan utama untuk mengelola download dataset dari berbagai sumber."""
    
    def __init__(self, output_dir: str = "data", config: Optional[Dict] = None, logger=None, num_workers: int = 4):
        """Inisialisasi DownloadService dengan komponen teroptimasi."""
        self.config = config or {}
        self.data_dir, self.logger, self.num_workers = Path(output_dir), logger or get_logger("dataset_download_service"), num_workers
        self.temp_dir, self.downloads_dir = self.data_dir / ".temp", self.data_dir / "downloads"
        [os.makedirs(d, exist_ok=True) for d in [self.temp_dir, self.downloads_dir]]
        self.utils = DatasetUtils(self.config, output_dir, logger)
        rf_config = self.config.get('data', {}).get('roboflow', {})
        self.api_key, self.workspace, self.project, self.version = rf_config.get('api_key') or os.environ.get("ROBOFLOW_API_KEY"), rf_config.get('workspace', 'smartcash-wo2us'), rf_config.get('project', 'rupiah-emisi-2022'), rf_config.get('version', '3')
        from smartcash.dataset.services.downloader.roboflow_downloader import RoboflowDownloader
        from smartcash.dataset.services.downloader.download_validator import DownloadValidator
        from smartcash.dataset.services.downloader.file_processor import DownloadFileProcessor
        from smartcash.dataset.services.downloader.backup_service import BackupService
        self.roboflow_downloader, self.validator = RoboflowDownloader(logger=self.logger), DownloadValidator(logger=self.logger, num_workers=self.num_workers)
        self.processor, self.backup_service = DownloadFileProcessor(logger=self.logger, num_workers=self.num_workers), BackupService(logger=self.logger)
        try:
            from smartcash.components.observer.manager_observer import ObserverManager
            self.observer_manager = ObserverManager()
        except ImportError:
            self.observer_manager = None
        self.logger.info(f"📥 DatasetDownloadService diinisialisasi dengan {num_workers} workers\n   • Data dir: {self.data_dir}\n   • Default sumber: {self.workspace}/{self.project}:{self.version}")
    
    def download_from_roboflow(self, api_key: Optional[str] = None, workspace: Optional[str] = None, project: Optional[str] = None, version: Optional[str] = None, format: str = "yolov5pytorch", output_dir: Optional[str] = None, show_progress: bool = True, verify_integrity: bool = True, backup_existing: bool = False) -> Dict[str, Any]:
        """Download dataset dari Roboflow dengan penanganan error yang lebih baik."""
        start_time = time.time()
        notify_service_event("download", "start", self, self.observer_manager, message="Memulai proses download dataset", step="download")
        api_key, workspace, project, version = api_key or self.api_key, workspace or self.workspace, project or self.project, version or self.version
        if not api_key: raise DatasetError("🔑 API key tidak tersedia. Berikan api_key melalui parameter atau config.")
        if not workspace or not project or not version: raise DatasetError("📋 Workspace, project, dan version diperlukan.")
        output_dir = output_dir or str(self.downloads_dir / f"{workspace}_{project}_{version}")
        output_path, temp_download_path = Path(output_dir), Path(output_dir).with_name(f"{Path(output_dir).name}_temp")
        try:
            if self._handle_backup(output_path, backup_existing, show_progress): self.backup_service.cleanup_old_backups()
            if temp_download_path.exists(): shutil.rmtree(temp_download_path)
            os.makedirs(temp_download_path, exist_ok=True)
            notify_service_event("download", "progress", self, self.observer_manager, step="metadata", message="Mendapatkan metadata dataset", progress=2, total_steps=5, current_step=2)
            metadata = self.roboflow_downloader.get_roboflow_metadata(workspace, project, version, api_key, format, self.temp_dir)
            if 'export' not in metadata or 'link' not in metadata['export']: raise DatasetError("❌ Format metadata tidak valid, tidak ada link download")
            download_url, file_size_mb = metadata['export']['link'], metadata.get('export', {}).get('size', 0)
            if file_size_mb > 0: self.logger.info(f"📦 Ukuran dataset: {file_size_mb:.2f} MB")
            notify_service_event("download", "progress", self, self.observer_manager, step="download", message=f"Mendownload dataset ({file_size_mb:.2f} MB)", progress=3, total_steps=5, current_step=3)
            download_success = self.roboflow_downloader.process_roboflow_download(download_url, temp_download_path, show_progress)
            if not download_success: raise DatasetError("❌ Proses download dan ekstraksi gagal")
            if verify_integrity:
                notify_service_event("download", "progress", self, self.observer_manager, step="verify", message="Verifikasi integritas dataset", progress=4, total_steps=5, current_step=4)
                valid = self.validator.verify_download(str(temp_download_path), metadata)
                if not valid: self.logger.warning("⚠️ Verifikasi dataset gagal, tapi melanjutkan proses")
            notify_service_event("download", "progress", self, self.observer_manager, step="finalize", message="Memindahkan dataset ke lokasi final", progress=5, total_steps=5, current_step=5)
            if output_path.exists():
                self.logger.info(f"🧹 Menghapus direktori sebelumnya: {output_path}")
                # Tangani kasus symbolic link
                if output_path.is_symlink():
                    os.unlink(str(output_path))
                else:
                    shutil.rmtree(output_path, ignore_errors=True)
            self.logger.info(f"🔄 Memindahkan dataset ke lokasi final: {output_path}"); shutil.move(str(temp_download_path), str(output_path))
            stats, elapsed_time = self.validator.get_dataset_stats(output_dir), time.time() - start_time
            notify_service_event("download", "complete", self, self.observer_manager, message=f"Download dataset selesai: {stats.get('total_images', 0)} gambar", duration=elapsed_time)
            self.logger.success(f"✅ Dataset {workspace}/{project}:{version} berhasil didownload ke {output_dir} ({elapsed_time:.1f}s)\n   • Ukuran: {file_size_mb:.2f} MB\n   • Gambar: {stats.get('total_images', 0)} file\n   • Label: {stats.get('total_labels', 0)} file")
            return {"status": "success", "workspace": workspace, "project": project, "version": version, "format": format, "output_dir": output_dir, "stats": stats, "duration": elapsed_time}
        except Exception as e:
            notify_service_event("download", "error", self, self.observer_manager, message=f"Error download dataset: {str(e)}")
            if temp_download_path.exists(): shutil.rmtree(temp_download_path, ignore_errors=True)
            self.logger.error(f"❌ Error download dataset: {str(e)}")
            raise DatasetError(f"Error download dataset: {str(e)}")
    
    def _handle_backup(self, output_path: Path, backup_existing: bool, show_progress: bool) -> bool:
        """Buat backup dataset yang ada jika diperlukan dan bukan symlink."""
        # Skip backup jika path tidak ada, kosong, atau merupakan symlink
        if not backup_existing or not output_path.exists() or output_path.is_symlink() or not any(output_path.iterdir()):
            return False
            
        notify_service_event("download", "progress", self, self.observer_manager, 
                        step="backup", message="Backup dataset yang ada", 
                        progress=1, total_steps=5, current_step=1)
                        
        backup_result = self.backup_service.backup_dataset(output_path, show_progress=show_progress)
        
        if backup_result.get("status") == "success":
            return True
        elif backup_result.get("status") == "empty":
            self.logger.info("ℹ️ Direktori output ada tapi kosong, skip backup")
            return False
        else:
            self.logger.warning(f"⚠️ Gagal backup: {backup_result.get('message')}")
            return False

    def export_to_local(self, source_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None, show_progress: bool = True, backup_existing: bool = False) -> Dict[str, Any]:
        """Export dataset dari format Roboflow ke struktur folder lokal standar."""
        start_time = time.time()
        notify_service_event("export", "start", self, self.observer_manager, message="Memulai ekspor dataset ke struktur lokal")
        self.logger.info("📤 Mengexport dataset ke struktur folder lokal...")
        src_path = Path(source_dir)
        if not src_path.exists(): raise DatasetError(f"❌ Direktori sumber tidak ditemukan: {src_path}")
        dst_path = Path(output_dir) if output_dir else self.data_dir
        if self._handle_backup(dst_path, backup_existing, show_progress): self.backup_service.cleanup_old_backups()
        notify_service_event("export", "progress", self, self.observer_manager, step="export", message=f"Mengekspor dataset ke {dst_path}", progress=2, total_steps=3, current_step=2)
        result = self.processor.export_to_local(src_path, dst_path, show_progress)
        notify_service_event("export", "progress", self, self.observer_manager, step="verify", message="Memvalidasi hasil ekspor", progress=3, total_steps=3, current_step=3)
        valid, elapsed_time = self.validator.verify_local_dataset(dst_path), time.time() - start_time
        notify_service_event("export", "complete", self, self.observer_manager, message=f"Ekspor dataset selesai: {result.get('copied', 0)} file", duration=elapsed_time, status="success" if valid else "warning")
        if valid: self.logger.success(f"✅ Dataset berhasil diexport ({elapsed_time:.1f}s):\n   • Files: {result.get('copied', 0)} file\n   • Errors: {result.get('errors', 0)} error\n   • Output: {dst_path}")
        else: self.logger.warning(f"⚠️ Dataset berhasil diexport tetapi validasi gagal ({elapsed_time:.1f}s):\n   • Files: {result.get('copied', 0)} file\n   • Errors: {result.get('errors', 0)} error\n   • Output: {dst_path}")
        return {'paths': {split: str(dst_path / split) for split in DEFAULT_SPLITS}, 'output_dir': str(dst_path), 'duration': elapsed_time, 'status': 'success' if valid else 'warning', **result}
    
    def process_zip_file(self, zip_path: Union[str, Path], output_dir: Union[str, Path], extract_only: bool = False, validate_after: bool = True, remove_zip: bool = False, show_progress: bool = True) -> Dict[str, Any]:
        """Proses lengkap file ZIP dengan delegasi ke file_processor."""
        return self.processor.process_zip_file(zip_path=zip_path, output_dir=output_dir, extract_only=extract_only, remove_zip=remove_zip, show_progress=show_progress)
    
    def pull_dataset(self, format: str = "yolov5pytorch", api_key: Optional[str] = None, workspace: Optional[str] = None, project: Optional[str] = None, version: Optional[str] = None, show_progress: bool = True, force_download: bool = False, backup_existing: bool = False) -> Dict[str, Any]:
        """One-step untuk download dan setup dataset siap pakai."""
        start_time = time.time()
        notify_service_event("pull_dataset", "start", self, self.observer_manager, message="Memulai persiapan dataset")
        if not force_download and self.validator.is_dataset_available(self.data_dir, verify_content=True):
            self.logger.info("✅ Dataset sudah tersedia di lokal")
            notify_service_event("pull_dataset", "complete", self, self.observer_manager, message="Dataset sudah tersedia di lokal")
            stats = self.validator.get_local_stats(self.data_dir)
            return {'status': 'local', 'paths': {split: str(self.data_dir / split) for split in DEFAULT_SPLITS}, 'data_dir': str(self.data_dir), 'stats': stats, 'duration': 0}
        action_msg = "Memulai download ulang dataset dari Roboflow..." if force_download else "Dataset belum tersedia atau tidak lengkap, mendownload dari Roboflow..."
        self.logger.info("🔄 " + action_msg)
        try:
            download_result = self.download_from_roboflow(api_key=api_key, workspace=workspace, project=project, version=version, format=format, show_progress=show_progress, backup_existing=backup_existing)
            export_result = self.export_to_local(download_result.get('output_dir'), self.data_dir, show_progress=show_progress, backup_existing=backup_existing)
            elapsed_time = time.time() - start_time
            notify_service_event("pull_dataset", "complete", self, self.observer_manager, message=f"Dataset siap digunakan: {export_result.get('copied', 0)} file", duration=elapsed_time)
            return {'status': 'downloaded', 'paths': export_result.get('paths', {}), 'data_dir': str(self.data_dir), 'download_dir': download_result.get('output_dir'), 'export_result': export_result, 'download_result': download_result, 'stats': self.validator.get_local_stats(self.data_dir), 'duration': elapsed_time}
        except Exception as e:
            notify_service_event("pull_dataset", "error", self, self.observer_manager, message=f"Error saat persiapan dataset: {str(e)}")
            self.logger.error(f"❌ Error saat pull dataset: {str(e)}")
            if self.validator.is_dataset_available(self.data_dir, verify_content=False):
                self.logger.warning("⚠️ Download gagal tetapi dataset masih tersedia di lokal (mungkin tidak lengkap)")
                stats = self.validator.get_local_stats(self.data_dir)
                return {'status': 'partial', 'paths': {split: str(self.data_dir / split) for split in DEFAULT_SPLITS}, 'data_dir': str(self.data_dir), 'stats': stats, 'error': str(e)}
            raise DatasetError(f"Error pull dataset: {str(e)}")
    
    def import_from_zip(self, zip_file: Union[str, Path], target_dir: Optional[Union[str, Path]] = None, remove_zip: bool = False, show_progress: bool = True, backup_existing: bool = False) -> Dict[str, Any]:
        """Import dataset dari file ZIP."""
        start_time, zip_path, target_path = time.time(), Path(zip_file), Path(target_dir) if target_dir else self.data_dir
        notify_service_event("zip_import", "start", self, self.observer_manager, message="Memulai import dataset dari file ZIP", zip_file=str(zip_path))
        self.logger.info(f"📦 Mengimport dataset dari {zip_path} ke {target_path}")
        try:
            if self._handle_backup(target_path, backup_existing, show_progress): self.backup_service.cleanup_old_backups()
            result = self.processor.process_zip_file(zip_path=zip_path, output_dir=target_path, extract_only=False, remove_zip=remove_zip, show_progress=show_progress)
            elapsed_time = time.time() - start_time
            result['elapsed_time'] = elapsed_time
            notify_service_event("zip_import", "complete", self, self.observer_manager, message=f"Import dataset selesai: {result.get('stats', {}).get('total_images', 0)} gambar", duration=elapsed_time)
            return result
        except Exception as e:
            notify_service_event("zip_import", "error", self, self.observer_manager, message=f"Error saat import dataset: {str(e)}")
            self.logger.error(f"❌ Error saat import dataset: {str(e)}")
            raise DatasetError(f"Error saat import dataset: {str(e)}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Dapatkan informasi dataset dari konfigurasi dan status lokal."""
        is_available = self.validator.is_dataset_available(self.data_dir)
        local_stats = self.validator.get_local_stats(self.data_dir) if is_available else {}
        info = {'name': self.project, 'workspace': self.workspace, 'version': self.version, 'is_available_locally': is_available, 'local_stats': local_stats, 'data_dir': str(self.data_dir), 'has_api_key': bool(self.api_key)}
        if is_available:
            total_images = sum(local_stats.get(split, 0) for split in DEFAULT_SPLITS)
            self.logger.info(f"🔍 Dataset (Lokal): {info['name']} v{info['version']} | Total: {total_images} gambar | Train: {local_stats.get('train', 0)}, Valid: {local_stats.get('valid', 0)}, Test: {local_stats.get('test', 0)}")
        else:
            self.logger.info(f"🔍 Dataset (akan didownload): {info['name']} v{info['version']} dari {info['workspace']}")
        if not info['has_api_key']: self.logger.warning(f"⚠️ API key untuk Roboflow tidak tersedia")
        return info
    
    def check_dataset_structure(self) -> Dict[str, Any]:
        """Periksa struktur dataset dan tampilkan laporan."""
        result = {'data_dir': str(self.data_dir), 'is_valid': False, 'splits': {}}
        self.logger.info(f"🔍 Memeriksa struktur dataset di {self.data_dir}...")
        stats = self.validator.get_dataset_stats(self.data_dir)
        for split in DEFAULT_SPLITS:
            split_info = {'exists': False, 'has_images_dir': False, 'has_labels_dir': False, 'image_count': 0, 'label_count': 0, 'valid': False}
            split_dir = self.data_dir / split
            if split_dir.exists():
                split_info['exists'] = True
                split_info['has_images_dir'] = (split_dir / 'images').exists()
                split_info['has_labels_dir'] = (split_dir / 'labels').exists()
                if split in stats.get('splits', {}):
                    split_stats = stats['splits'][split]
                    split_info['image_count'] = split_stats.get('images', 0)
                    split_info['label_count'] = split_stats.get('labels', 0)
                    split_info['valid'] = split_info['image_count'] > 0 and split_info['label_count'] > 0
                self.logger.info(f"   • {split}: {'✅' if split_info['valid'] else '❌'} ({split_info['image_count']} gambar, {split_info['label_count']} label)")
            else:
                self.logger.warning(f"   • {split}: ❌ Direktori tidak ditemukan")
            result['splits'][split] = split_info
        result['is_valid'] = any(info['valid'] for info in result['splits'].values())
        self.logger.success(f"✅ Struktur dataset valid di {self.data_dir}") if result['is_valid'] else self.logger.warning(f"⚠️ Struktur dataset tidak valid atau tidak lengkap di {self.data_dir}")
        return result