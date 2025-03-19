"""
File: smartcash/dataset/services/downloader/download_service.py
Deskripsi: Layanan utama untuk mengelola download dataset dengan implementasi one-liner dan terintegrasi
"""

import os, time, shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import DatasetError
from smartcash.dataset.utils.dataset_utils import DatasetUtils, DEFAULT_SPLITS

# Import layanan-layanan khusus
from smartcash.dataset.services.downloader.roboflow_downloader import RoboflowDownloader
from smartcash.dataset.services.downloader.download_validator import DownloadValidator
from smartcash.dataset.services.downloader.file_processor import FileProcessor
from smartcash.dataset.services.downloader.backup_service import BackupService
from smartcash.components.observer.manager_observer import ObserverManager
from smartcash.components.observer import notify, EventTopics


class DownloadService:
    """Layanan utama untuk mengelola download dataset dari berbagai sumber."""
    
    def __init__(self, output_dir: str = "data", config: Optional[Dict] = None, logger=None, num_workers: int = 4):
        """Inisialisasi DownloadService."""
        self.config = config or {}
        self.data_dir = Path(output_dir)
        self.logger = logger or get_logger("dataset_download_service")
        self.num_workers = num_workers
        self.temp_dir, self.downloads_dir = self.data_dir / ".temp", self.data_dir / "downloads"
        [os.makedirs(d, exist_ok=True) for d in [self.temp_dir, self.downloads_dir]]
        
        # Inisialisasi komponen
        self.utils = DatasetUtils(self.config, output_dir, logger)
        
        # Ambil API key dan konfigurasi Roboflow
        rf_config = self.config.get('data', {}).get('roboflow', {})
        self.api_key = rf_config.get('api_key') or os.environ.get("ROBOFLOW_API_KEY")
        self.workspace = rf_config.get('workspace', 'smartcash-wo2us')
        self.project = rf_config.get('project', 'rupiah-emisi-2022')
        self.version = rf_config.get('version', '3')
        
        # Inisialisasi layanan-layanan khusus
        self.roboflow_downloader = RoboflowDownloader(logger=self.logger)
        self.validator = DownloadValidator(logger=self.logger)
        self.processor = FileProcessor(logger=self.logger, num_workers=self.num_workers)
        self.backup_service = BackupService(logger=self.logger)
        
        # Untuk tracking progress menggunakan observer - PERBAIKAN: Ubah dari get_instance() ke inisialisasi langsung
        try:
            self.observer_manager = ObserverManager()
        except ImportError:
            self.observer_manager = None
        
        self.logger.info(f"📥 DatasetDownloadService diinisialisasi dengan {num_workers} workers\n"
                        f"   • Data dir: {self.data_dir}\n"
                        f"   • Default sumber: {self.workspace}/{self.project}:{self.version}")
    
    def _notify(self, event_type, **kwargs):
        """Helper untuk mengirimkan notifikasi observer dengan one-liner."""
        if self.observer_manager: notify(event_type, self, **kwargs)
            
    def download_from_roboflow(self, api_key: Optional[str] = None, workspace: Optional[str] = None,
                              project: Optional[str] = None, version: Optional[str] = None,
                              format: str = "yolov5pytorch", output_dir: Optional[str] = None,
                              show_progress: bool = True, verify_integrity: bool = True,
                              backup_existing: bool = True) -> Dict[str, Any]:
        """Download dataset dari Roboflow."""
        start_time = time.time()
        
        # Setup parameter dan notifikasi
        self._notify(EventTopics.DOWNLOAD_START, step="download", message="Memulai proses download dataset", 
                    progress=0, total_steps=5, current_step=1, status="info")
        api_key, workspace = api_key or self.api_key, workspace or self.workspace
        project, version = project or self.project, version or self.version
        
        # Validasi parameter
        if not api_key: raise DatasetError("🔑 API key tidak tersedia. Berikan api_key melalui parameter atau config.")
        if not workspace or not project or not version: raise DatasetError("📋 Workspace, project, dan version diperlukan.")
        
        # Setup direktori output
        output_dir = output_dir or str(self.downloads_dir / f"{workspace}_{project}_{version}")
        output_path, temp_download_path = Path(output_dir), Path(output_dir).with_name(f"{Path(output_dir).name}_temp")
        
        try:
            # Backup existing dataset if needed
            if backup_existing and output_path.exists() and any(output_path.iterdir()):
                self._notify(EventTopics.DOWNLOAD_PROGRESS, step="backup", message="Backup dataset yang ada",
                         progress=1, total_steps=5, current_step=1, status="info")
                backup_result = self.backup_service.backup_dataset(output_path, show_progress=show_progress)
            
            # Prepare temporary directory
            if temp_download_path.exists(): shutil.rmtree(temp_download_path)
            temp_download_path.mkdir(parents=True, exist_ok=True)
            
            # Download metadata
            self._notify(EventTopics.DOWNLOAD_PROGRESS, step="metadata", message="Mendapatkan metadata dataset",
                     progress=2, total_steps=5, current_step=2, status="info")
            metadata = self.roboflow_downloader.get_roboflow_metadata(workspace, project, version, api_key, format, self.temp_dir)
            
            # Validate metadata and get download URL
            if 'export' not in metadata or 'link' not in metadata['export']: 
                raise DatasetError("❌ Format metadata tidak valid, tidak ada link download")
            download_url = metadata['export']['link']
            file_size_mb = metadata.get('export', {}).get('size', 0)
            if file_size_mb > 0: self.logger.info(f"📦 Ukuran dataset: {file_size_mb:.2f} MB")
            
            # Download the dataset
            self._notify(EventTopics.DOWNLOAD_PROGRESS, step="download", 
                     message=f"Mendownload dataset ({file_size_mb:.2f} MB)",
                     progress=3, total_steps=5, current_step=3, status="info")
            download_success = self.roboflow_downloader.process_roboflow_download(
                download_url, temp_download_path, show_progress=show_progress)
            if not download_success: raise DatasetError("❌ Proses download dan ekstraksi gagal")
            
            # Verify downloaded dataset
            if verify_integrity:
                self._notify(EventTopics.DOWNLOAD_PROGRESS, step="verify", message="Verifikasi integritas dataset",
                         progress=4, total_steps=5, current_step=4, status="info")
                valid = self.validator.verify_download(str(temp_download_path), metadata)
                if not valid: self.logger.warning("⚠️ Verifikasi dataset gagal")
            
            # Move to final location
            if output_path.exists(): 
                self.logger.info(f"🧹 Menghapus direktori sebelumnya: {output_path}")
                shutil.rmtree(output_path)
                
            self._notify(EventTopics.DOWNLOAD_PROGRESS, step="finalize", message="Memindahkan dataset ke lokasi final",
                     progress=5, total_steps=5, current_step=5, status="info")
            self.logger.info(f"🔄 Memindahkan dataset ke lokasi final: {output_path}")
            shutil.move(str(temp_download_path), str(output_path))
            
            # Prepare response
            stats = self.validator.get_dataset_stats(output_dir)
            elapsed_time = time.time() - start_time
            
            # Send completion notification
            self._notify(EventTopics.DOWNLOAD_COMPLETE, message=f"Download dataset selesai: {stats.get('total_images', 0)} gambar",
                     duration=elapsed_time, status="success")
            
            self.logger.success(f"✅ Dataset {workspace}/{project}:{version} berhasil didownload ke {output_dir} "
                              f"({elapsed_time:.1f}s)\n"
                              f"   • Ukuran: {file_size_mb:.2f} MB\n"
                              f"   • Gambar: {stats.get('total_images', 0)} file\n"
                              f"   • Label: {stats.get('total_labels', 0)} file")
            
            # Prepare result dictionary
            result = {"status": "success", "workspace": workspace, "project": project, "version": version,
                    "format": format, "output_dir": output_dir, "stats": stats, "duration": elapsed_time}
            
            # Add backup info if applicable
            if backup_existing and 'backup_result' in locals():
                result['backup_path'] = backup_result.get('backup_dir')
                
            return result
            
        except Exception as e:
            # Handle errors
            self._notify(EventTopics.DOWNLOAD_ERROR, message=f"Error download dataset: {str(e)}", status="error")
            if temp_download_path.exists(): shutil.rmtree(temp_download_path)
            self.logger.error(f"❌ Error download dataset: {str(e)}")
            raise DatasetError(f"Error download dataset: {str(e)}")
    
    def export_to_local(self, source_dir: Union[str, Path], output_dir: Optional[Union[str, Path]] = None,
                       show_progress: bool = True, backup_existing: bool = True) -> Dict[str, Any]:
        """Export dataset dari format Roboflow ke struktur folder lokal standar."""
        start_time = time.time()
        self._notify(EventTopics.EXPORT_START, message="Memulai ekspor dataset ke struktur lokal",
                 progress=0, total_steps=3, current_step=0, status="info")
        self.logger.info(f"📤 Mengexport dataset ke struktur folder lokal...")
        
        # Validate paths
        src_path = Path(source_dir)
        if not src_path.exists(): raise DatasetError(f"❌ Direktori sumber tidak ditemukan: {src_path}")
        dst_path = Path(output_dir) if output_dir else self.data_dir
        
        # Backup existing data if needed
        backup_path = None
        if backup_existing and dst_path.exists() and any(dst_path.iterdir()):
            self._notify(EventTopics.EXPORT_PROGRESS, step="backup", message="Backup data sebelumnya",
                     progress=1, total_steps=3, current_step=1, status="info")
            backup_result = self.backup_service.backup_dataset(dst_path, show_progress=show_progress)
            backup_path = backup_result['backup_dir']
        
        # Export the dataset
        self._notify(EventTopics.EXPORT_PROGRESS, step="export", message=f"Mengekspor dataset ke {dst_path}",
                 progress=2, total_steps=3, current_step=2, status="info")
        result = self.processor.export_to_local(src_path, dst_path, show_progress, self.num_workers)
        
        # Verify the exported dataset
        self._notify(EventTopics.EXPORT_PROGRESS, step="verify", message="Memvalidasi hasil ekspor",
                 progress=3, total_steps=3, current_step=3, status="info")
        valid = self.validator.verify_local_dataset(dst_path)
        
        # Prepare response
        elapsed_time = time.time() - start_time
        self._notify(EventTopics.EXPORT_COMPLETE, message=f"Ekspor dataset selesai: {result['copied']} file",
                 duration=elapsed_time, status="success" if valid else "warning")
        
        # Log appropriate message based on validation result
        if valid:
            self.logger.success(f"✅ Dataset berhasil diexport ({elapsed_time:.1f}s):\n"
                              f"   • Files: {result['copied']} file\n"
                              f"   • Errors: {result['errors']} error\n"
                              f"   • Output: {dst_path}")
        else:
            self.logger.warning(f"⚠️ Dataset berhasil diexport tetapi validasi gagal ({elapsed_time:.1f}s):\n"
                              f"   • Files: {result['copied']} file\n"
                              f"   • Errors: {result['errors']} error\n"
                              f"   • Output: {dst_path}")
        
        # Prepare result dictionary
        result.update({
            'paths': {split: str(dst_path / split) for split in DEFAULT_SPLITS},
            'output_dir': str(dst_path),
            'backup_path': backup_path,
            'duration': elapsed_time,
            'status': 'success' if valid else 'warning'
        })
        
        return result
    
    def process_zip_file(self, zip_path: Union[str, Path], output_dir: Union[str, Path], extract_only: bool = False,
                        validate_after: bool = True, remove_zip: bool = False, show_progress: bool = True) -> Dict[str, Any]:
        """Proses lengkap file ZIP: ekstrak, restrukturisasi, dan validasi."""
        # Delegate to file_processor
        return self.processor.process_zip_file(
            zip_path=zip_path,
            output_dir=output_dir,
            extract_only=extract_only,
            remove_zip=remove_zip,
            show_progress=show_progress
        )
    
    def pull_dataset(self, format: str = "yolov5pytorch", api_key: Optional[str] = None, workspace: Optional[str] = None,
                   project: Optional[str] = None, version: Optional[str] = None, show_progress: bool = True,
                   force_download: bool = False, backup_existing: bool = True) -> Dict[str, Any]:
        """One-step untuk download dan setup dataset siap pakai."""
        start_time = time.time()
        self._notify(EventTopics.PULL_DATASET_START, message="Memulai persiapan dataset", status="info")
        
        # Check if dataset already exists
        if not force_download and self.validator.is_dataset_available(self.data_dir, verify_content=True):
            self.logger.info("✅ Dataset sudah tersedia di lokal")
            self._notify(EventTopics.PULL_DATASET_COMPLETE, message="Dataset sudah tersedia di lokal", status="success")
            stats = self.validator.get_local_stats(self.data_dir)
            return {
                'status': 'local',
                'paths': {split: str(self.data_dir / split) for split in DEFAULT_SPLITS},
                'data_dir': str(self.data_dir),
                'stats': stats,
                'duration': 0
            }
        
        # Dataset needs to be downloaded
        self.logger.info("🔄 " + ("Memulai download ulang dataset dari Roboflow..." if force_download else 
                                "Dataset belum tersedia atau tidak lengkap, mendownload dari Roboflow..."))
        
        try:
            # Download and export the dataset
            download_result = self.download_from_roboflow(
                api_key=api_key, workspace=workspace, project=project, version=version, format=format,
                show_progress=show_progress, backup_existing=backup_existing)
            
            export_result = self.export_to_local(
                download_result.get('output_dir'), self.data_dir, show_progress=show_progress, 
                backup_existing=backup_existing)
            
            # Prepare response
            elapsed_time = time.time() - start_time
            self._notify(EventTopics.PULL_DATASET_COMPLETE, 
                     message=f"Dataset siap digunakan: {export_result.get('copied', 0)} file",
                     duration=elapsed_time, status="success")
            
            return {
                'status': 'downloaded',
                'paths': export_result['paths'],
                'data_dir': str(self.data_dir),
                'download_dir': download_result.get('output_dir'),
                'export_result': export_result,
                'download_result': download_result,
                'stats': self.validator.get_local_stats(self.data_dir),
                'duration': elapsed_time
            }
            
        except Exception as e:
            # Handle errors
            self._notify(EventTopics.PULL_DATASET_ERROR, 
                     message=f"Error saat persiapan dataset: {str(e)}", status="error")
            self.logger.error(f"❌ Error saat pull dataset: {str(e)}")
            
            # Check if partial dataset is available
            if self.validator.is_dataset_available(self.data_dir, verify_content=False):
                self.logger.warning("⚠️ Download gagal tetapi dataset masih tersedia di lokal (mungkin tidak lengkap)")
                stats = self.validator.get_local_stats(self.data_dir)
                return {
                    'status': 'partial',
                    'paths': {split: str(self.data_dir / split) for split in DEFAULT_SPLITS},
                    'data_dir': str(self.data_dir),
                    'stats': stats,
                    'error': str(e)
                }
            
            raise DatasetError(f"Error pull dataset: {str(e)}")
    
    def import_from_zip(self, zip_file: Union[str, Path], target_dir: Optional[Union[str, Path]] = None,
                      remove_zip: bool = False, show_progress: bool = True, backup_existing: bool = True) -> Dict[str, Any]:
        """Import dataset dari file ZIP."""
        start_time = time.time()
        zip_path, target_path = Path(zip_file), Path(target_dir) if target_dir else self.data_dir
        
        self._notify(EventTopics.ZIP_IMPORT_START, message="Memulai import dataset dari file ZIP",
                 zip_file=str(zip_path), status="info")
        self.logger.info(f"📦 Mengimport dataset dari {zip_path} ke {target_path}")
        
        try:
            # Backup existing data if needed
            backup_path = None
            if backup_existing and target_path.exists() and any(target_path.iterdir()):
                backup_result = self.backup_service.backup_dataset(target_path, show_progress=show_progress)
                backup_path = backup_result.get('backup_dir')
            
            # Process the ZIP file
            result = self.processor.process_zip_file(
                zip_path=zip_path, output_dir=target_path, extract_only=False,
                remove_zip=remove_zip, show_progress=show_progress)
            
            # Prepare response
            elapsed_time = time.time() - start_time
            result['elapsed_time'] = elapsed_time
            if backup_path: result['backup_path'] = backup_path
            
            self._notify(EventTopics.ZIP_IMPORT_COMPLETE,
                     message=f"Import dataset selesai: {result.get('stats', {}).get('total_images', 0)} gambar",
                     duration=elapsed_time, status="success")
            
            return result
            
        except Exception as e:
            # Handle errors
            self._notify(EventTopics.ZIP_IMPORT_ERROR, message=f"Error saat import dataset: {str(e)}", status="error")
            self.logger.error(f"❌ Error saat import dataset: {str(e)}")
            raise DatasetError(f"Error saat import dataset: {str(e)}")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Dapatkan informasi dataset dari konfigurasi dan status lokal."""
        is_available = self.validator.is_dataset_available(self.data_dir)
        local_stats = self.validator.get_local_stats(self.data_dir) if is_available else {}
        
        info = {
            'name': self.project,
            'workspace': self.workspace,
            'version': self.version,
            'is_available_locally': is_available,
            'local_stats': local_stats,
            'data_dir': str(self.data_dir),
            'has_api_key': bool(self.api_key)
        }
        
        # Log appropriate information
        if is_available:
            total_images = sum(local_stats.get(split, 0) for split in DEFAULT_SPLITS)
            self.logger.info(f"🔍 Dataset (Lokal): {info['name']} v{info['version']} | "
                           f"Total: {total_images} gambar | "
                           f"Train: {local_stats.get('train', 0)}, "
                           f"Valid: {local_stats.get('valid', 0)}, "
                           f"Test: {local_stats.get('test', 0)}")
        else:
            self.logger.info(f"🔍 Dataset (akan didownload): {info['name']} "
                           f"v{info['version']} dari {info['workspace']}")
        
        if not info['has_api_key']: self.logger.warning(f"⚠️ API key untuk Roboflow tidak tersedia")
        
        return info
    
    def check_dataset_structure(self) -> Dict[str, Any]:
        """Periksa struktur dataset dan tampilkan laporan."""
        result = {'data_dir': str(self.data_dir), 'is_valid': False, 'splits': {}}
        self.logger.info(f"🔍 Memeriksa struktur dataset di {self.data_dir}...")
        
        # Check structure for each split
        for split in DEFAULT_SPLITS:
            split_dir = self.data_dir / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            split_info = {
                'exists': split_dir.exists(),
                'has_images_dir': images_dir.exists(),
                'has_labels_dir': labels_dir.exists(),
                'image_count': 0,
                'label_count': 0,
                'valid': False
            }
            
            if split_info['has_images_dir'] and split_info['has_labels_dir']:
                image_files = list(images_dir.glob('*.*'))
                label_files = list(labels_dir.glob('*.txt'))
                
                split_info['image_count'] = len(image_files)
                split_info['label_count'] = len(label_files)
                split_info['valid'] = split_info['image_count'] > 0 and split_info['label_count'] > 0
                
                self.logger.info(f"   • {split}: {'✅' if split_info['valid'] else '❌'} "
                               f"({split_info['image_count']} gambar, {split_info['label_count']} label)")
            else:
                self.logger.warning(f"   • {split}: ❌ Struktur direktori tidak lengkap")
            
            result['splits'][split] = split_info
        
        # Check overall validity
        result['is_valid'] = all(info['valid'] for info in result['splits'].values())
        
        if result['is_valid']:
            self.logger.success(f"✅ Struktur dataset valid di {self.data_dir}")
        else:
            self.logger.warning(f"⚠️ Struktur dataset tidak valid atau tidak lengkap di {self.data_dir}")
        
        return result