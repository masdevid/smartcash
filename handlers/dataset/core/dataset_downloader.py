"""
File: smartcash/handlers/dataset/core/dataset_downloader.py
Author: Alfrida Sabar
Deskripsi: Downloader dataset terintegrasi dengan dukungan threading, progress tracking
           dan verifikasi hasil download
"""

import os, json, time, hashlib, threading, zipfile, dotenv
import requests, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.utils.logger import get_logger
from smartcash.utils.observer import EventDispatcher, EventTopics
from smartcash.utils.config_manager import ConfigManager
from smartcash.utils.dataset.dataset_utils import DEFAULT_SPLITS
from smartcash.handlers.dataset.core.roboflow_downloader import RoboflowDownloader
from smartcash.handlers.dataset.core.download_validator import DownloadValidator
from smartcash.handlers.dataset.core.file_processor import FileProcessor


class DatasetDownloader:
    """Downloader dataset dari berbagai sumber dengan dukungan paralel dan progress tracking."""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        config_file: Optional[str] = None,
        logger = None,
        num_workers: int = 4,
        chunk_size: int = 8192,
        retry_limit: int = 3
    ):
        """Inisialisasi DatasetDownloader."""
        self.logger = logger or get_logger("dataset_downloader")
        dotenv.load_dotenv()
        
        # Setup paths dan konfigurasi
        if config is None and config_file is not None:
            self.config = ConfigManager.load_config(
                filename=str(config_file), 
                fallback_to_pickle=True,
                logger=self.logger
            )
        else:
            self.config = config or {}
            
        self.data_dir = Path(data_dir if data_dir else self.config.get('data_dir', 'data'))
        self.temp_dir = self.data_dir / ".temp"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Download parameters
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.retry_limit = retry_limit
        self.retry_delay = 1.0
        self.timeout = 30
        
        # Roboflow settings
        rf_config = self.config.get('data', {}).get('roboflow', {})
        self.api_key = api_key or rf_config.get('api_key') or os.getenv("ROBOFLOW_API_KEY")
        self.workspace = rf_config.get('workspace', 'smartcash-wo2us')
        self.project = rf_config.get('project', 'rupiah-emisi-2022')
        self.version = rf_config.get('version', '3')
        
        # Inisialisasi komponen
        self.rf_downloader = RoboflowDownloader(
            logger=self.logger, 
            timeout=self.timeout,
            chunk_size=self.chunk_size,
            retry_limit=self.retry_limit
        )
        
        self.validator = DownloadValidator(
            logger=self.logger
        )
        
        self.file_processor = FileProcessor(
            logger=self.logger,
            num_workers=self.num_workers
        )
        
        if not self.api_key:
            self.logger.warning("⚠️ Roboflow API key tidak ditemukan. Fitur download mungkin tidak berfungsi.")
    
    def download_dataset(
        self,
        format: str = "yolov5pytorch",
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[str] = None,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
        verify_integrity: bool = True
    ) -> str:
        """
        Download dataset dari Roboflow.
        
        Args:
            format: Format dataset ('yolov5pytorch', 'coco', etc)
            api_key: Roboflow API key
            workspace: Roboflow workspace
            project: Roboflow project
            version: Roboflow version
            output_dir: Directory untuk menyimpan dataset
            show_progress: Tampilkan progress bar
            verify_integrity: Verifikasi integritas dataset setelah download
            
        Returns:
            Path ke direktori dataset
        """
        # Gunakan nilai config jika parameter tidak diberikan
        api_key = api_key or self.api_key
        workspace = workspace or self.workspace
        project = project or self.project
        version = version or self.version
        output_dir = output_dir or os.path.join(self.data_dir, f"roboflow_{workspace}_{project}_{version}")
        
        # Validasi
        if not api_key:
            raise ValueError("🔑 API key tidak tersedia. Berikan api_key melalui parameter atau config.")
        if not workspace or not project or not version:
            raise ValueError("📋 Workspace, project, dan version diperlukan untuk download dataset.")
        
        # Hapus download sebelumnya jika ada
        self.file_processor.clean_existing_download(output_dir)
        
        # Buat directory jika belum ada
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Notifikasi start download
        self.logger.info(f"🚀 Memulai download dataset {workspace}/{project} versi {version}")
        EventDispatcher.notify(
            event_type=EventTopics.DOWNLOAD_START,
            sender=self,
            workspace=workspace,
            project=project,
            version=version
        )
        
        try:
            # Dapatkan metadata terlebih dahulu
            metadata = self.rf_downloader.get_roboflow_metadata(
                workspace, project, version, api_key, format, self.temp_dir
            )
            
            # Verifikasi dan dapatkan link download sebenarnya
            download_url = metadata['export']['link']
            file_size_mb = metadata['export'].get('size', 0)
            
            if file_size_mb > 0:
                self.logger.info(f"📦 Ukuran dataset: {file_size_mb:.2f} MB")
            
            # Lakukan download dataset
            if show_progress:
                download_success = self.rf_downloader.process_roboflow_download(
                    download_url, output_path, show_progress=True
                )
                if not download_success:
                    raise ValueError("Proses download dan ekstraksi gagal")
            else:
                # Jika tidak memerlukan progress tracking,
                # Gunakan Roboflow API jika tersedia
                try:
                    from roboflow import Roboflow
                    rf = Roboflow(api_key=api_key)
                    workspace_obj = rf.workspace(workspace)
                    project_obj = workspace_obj.project(project)
                    version_obj = project_obj.version(version)
                    dataset = version_obj.download(model_format=format, location=output_dir)
                except ImportError:
                    # Jika Roboflow tidak tersedia, gunakan HTTP request
                    download_success = self.rf_downloader.process_roboflow_download(
                        download_url, output_path, show_progress=False
                    )
                    if not download_success:
                        raise ValueError("Proses download dan ekstraksi gagal")
            
            # Verifikasi hasil download
            if verify_integrity:
                if not self.validator.verify_download(output_dir, metadata):
                    self.logger.warning("⚠️ Verifikasi dataset gagal, namun download selesai")
            
            self.logger.success(f"✅ Dataset {workspace}/{project}:{version} berhasil didownload ke {output_dir}")
            
            # Notifikasi download selesai
            EventDispatcher.notify(
                event_type=EventTopics.DOWNLOAD_COMPLETE,
                sender=self,
                workspace=workspace,
                project=project,
                version=version,
                output_dir=output_dir,
                stats=self.validator.get_dataset_stats(output_dir)
            )
            
            return output_dir
            
        except Exception as e:
            self.logger.error(f"❌ Error download dataset: {str(e)}")
            EventDispatcher.notify(
                event_type=EventTopics.DOWNLOAD_ERROR,
                sender=self,
                error=str(e)
            )
            raise
    
    def export_to_local(self, roboflow_dir: Union[str, Path], show_progress: bool = True) -> Tuple[str, str, str]:
        """Export dataset Roboflow ke struktur folder lokal standar."""
        self.logger.start("📤 Mengexport dataset ke struktur folder lokal...")
        rf_path = Path(roboflow_dir)
        
        # Verifikasi sumber ada
        if not rf_path.exists():
            raise FileNotFoundError(f"❌ Direktori sumber tidak ditemukan: {rf_path}")
        
        # Proses export
        result = self.file_processor.export_to_local(
            rf_path, self.data_dir, show_progress, self.num_workers
        )
        
        # Verifikasi hasil export
        self.validator.verify_local_dataset(self.data_dir)
            
        # Return paths
        output_dirs = tuple(str(self.data_dir / split) for split in DEFAULT_SPLITS)
        self.logger.success(f"✅ Dataset berhasil diexport: {result['copied']} file → {self.data_dir}")
        return output_dirs
            
    def pull_dataset(
        self, 
        format: str = "yolov5pytorch", 
        show_progress: bool = True,
        api_key: str = None, 
        workspace: str = None, 
        project: str = None, 
        version: str = None
    ) -> tuple:
        """One-step untuk download dan setup dataset siap pakai."""
        try:
            # Override parameter sementara jika diberikan
            old_values = {}
            for param in ['api_key', 'workspace', 'project', 'version']:
                val = locals()[param]
                if val is not None:
                    old_values[param] = getattr(self, param)
                    setattr(self, param, val)
            
            # Jika dataset sudah ada dan lengkap, gunakan itu
            if self.validator.is_dataset_available(self.data_dir, verify_content=True):
                self.logger.info("✅ Dataset sudah tersedia di lokal")
                return tuple(str(self.data_dir / split) for split in DEFAULT_SPLITS)
            
            # Download dari Roboflow dan export ke lokal
            self.logger.info("🔄 Dataset belum tersedia atau tidak lengkap, mendownload dari Roboflow...")
            roboflow_dir = self.download_dataset(
                format=format, 
                api_key=api_key, 
                workspace=workspace, 
                project=project, 
                version=version, 
                show_progress=show_progress
            )
            return self.export_to_local(roboflow_dir, show_progress)
        except Exception as e:
            self.logger.error(f"❌ Gagal pull dataset: {str(e)}")
            raise
        finally:
            # Kembalikan nilai semula
            for param, val in old_values.items():
                setattr(self, param, val)
    
    def import_from_zip(
        self, 
        zip_path: Union[str, Path], 
        target_dir: Optional[Union[str, Path]] = None,
        format: str = "yolov5pytorch"
    ) -> str:
        """Import dataset dari file zip."""
        self.logger.info(f"📦 Importing dataset dari {zip_path}...")
        
        # Verifikasi file zip
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"❌ File zip tidak ditemukan: {zip_path}")
            
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"❌ File bukan format ZIP yang valid: {zip_path}")
            
        # Extract zip
        target_dir = Path(target_dir) if target_dir else self.data_dir
        extract_dir = target_dir / zip_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Lakukan ekstraksi
        self.file_processor.extract_zip(zip_path, extract_dir, remove_zip=False, show_progress=True)
        
        # Fix struktur dataset jika perlu
        valid_structure = self.file_processor.fix_dataset_structure(extract_dir)
        if not valid_structure:
            self.logger.warning(f"⚠️ Struktur dataset tidak sesuai format YOLOv5")
        
        # Verifikasi hasil import
        if not self.validator.verify_dataset_structure(extract_dir):
            self.logger.warning(f"⚠️ Dataset tidak lengkap setelah import")
        else:
            self.logger.success(f"✅ Dataset berhasil diimport ke {extract_dir}")
        
        # Copy ke data_dir jika berbeda
        if extract_dir != self.data_dir:
            self.logger.info(f"🔄 Menyalin dataset ke {self.data_dir}...")
            self.file_processor.copy_dataset_to_data_dir(extract_dir, self.data_dir)
            
        return str(extract_dir)
    
    def get_dataset_info(self) -> Dict:
        """Mendapatkan informasi dataset dari konfigurasi dan status lokal."""
        is_available = self.validator.is_dataset_available(self.data_dir)
        local_stats = self.validator.get_local_stats(self.data_dir) if is_available else {}
        
        info = {
            'name': self.project,
            'workspace': self.workspace,
            'version': self.version,
            'is_available_locally': is_available,
            'local_stats': local_stats
        }
        
        # Log informasi
        if is_available:
            self.logger.info(f"🔍 Dataset (Lokal): {info['name']} v{info['version']} | "
                            f"Train: {local_stats.get('train', 0)}, Valid: {local_stats.get('valid', 0)}, Test: {local_stats.get('test', 0)}")
        else:
            self.logger.info(f"🔍 Dataset (akan didownload): {info['name']} v{info['version']} dari {info['workspace']}")
        
        return info