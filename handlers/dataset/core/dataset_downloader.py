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
    
    def __init__(self,config: Optional[Dict] = None,data_dir: Optional[str] = None,api_key: Optional[str] = None,config_file: Optional[str] = None,logger = None,num_workers: int = 4,chunk_size: int = 8192,retry_limit: int = 3):
        """Inisialisasi DatasetDownloader."""
        self.logger = logger or get_logger("dataset_downloader")
        dotenv.load_dotenv()
        
        # Setup paths dan konfigurasi
        self.config = (config if config is not None else 
                      ConfigManager.load_config(str(config_file), True, self.logger) if config_file else {})
        
        self.data_dir = Path(data_dir or self.config.get('data_dir', 'data'))
        self.temp_dir = self.data_dir / ".temp"
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        # Download parameters
        self.num_workers, self.chunk_size, self.retry_limit = num_workers, chunk_size, retry_limit
        self.retry_delay, self.timeout = 1.0, 30

        # Roboflow settings
        rf_config = self.config.get('data', {}).get('roboflow', {})
        self.api_key = api_key or rf_config.get('api_key') or os.getenv("ROBOFLOW_API_KEY")
        for attr in ['workspace', 'project', 'version']:
            setattr(self, attr, rf_config.get(attr, 'smartcash-wo2us' if attr == 'workspace' else 
                                              'rupiah-emisi-2022' if attr == 'project' else '3'))

        # Inisialisasi komponen
        self.rf_downloader = RoboflowDownloader(logger=self.logger, timeout=self.timeout,chunk_size=self.chunk_size, retry_limit=self.retry_limit)
        self.validator = DownloadValidator(logger=self.logger)
        self.file_processor = FileProcessor(logger=self.logger, num_workers=self.num_workers)
        
        if not self.api_key:
            self.logger.warning("⚠️ Roboflow API key tidak ditemukan. Fitur download mungkin tidak berfungsi.")

    def _notify_event(self, event_type: str, **kwargs) -> None:
        """Helper untuk event notification."""
        EventDispatcher.notify(event_type=event_type, sender=self, **kwargs)

    def download_dataset(self,format: str = "yolov5pytorch",api_key: Optional[str] = None,workspace: Optional[str] = None,project: Optional[str] = None,version: Optional[str] = None,output_dir: Optional[str] = None,show_progress: bool = True,verify_integrity: bool = True) -> str:
        """Download dataset dari Roboflow."""
        params = {k: (locals().get(k) or getattr(self, k)) for k in ['api_key', 'workspace', 'project', 'version']}
        output_dir = output_dir or os.path.join(self.data_dir, f"roboflow_{params['workspace']}_{params['project']}_{params['version']}")
        
        if not params['api_key']:
            raise ValueError("🔑 API key tidak tersedia. Berikan api_key melalui parameter atau config.")
        if not all(params.values()):
            raise ValueError("📋 Workspace, project, dan version diperlukan untuk download dataset.")
        
        self.file_processor.clean_existing_download(output_dir)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"🚀 Memulai download dataset {params['workspace']}/{params['project']} versi {params['version']}")
        self._notify_event(EventTopics.DOWNLOAD_START, **params)

        try:
            metadata = self.rf_downloader.get_roboflow_metadata(
                params['workspace'], params['project'], params['version'], params['api_key'], 
                format, self.temp_dir
            )
            download_url, file_size_mb = metadata['export']['link'], metadata['export'].get('size', 0)
            
            if file_size_mb > 0:
                self.logger.info(f"📦 Ukuran dataset: {file_size_mb:.2f} MB")

            if show_progress:
                if not self.rf_downloader.process_roboflow_download(download_url, output_dir, show_progress=True):
                    raise ValueError("Proses download dan ekstraksi gagal")
            else:
                try:  # Prioritize Roboflow SDK
                    from roboflow import Roboflow
                    version_obj = Roboflow(api_key=params['api_key']).workspace(params['workspace']
                        ).project(params['project']).version(params['version'])
                    version_obj.download(model_format=format, location=output_dir)
                except (ImportError, AttributeError):
                    if not self.rf_downloader.process_roboflow_download(download_url, output_dir, show_progress=False):
                        raise ValueError("Proses download dan ekstraksi gagal")

            if verify_integrity and not self.validator.verify_download(output_dir, metadata):
                self.logger.warning("⚠️ Verifikasi dataset gagal, namun download selesai")
            
            self.logger.success(f"✅ Dataset {params['workspace']}/{params['project']}:{params['version']} berhasil didownload ke {output_dir}")
            stats = self.validator.get_dataset_stats(output_dir)
            self._notify_event(EventTopics.DOWNLOAD_COMPLETE, output_dir=output_dir, stats=stats, **params)
            return output_dir
            
        except Exception as e:
            self.logger.error(f"❌ Error download dataset: {str(e)}")
            self._notify_event(EventTopics.DOWNLOAD_ERROR, error=str(e))
            raise

    def export_to_local(self, roboflow_dir: Union[str, Path], show_progress: bool = True) -> Tuple[str, str, str]:
        """Export dataset Roboflow ke struktur folder lokal standar."""
        self.logger.start("📤 Mengexport dataset ke struktur folder lokal...")
        if not (rf_path := Path(roboflow_dir)).exists():
            raise FileNotFoundError(f"❌ Direktori sumber tidak ditemukan: {rf_path}")

        self.file_processor.export_to_local(rf_path, self.data_dir, show_progress, self.num_workers)
        self.validator.verify_local_dataset(self.data_dir)
            
        # Explicit Path conversion for output directories
        output_dirs = tuple(str(Path(self.data_dir) / split) for split in DEFAULT_SPLITS)
        self.logger.success(f"✅ Dataset berhasil diexport: {len(list(self.data_dir.glob('**/*')))} file → {self.data_dir}")
        return output_dirs
            
    def pull_dataset(
        self, 
        format: str = "yolov5pytorch", 
        show_progress: bool = True,
        **kwargs
    ) -> tuple:
        """One-step untuk download dan setup dataset siap pakai."""
        try:
            old_values = {k: getattr(self, k) for k in kwargs.keys() if hasattr(self, k)}
            for k, v in kwargs.items(): 
                if hasattr(self, k): setattr(self, k, v)

            if self.validator.is_dataset_available(self.data_dir, verify_content=True):
                self.logger.info("✅ Dataset sudah tersedia di lokal")
                return tuple(str(Path(self.data_dir) / split) for split in DEFAULT_SPLITS)
            
            self.logger.info("🔄 Dataset belum tersedia atau tidak lengkap, mendownload dari Roboflow...")
            return self.export_to_local(
                self.download_dataset(format=format, show_progress=show_progress, **kwargs), 
                show_progress
            )
        finally:
            for k, v in old_values.items(): setattr(self, k, v)

    def import_from_zip(
        self, 
        zip_path: Union[str, Path], 
        target_dir: Optional[Union[str, Path]] = None,
        format: str = "yolov5pytorch"
    ) -> str:
        """Import dataset dari file zip."""
        self.logger.info(f"📦 Importing dataset dari {zip_path}...")
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"❌ File zip tidak ditemukan: {zip_path}")
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"❌ File bukan format ZIP yang valid: {zip_path}")

        extract_dir = Path(target_dir or self.data_dir) / zip_path.stem
        self.file_processor.extract_zip(zip_path, extract_dir, remove_zip=False, show_progress=True)
        
        if not self.file_processor.fix_dataset_structure(extract_dir):
            self.logger.warning("⚠️ Struktur dataset tidak sesuai format YOLOv5")
        if not self.validator.verify_dataset_structure(extract_dir):
            self.logger.warning("⚠️ Dataset tidak lengkap setelah import")
        else:
            self.logger.success(f"✅ Dataset berhasil diimport ke {extract_dir}")

        if extract_dir != self.data_dir:
            self.logger.info(f"🔄 Menyalin dataset ke {self.data_dir}...")
            self.file_processor.copy_dataset_to_data_dir(extract_dir, self.data_dir)
            
        return str(extract_dir)

    def get_dataset_info(self) -> Dict:
        """Mendapatkan informasi dataset dari konfigurasi dan status lokal."""
        is_available = self.validator.is_dataset_available(self.data_dir)
        info = {
            'name': self.project,
            'workspace': self.workspace,
            'version': self.version,
            'is_available_locally': is_available,
            'local_stats': self.validator.get_local_stats(self.data_dir) if is_available else {}
        }
        
        stats = info['local_stats']
        log_msg = (f"🔍 Dataset (Lokal): {info['name']} v{info['version']} | "
                  f"Train: {stats.get('train',0)}, Valid: {stats.get('valid',0)}, Test: {stats.get('test',0)}"
                  ) if is_available else f"🔍 Dataset (akan didownload): {info['name']} v{info['version']} dari {info['workspace']}"
        self.logger.info(log_msg)
        
        return info