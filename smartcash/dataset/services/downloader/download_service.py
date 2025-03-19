"""
File: smartcash/dataset/services/downloader/download_service.py
Deskripsi: Layanan utama untuk mengelola download dataset dari berbagai sumber
"""
import os
import json
import time
import hashlib
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.common.layer_config import get_layer_config
from smartcash.dataset.utils.dataset_utils import DatasetUtils, DEFAULT_SPLITS

# Sisa kode tidak berubah


class DownloadService:
    """
    Layanan untuk mendownload dan menyiapkan dataset dari berbagai sumber.
    Mendukung download dari Roboflow, import dari zip, dan validasi dataset.
    """
    
    def __init__(self, config: Dict, data_dir: str, logger=None, num_workers: int = 4):
        """
        Inisialisasi DownloadService.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori data
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
        """
        self.config = config
        self.data_dir = Path(data_dir)
        self.logger = logger or get_logger("download_service")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(config, data_dir, logger)
        
        # Setup direktori temp dan downloads
        self.temp_dir = self.data_dir / ".temp"
        self.downloads_dir = self.data_dir / "downloads"
        
        # Buat direktori jika belum ada
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.downloads_dir, exist_ok=True)
        
        # Ambil API key dari config atau environment
        rf_config = self.config.get('data', {}).get('roboflow', {})
        self.api_key = rf_config.get('api_key') or os.environ.get("ROBOFLOW_API_KEY")
        self.workspace = rf_config.get('workspace', 'smartcash-wo2us')
        self.project = rf_config.get('project', 'rupiah-emisi-2022')
        self.version = rf_config.get('version', '3')
        
        # Inisialisasi komponen-komponen
        self._init_components()
        
        self.logger.info(
            f"📥 DownloadService diinisialisasi dengan {num_workers} workers\n"
            f"   • Data dir: {self.data_dir}\n"
            f"   • Default sumber: {self.workspace}/{self.project}:{self.version}"
        )
    
    def export_to_local(
        self,
        source_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True,
        clear_target: bool = True  # Tambahkan parameter untuk membersihkan direktori target
    ) -> Dict[str, Any]:
        """
        Export dataset dari format Roboflow ke struktur folder lokal standar.
        
        Args:
            source_dir: Direktori sumber (hasil download dari Roboflow)
            output_dir: Direktori tujuan (opsional, default ke data_dir)
            show_progress: Tampilkan progress bar
            clear_target: Hapus file yang sudah ada di direktori target
            
        Returns:
            Dictionary berisi hasil ekspor
        """
        self.logger.info(f"📤 Mengexport dataset ke struktur folder lokal...")
        
        # Normalisasi path
        src_path = Path(source_dir)
        if not src_path.exists():
            raise FileNotFoundError(f"❌ Direktori sumber tidak ditemukan: {src_path}")
                
        # Setup output path
        dst_path = Path(output_dir) if output_dir else self.data_dir
        
        # Proses export
        result = self.processor.export_to_local(
            src_path, dst_path, show_progress, self.num_workers, clear_target
        )
        
        # Verifikasi hasil export
        valid = self.validator.verify_local_dataset(dst_path)
        
        if valid:
            self.logger.success(
                f"✅ Dataset berhasil diexport:\n"
                f"   • Files: {result['copied']} file\n"
                f"   • Errors: {result['errors']} error\n"
                f"   • Output: {dst_path}"
            )
        else:
            self.logger.warning(
                f"⚠️ Dataset berhasil diexport tetapi validasi gagal:\n"
                f"   • Files: {result['copied']} file\n"
                f"   • Errors: {result['errors']} error\n"
                f"   • Output: {dst_path}"
            )
        
        # Return tuple dari path untuk setiap split
        result['paths'] = {split: str(dst_path / split) for split in DEFAULT_SPLITS}
        result['output_dir'] = str(dst_path)
        
        return result
    
    def import_from_zip(
        self,
        zip_path: Union[str, Path],
        target_dir: Optional[Union[str, Path]] = None,
        remove_zip: bool = False,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Import dataset dari file zip.
        
        Args:
            zip_path: Path ke file zip
            target_dir: Direktori tujuan (opsional)
            remove_zip: Apakah file zip dihapus setelah ekstraksi
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary berisi hasil import
        """
        self.logger.info(f"📦 Importing dataset dari {zip_path}...")
        
        # Verifikasi file zip
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"❌ File zip tidak ditemukan: {zip_path}")
            
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"❌ File bukan format ZIP yang valid: {zip_path}")
            
        # Setup target dir
        target_dir = Path(target_dir) if target_dir else self.data_dir / zip_path.stem
        os.makedirs(target_dir, exist_ok=True)
        
        # Ekstrak zip
        self.processor.extract_zip(
            zip_path, target_dir, remove_zip=remove_zip, show_progress=show_progress
        )
        
        # Fix struktur dataset jika perlu
        valid_structure = self.processor.fix_dataset_structure(target_dir)
        
        if not valid_structure:
            self.logger.warning(f"⚠️ Struktur dataset tidak sesuai format YOLOv5")
        
        # Verifikasi hasil import
        valid = self.validator.verify_dataset_structure(target_dir)
        
        if valid:
            self.logger.success(f"✅ Dataset berhasil diimport ke {target_dir}")
        else:
            self.logger.warning(f"⚠️ Dataset tidak lengkap setelah import ke {target_dir}")
        
        # Jika target dir berbeda dengan data_dir, copy ke data_dir
        if target_dir != self.data_dir:
            self.logger.info(f"🔄 Menyalin dataset ke {self.data_dir}...")
            copy_result = self.processor.copy_dataset_to_data_dir(target_dir, self.data_dir)
            
            self.logger.info(
                f"   • Files disalin: {copy_result['copied']}\n"
                f"   • Errors: {copy_result['errors']}"
            )
        
        # Hitung statistik
        stats = self.validator.get_dataset_stats(target_dir)
        
        return {
            'status': 'success' if valid else 'warning',
            'target_dir': str(target_dir),
            'data_dir': str(self.data_dir),
            'structure_fixed': valid_structure,
            'valid': valid,
            'stats': stats
        }
    
    def pull_dataset(
        self, 
        format: str = "yolov5pytorch", 
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[str] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        One-step untuk download dan setup dataset siap pakai.
        
        Args:
            format: Format dataset ('yolov5pytorch', 'coco', etc)
            api_key: Roboflow API key (opsional)
            workspace: Roboflow workspace (opsional)
            project: Roboflow project (opsional)
            version: Roboflow version (opsional)
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary berisi paths dan info dataset
        """
        # Cek apakah dataset sudah tersedia
        if self.validator.is_dataset_available(self.data_dir, verify_content=True):
            self.logger.info("✅ Dataset sudah tersedia di lokal")
            
            # Return info
            stats = self.validator.get_local_stats(self.data_dir)
            return {
                'status': 'local',
                'paths': {split: str(self.data_dir / split) for split in DEFAULT_SPLITS},
                'data_dir': str(self.data_dir),
                'stats': stats
            }
        
        # Dataset belum tersedia, download dari Roboflow
        self.logger.info("🔄 Dataset belum tersedia atau tidak lengkap, mendownload dari Roboflow...")
        
        # Download dataset
        download_dir = self.download_dataset(
            api_key=api_key,
            workspace=workspace,
            project=project,
            version=version,
            format=format,
            show_progress=show_progress
        )
        
        # Export ke struktur lokal
        export_result = self.export_to_local(download_dir, self.data_dir, show_progress)
        
        # Return info
        return {
            'status': 'downloaded',
            'paths': export_result['paths'],
            'data_dir': str(self.data_dir),
            'download_dir': download_dir,
            'stats': self.validator.get_local_stats(self.data_dir)
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Dapatkan informasi dataset dari konfigurasi dan status lokal.
        
        Returns:
            Dictionary berisi info dataset
        """
        is_available = self.validator.is_dataset_available(self.data_dir)
        local_stats = self.validator.get_local_stats(self.data_dir) if is_available else {}
        
        info = {
            'name': self.project,
            'workspace': self.workspace,
            'version': self.version,
            'is_available_locally': is_available,
            'local_stats': local_stats,
            'data_dir': str(self.data_dir)
        }
        
        # Log informasi
        if is_available:
            self.logger.info(
                f"🔍 Dataset (Lokal): {info['name']} v{info['version']} | "
                f"Train: {local_stats.get('train', 0)}, "
                f"Valid: {local_stats.get('valid', 0)}, "
                f"Test: {local_stats.get('test', 0)}"
            )
        else:
            self.logger.info(
                f"🔍 Dataset (akan didownload): {info['name']} "
                f"v{info['version']} dari {info['workspace']}"
            )
        
        return info
    
    def _init_components(self) -> None:
        """Inisialisasi komponen-komponen service."""
        # Import di sini untuk menghindari circular import
        from smartcash.dataset.services.downloader.roboflow_downloader import RoboflowDownloader
        from smartcash.dataset.services.downloader.download_validator import DownloadValidator
        from smartcash.dataset.services.downloader.file_processor import FileProcessor
        
        # Inisialisasi komponen
        self.roboflow_downloader = RoboflowDownloader(
            logger=self.logger,
            timeout=30,
            chunk_size=8192,
            retry_limit=3
        )
        
        self.validator = DownloadValidator(
            logger=self.logger
        )
        
        self.processor = FileProcessor(
            logger=self.logger,
            num_workers=self.num_workers
        )