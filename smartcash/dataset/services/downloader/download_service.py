"""
File: smartcash/dataset/services/downloader/download_service.py
Deskripsi: Layanan utama untuk mengelola download dataset dari berbagai sumber dengan alur kerja yang disempurnakan
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
from smartcash.common.exceptions import DatasetError
# Import komponen yang diperlukan di awal untuk menghindari circular import
from smartcash.dataset.services.downloader.roboflow_downloader import RoboflowDownloader
from smartcash.dataset.services.downloader.download_validator import DownloadValidator
from smartcash.dataset.services.downloader.file_processor import FileProcessor


class DownloadService:
    """
    Layanan untuk mendownload dan menyiapkan dataset dari berbagai sumber.
    Mendukung download dari Roboflow, import dari zip, dan validasi dataset.
    """
    
    def __init__(self, output_dir: str = "data", config: Optional[Dict] = None, logger=None, num_workers: int = 4):
        """
        Inisialisasi DownloadService.
        
        Args:
            output_dir: Direktori data output
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk proses paralel
        """
        self.config = config or {}
        self.data_dir = Path(output_dir)
        self.logger = logger or get_logger("download_service")
        self.num_workers = num_workers
        
        # Setup utils
        self.utils = DatasetUtils(self.config, output_dir, logger)
        
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
        
        # Inisialisasi komponen-komponen langsung di constructor
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
        
        self.logger.info(
            f"📥 DownloadService diinisialisasi dengan {num_workers} workers\n"
            f"   • Data dir: {self.data_dir}\n"
            f"   • Default sumber: {self.workspace}/{self.project}:{self.version}"
        )
    
    def download_from_roboflow(
        self,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[str] = None,
        format: str = "yolov5pytorch",
        output_dir: Optional[str] = None,
        show_progress: bool = True,
        verify_integrity: bool = True
    ) -> Dict[str, Any]:
        """
        Download dataset dari Roboflow.
        
        Args:
            api_key: Roboflow API key (opsional, default dari config)
            workspace: Roboflow workspace (opsional)
            project: Roboflow project (opsional)
            version: Roboflow version (opsional)
            format: Format dataset ('yolov5pytorch', 'coco', etc)
            output_dir: Directory untuk menyimpan dataset (opsional)
            show_progress: Tampilkan progress bar
            verify_integrity: Verifikasi integritas dataset setelah download
            
        Returns:
            Dictionary berisi informasi hasil download
        """
        start_time = time.time()
        
        # Gunakan nilai config jika parameter tidak diberikan
        api_key = api_key or self.api_key
        workspace = workspace or self.workspace
        project = project or self.project
        version = version or self.version
        
        # Validasi
        if not api_key:
            raise DatasetError("🔑 API key tidak tersedia. Berikan api_key melalui parameter atau config.")
        if not workspace or not project or not version:
            raise DatasetError("📋 Workspace, project, dan version diperlukan untuk download dataset.")
        
        # Setup direktori output
        if not output_dir:
            output_dir = str(self.downloads_dir / f"{workspace}_{project}_{version}")
            
        output_path = Path(output_dir)
        temp_download_path = output_path.with_name(f"{output_path.name}_temp")
        
        # Buat direktori temporari untuk download
        if temp_download_path.exists():
            shutil.rmtree(temp_download_path)
        temp_download_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download metadata terlebih dahulu
            metadata = self.roboflow_downloader.get_roboflow_metadata(
                workspace, project, version, api_key, format, self.temp_dir
            )
            
            # Dapatkan link download
            if 'export' not in metadata or 'link' not in metadata['export']:
                raise DatasetError("❌ Format metadata tidak valid, tidak ada link download")
                
            download_url = metadata['export']['link']
            file_size_mb = metadata.get('export', {}).get('size', 0)
            
            if file_size_mb > 0:
                self.logger.info(f"📦 Ukuran dataset: {file_size_mb:.2f} MB")
            
            # Lakukan download dataset ke direktori temporari
            download_success = self.roboflow_downloader.process_roboflow_download(
                download_url, temp_download_path, show_progress=show_progress
            )
            
            if not download_success:
                raise DatasetError("❌ Proses download dan ekstraksi gagal")
            
            # Verifikasi hasil download
            if verify_integrity:
                valid = self.validator.verify_download(str(temp_download_path), metadata)
                
                if not valid:
                    self.logger.warning("⚠️ Verifikasi dataset gagal")
            
            # Jika direktori output sudah ada, hapus terlebih dahulu
            if output_path.exists():
                self.logger.info(f"🧹 Menghapus direktori sebelumnya: {output_path}")
                shutil.rmtree(output_path)
            
            # Pindahkan hasil download ke lokasi final
            shutil.move(str(temp_download_path), str(output_path))
            
            # Tampilkan statistik download
            stats = self.validator.get_dataset_stats(output_dir)
            elapsed_time = time.time() - start_time
            
            self.logger.success(
                f"✅ Dataset {workspace}/{project}:{version} berhasil didownload ke {output_dir} ({elapsed_time:.1f}s)\n"
                f"   • Ukuran: {file_size_mb:.2f} MB\n"
                f"   • Gambar: {stats.get('total_images', 0)} file\n"
                f"   • Label: {stats.get('total_labels', 0)} file"
            )
            
            return {
                "status": "success",
                "workspace": workspace,
                "project": project,
                "version": version,
                "format": format,
                "output_dir": output_dir,
                "stats": stats,
                "duration": elapsed_time
            }
            
        except Exception as e:
            # Hapus direktori temporari jika ada error
            if temp_download_path.exists():
                shutil.rmtree(temp_download_path)
                
            self.logger.error(f"❌ Error download dataset: {str(e)}")
            raise DatasetError(f"Error download dataset: {str(e)}")
    
    def export_to_local(
        self,
        source_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Export dataset dari format Roboflow ke struktur folder lokal standar.
        
        Args:
            source_dir: Direktori sumber (hasil download dari Roboflow)
            output_dir: Direktori tujuan (opsional, default ke data_dir)
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary berisi hasil ekspor
        """
        start_time = time.time()
        self.logger.info(f"📤 Mengexport dataset ke struktur folder lokal...")
        
        # Normalisasi path
        src_path = Path(source_dir)
        if not src_path.exists():
            raise DatasetError(f"❌ Direktori sumber tidak ditemukan: {src_path}")
            
        # Setup output path
        dst_path = Path(output_dir) if output_dir else self.data_dir
        
        # Backup direktori tujuan jika sudah ada data
        if dst_path.exists() and any(dst_path.iterdir()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = dst_path.with_name(f"{dst_path.name}_backup_{timestamp}")
            
            try:
                self.logger.info(f"📦 Membuat backup data sebelumnya: {backup_path}")
                shutil.copytree(dst_path, backup_path)
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membuat backup: {str(e)}")
        
        # Proses export
        result = self.processor.export_to_local(
            src_path, dst_path, show_progress, self.num_workers
        )
        
        # Verifikasi hasil export
        valid = self.validator.verify_local_dataset(dst_path)
        elapsed_time = time.time() - start_time
        
        if valid:
            self.logger.success(
                f"✅ Dataset berhasil diexport ({elapsed_time:.1f}s):\n"
                f"   • Files: {result['copied']} file\n"
                f"   • Errors: {result['errors']} error\n"
                f"   • Output: {dst_path}"
            )
        else:
            self.logger.warning(
                f"⚠️ Dataset berhasil diexport tetapi validasi gagal ({elapsed_time:.1f}s):\n"
                f"   • Files: {result['copied']} file\n"
                f"   • Errors: {result['errors']} error\n"
                f"   • Output: {dst_path}"
            )
        
        # Return tuple dari path untuk setiap split
        result['paths'] = {split: str(dst_path / split) for split in DEFAULT_SPLITS}
        result['output_dir'] = str(dst_path)
        result['duration'] = elapsed_time
        result['status'] = 'success' if valid else 'warning'
        
        return result
    
    def process_zip_file(
        self,
        zip_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        extract_only: bool = False,
        validate_after: bool = True
    ) -> Dict[str, Any]:
        """
        Proses file ZIP dataset, ekstrak dan verifikasi.
        
        Args:
            zip_path: Path ke file ZIP
            output_dir: Direktori output untuk ekstraksi
            extract_only: Hanya ekstrak tanpa memproses struktur
            validate_after: Validasi dataset setelah ekstraksi
            
        Returns:
            Dictionary berisi hasil proses
        """
        start_time = time.time()
        self.logger.info(f"📦 Memproses file ZIP dataset: {zip_path}")
        
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise DatasetError(f"❌ File ZIP tidak ditemukan: {zip_path}")
            
        if not zipfile.is_zipfile(zip_path):
            raise DatasetError(f"❌ File bukan format ZIP yang valid: {zip_path}")
        
        # Setup direktori output
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.data_dir / zip_path.stem
            
        os.makedirs(output_path, exist_ok=True)
        
        # Ekstrak file ZIP
        tmp_extract_dir = output_path.with_name(f"{output_path.name}_extract_temp")
        if tmp_extract_dir.exists():
            shutil.rmtree(tmp_extract_dir)
        tmp_extract_dir.mkdir(parents=True)
        
        try:
            # Extract ZIP ke direktori temporari
            self.logger.info(f"📂 Mengekstrak file ZIP ke {tmp_extract_dir}")
            extract_success = self.roboflow_downloader._extract_zip(
                zip_path, tmp_extract_dir, remove_zip=False, show_progress=True
            )
            
            if not extract_success:
                raise DatasetError(f"❌ Gagal mengekstrak file ZIP: {zip_path}")
                
            # Periksa hasil ekstraksi
            extracted_files = list(tmp_extract_dir.glob('**/*'))
            if not extracted_files:
                raise DatasetError(f"❌ Tidak ada file terdeteksi setelah ekstraksi")
                
            self.logger.info(f"✅ Ekstraksi selesai: {len(extracted_files)} file")
            
            # Proses struktur dataset jika diperlukan
            if not extract_only:
                # Fix struktur dataset jika perlu
                self.logger.info(f"🔧 Menyesuaikan struktur dataset...")
                valid_structure = self.processor.fix_dataset_structure(tmp_extract_dir)
                
                # Pindahkan hasil ke direktori final
                self.logger.info(f"🔄 Memindahkan dataset ke {output_path}")
                self.processor.copy_dataset_to_data_dir(tmp_extract_dir, output_path)
            else:
                # Jika hanya ekstrak, langsung pindahkan semua file
                for item in tmp_extract_dir.iterdir():
                    if item.is_file():
                        shutil.copy2(item, output_path)
                    elif item.is_dir():
                        shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
                        
            # Hapus direktori temporari
            shutil.rmtree(tmp_extract_dir)
            
            # Validasi hasil
            if validate_after:
                self.logger.info(f"✅ Memvalidasi struktur dataset...")
                valid = self.validator.verify_dataset_structure(output_path)
                
                if not valid:
                    self.logger.warning(f"⚠️ Struktur dataset tidak sesuai format standar")
            
            # Tampilkan statistik
            stats = self.validator.get_dataset_stats(str(output_path))
            elapsed_time = time.time() - start_time
            
            result = {
                "status": "success",
                "output_dir": str(output_path),
                "file_count": len(extracted_files),
                "stats": stats,
                "duration": elapsed_time
            }
            
            self.logger.success(
                f"✅ Proses file ZIP selesai ({elapsed_time:.1f}s):\n"
                f"   • Total file: {len(extracted_files)}\n"
                f"   • Gambar: {stats.get('total_images', 0)}\n"
                f"   • Label: {stats.get('total_labels', 0)}\n"
                f"   • Output: {output_path}"
            )
            
            return result
            
        except Exception as e:
            # Hapus direktori temporari jika ada error
            if tmp_extract_dir.exists():
                shutil.rmtree(tmp_extract_dir)
                
            self.logger.error(f"❌ Error memproses file ZIP: {str(e)}")
            raise DatasetError(f"Error memproses file ZIP: {str(e)}")
    
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
        # Delegasikan ke method process_zip_file yang lebih robust
        try:
            self.logger.info(f"📦 Importing dataset dari {zip_path}...")
            
            # Verifikasi file zip
            zip_path = Path(zip_path)
            if not zip_path.exists():
                raise DatasetError(f"❌ File zip tidak ditemukan: {zip_path}")
                
            if not zipfile.is_zipfile(zip_path):
                raise DatasetError(f"❌ File bukan format ZIP yang valid: {zip_path}")
                
            # Setup target dir
            target_dir = Path(target_dir) if target_dir else self.data_dir / zip_path.stem
            
            # Panggil method process_zip_file
            result = self.process_zip_file(
                zip_path=zip_path,
                output_dir=target_dir,
                extract_only=False,
                validate_after=True
            )
            
            # Jika target dir berbeda dengan data_dir, copy ke data_dir
            if target_dir != self.data_dir:
                self.logger.info(f"🔄 Menyalin dataset ke {self.data_dir}...")
                copy_result = self.processor.copy_dataset_to_data_dir(target_dir, self.data_dir)
                
                self.logger.info(
                    f"   • Files disalin: {copy_result['copied']}\n"
                    f"   • Errors: {copy_result['errors']}"
                )
                
                # Tambahkan hasil copy ke result
                result['copied_to_data_dir'] = copy_result
            
            # Hapus file zip jika diminta
            if remove_zip and zip_path.exists():
                zip_path.unlink()
                self.logger.info(f"🗑️ File ZIP dihapus: {zip_path}")
                result['zip_removed'] = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error import dataset: {str(e)}")
            raise DatasetError(f"Error import dataset: {str(e)}")
    
    def pull_dataset(
        self, 
        format: str = "yolov5pytorch", 
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[str] = None,
        show_progress: bool = True,
        force_download: bool = False
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
            force_download: Paksa download ulang meskipun dataset sudah ada
            
        Returns:
            Dictionary berisi paths dan info dataset
        """
        start_time = time.time()
        
        # Cek apakah dataset sudah tersedia
        if not force_download and self.validator.is_dataset_available(self.data_dir, verify_content=True):
            self.logger.info("✅ Dataset sudah tersedia di lokal")
            
            # Return info
            stats = self.validator.get_local_stats(self.data_dir)
            return {
                'status': 'local',
                'paths': {split: str(self.data_dir / split) for split in DEFAULT_SPLITS},
                'data_dir': str(self.data_dir),
                'stats': stats,
                'duration': 0
            }
        
        # Dataset belum tersedia atau force download, download dari Roboflow
        if force_download:
            self.logger.info("🔄 Memulai download ulang dataset dari Roboflow...")
        else:
            self.logger.info("🔄 Dataset belum tersedia atau tidak lengkap, mendownload dari Roboflow...")
        
        try:
            # Download dataset
            download_result = self.download_from_roboflow(
                api_key=api_key,
                workspace=workspace,
                project=project,
                version=version,
                format=format,
                show_progress=show_progress
            )
            
            download_dir = download_result.get('output_dir')
            
            # Export ke struktur lokal
            export_result = self.export_to_local(download_dir, self.data_dir, show_progress)
            
            elapsed_time = time.time() - start_time
            
            # Return info
            return {
                'status': 'downloaded',
                'paths': export_result['paths'],
                'data_dir': str(self.data_dir),
                'download_dir': download_dir,
                'export_result': export_result,
                'download_result': download_result,
                'stats': self.validator.get_local_stats(self.data_dir),
                'duration': elapsed_time
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error saat pull dataset: {str(e)}")
            # Cek apakah dataset masih tersedia meskipun error
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
            # Tambahkan statistik ukuran dataset
            total_images = sum(local_stats.get(split, 0) for split in DEFAULT_SPLITS)
            
            self.logger.info(
                f"🔍 Dataset (Lokal): {info['name']} v{info['version']} | "
                f"Total: {total_images} gambar | "
                f"Train: {local_stats.get('train', 0)}, "
                f"Valid: {local_stats.get('valid', 0)}, "
                f"Test: {local_stats.get('test', 0)}"
            )
        else:
            self.logger.info(
                f"🔍 Dataset (akan didownload): {info['name']} "
                f"v{info['version']} dari {info['workspace']}"
            )
        
        # Tambahkan detail API key tersedia atau tidak
        has_api_key = bool(self.api_key)
        info['has_api_key'] = has_api_key
        
        if not has_api_key:
            self.logger.warning(f"⚠️ API key untuk Roboflow tidak tersedia")
        
        return info
    
    def check_dataset_structure(self) -> Dict[str, Any]:
        """
        Periksa struktur dataset dan tampilkan laporan.
        
        Returns:
            Dictionary berisi status struktur dataset
        """
        result = {
            'data_dir': str(self.data_dir),
            'is_valid': False,
            'splits': {}
        }
        
        self.logger.info(f"🔍 Memeriksa struktur dataset di {self.data_dir}...")
        
        # Periksa struktur folder
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
                # Hitung file
                image_files = list(images_dir.glob('*.*'))
                label_files = list(labels_dir.glob('*.txt'))
                
                split_info['image_count'] = len(image_files)
                split_info['label_count'] = len(label_files)
                split_info['valid'] = len(image_files) > 0 and len(label_files) > 0
                
                self.logger.info(
                    f"   • {split}: {'✅' if split_info['valid'] else '❌'} "
                    f"({split_info['image_count']} gambar, {split_info['label_count']} label)"
                )
            else:
                self.logger.warning(f"   • {split}: ❌ Struktur direktori tidak lengkap")
            
            result['splits'][split] = split_info
            
        # Validasi keseluruhan
        result['is_valid'] = all(info['valid'] for info in result['splits'].values())
        
        if result['is_valid']:
            self.logger.success(f"✅ Struktur dataset valid di {self.data_dir}")
        else:
            self.logger.warning(f"⚠️ Struktur dataset tidak valid atau tidak lengkap di {self.data_dir}")
        
        return result