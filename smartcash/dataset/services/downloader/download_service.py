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
        
        # Inisialisasi komponen-komponen
        self._init_components()
        
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