"""
File: smartcash/dataset/services/downloader/roboflow_downloader.py
Deskripsi: Komponen untuk download dataset dari Roboflow dengan dukungan resume dan progress tracking
"""

import os
import json
import time
import zipfile
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Any
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger


class RoboflowDownloader:
    """Komponen untuk download dataset dari Roboflow."""
    
    def __init__(
        self,
        logger=None,
        timeout: int = 30,
        chunk_size: int = 8192,
        retry_limit: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Inisialisasi RoboflowDownloader.
        
        Args:
            logger: Logger kustom (opsional)
            timeout: Timeout untuk request dalam detik
            chunk_size: Ukuran chunk untuk download
            retry_limit: Batas retry pada kegagalan
            retry_delay: Delay antar retry dalam detik
        """
        self.logger = logger or get_logger("roboflow_downloader")
        self.timeout = timeout
        self.chunk_size = chunk_size
        self.retry_limit = retry_limit
        self.retry_delay = retry_delay
    
    def get_roboflow_metadata(
        self,
        workspace: str,
        project: str,
        version: str,
        api_key: str,
        format: str,
        temp_dir: Union[str, Path]
    ) -> Dict:
        """
        Dapatkan metadata dari Roboflow sebelum download.
        
        Args:
            workspace: Roboflow workspace
            project: Roboflow project
            version: Roboflow version
            api_key: Roboflow API key
            format: Format download dataset
            temp_dir: Direktori temporer untuk menyimpan metadata
            
        Returns:
            Dictionary berisi metadata dataset
        """
        # Buat URL untuk metadata
        metadata_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{format}?api_key={api_key}"
        
        self.logger.info(f"📋 Mendapatkan metadata dataset dari Roboflow...")
        
        try:
            response = requests.get(metadata_url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse response sebagai JSON
            metadata = response.json()
            
            # Verifikasi metadata
            if 'export' not in metadata or 'link' not in metadata['export']:
                raise ValueError("Format metadata tidak valid, tidak ada link download")
                
            # Simpan metadata untuk verifikasi integritas
            temp_path = Path(temp_dir)
            temp_path.mkdir(parents=True, exist_ok=True)
            
            metadata_path = temp_path / f"{workspace}_{project}_{version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log informasi dataset
            if 'project' in metadata and 'classes' in metadata['project']:
                classes = metadata['project']['classes']
                num_classes = len(classes)
                total_images = metadata.get('version', {}).get('images', 0)
                
                # Tampilkan distribusi split
                splits = metadata.get('version', {}).get('splits', {})
                split_info = " | ".join([f"{s}: {c}" for s, c in splits.items() if c > 0])
                
                self.logger.info(
                    f"📊 Dataset memiliki {num_classes} kelas dengan "
                    f"{total_images} gambar total ({split_info})"
                )
                
                # Simpan informasi kelas untuk verifikasi label
                class_names = []
                for cls_name, count in classes.items():
                    class_names.append(f"{cls_name} ({count})")
                
                if class_names:
                    self.logger.info(
                        f"📊 Kelas: {', '.join(class_names[:5])}" + 
                        (f" dan {len(class_names)-5} lainnya" if len(class_names) > 5 else "")
                    )
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan metadata: {str(e)}")
            raise
    
    def process_roboflow_download(
        self,
        download_url: str,
        output_path: Union[str, Path],
        show_progress: bool = True
    ) -> bool:
        """
        Proses download dari Roboflow dan ekstraksi.
        
        Args:
            download_url: URL download dataset
            output_path: Path output untuk dataset
            show_progress: Tampilkan progress bar
            
        Returns:
            Boolean yang menunjukkan keberhasilan download dan ekstraksi
        """
        # Normalisasi path
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Siapkan path untuk file zip
        zip_path = output_path / "dataset.zip"
        
        try:
            # Download
            self.logger.info(f"📥 Downloading dataset dari Roboflow...")
            download_success = self._download_with_progress(download_url, zip_path, show_progress)
            
            if not download_success:
                return False
            
            # Verifikasi ukuran file zip
            if not zip_path.exists() or zip_path.stat().st_size < 1000:
                self.logger.error(f"❌ File ZIP tidak valid: {zip_path}")
                return False
            
            # Ekstrak file zip
            self.logger.info(f"📦 Mengekstrak dataset...")
            extract_success = self._extract_zip(zip_path, output_path, remove_zip=True, show_progress=show_progress)
            
            if not extract_success:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Proses download gagal: {str(e)}")
            # Hapus file zip yang mungkin rusak
            if zip_path.exists():
                zip_path.unlink()
            return False
    
    def _download_with_progress(
        self,
        url: str,
        output_path: Path,
        show_progress: bool = True
    ) -> bool:
        """
        Download file dengan progress tracking dan dukungan resume.
        
        Args:
            url: URL file yang akan didownload
            output_path: Path untuk menyimpan file
            show_progress: Tampilkan progress bar
            
        Returns:
            Sukses atau tidak
        """
        # Header untuk request
        headers = {}
        existing_size = 0
        
        # Cek apakah file sudah ada (untuk resume)
        if output_path.exists():
            existing_size = output_path.stat().st_size
            
            if existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'
                self.logger.info(f"🔄 Melanjutkan download dari {existing_size / (1024*1024):.2f} MB")
        
        # Buat parent directory jika belum ada
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Coba download dengan retry
        for attempt in range(1, self.retry_limit + 1):
            try:
                # Request dengan stream=True untuk download chunk by chunk
                self.logger.debug(f"📥 Downloading dari: {url} (attempt {attempt})")
                response = requests.get(
                    url,
                    stream=True,
                    headers=headers,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Dapatkan total size
                total_size = int(response.headers.get('content-length', 0))
                
                # Jika total size = 0 tetapi ada Range header, berarti file sudah selesai didownload
                if total_size == 0 and existing_size > 0:
                    self.logger.info(f"✅ File sudah lengkap ({existing_size / (1024*1024):.2f} MB)")
                    return True
                
                # Jika resume, tambahkan existing_size ke total_size
                if existing_size > 0:
                    total_size += existing_size
                
                # Setup progress bar
                mode = 'ab' if existing_size > 0 else 'wb'
                progress = None
                
                if show_progress:
                    progress = tqdm(
                        total=total_size,
                        initial=existing_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Downloading {output_path.name}"
                    )
                
                # Download file chunks
                with open(output_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=self.chunk_size):
                        if chunk:
                            f.write(chunk)
                            if progress:
                                progress.update(len(chunk))
                
                if progress:
                    progress.close()
                
                # Verifikasi file size
                file_size = output_path.stat().st_size
                if file_size == 0:
                    self.logger.error(f"❌ Download gagal: File size 0 bytes")
                    return False
                
                self.logger.info(f"✅ Download selesai: {file_size / (1024*1024):.2f} MB")
                return True
                
            except requests.exceptions.RequestException as e:
                if progress:
                    progress.close()
                
                if attempt < self.retry_limit:
                    wait_time = self.retry_delay * attempt
                    self.logger.warning(f"⚠️ Download error (attempt {attempt}/{self.retry_limit}): {str(e)}")
                    self.logger.info(f"🕒 Mencoba ulang dalam {wait_time:.1f} detik...")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"❌ Download gagal setelah {self.retry_limit} percobaan: {str(e)}")
                    return False
        
        return False
    
    def _extract_zip(
        self,
        zip_path: Union[str, Path],
        output_dir: Union[str, Path],
        remove_zip: bool = True,
        show_progress: bool = True
    ) -> bool:
        """
        Ekstrak file zip dengan progress bar.
        
        Args:
            zip_path: Path ke file zip
            output_dir: Direktori output untuk ekstraksi
            remove_zip: Hapus file zip setelah ekstraksi
            show_progress: Tampilkan progress bar
            
        Returns:
            Sukses atau tidak
        """
        try:
            zip_path = Path(zip_path)
            output_dir = Path(output_dir)
            
            # Cek apakah file zip valid
            if not zipfile.is_zipfile(zip_path):
                self.logger.error(f"❌ File tidak valid sebagai ZIP: {zip_path}")
                return False
                
            # Buka file zip dan periksa strukturnya terlebih dahulu
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                
                if not files:
                    self.logger.error(f"❌ File ZIP kosong: {zip_path}")
                    return False
                    
                total_size = sum(f.file_size for f in files)
                if total_size == 0:
                    self.logger.error(f"❌ File ZIP tidak berisi data: {zip_path}")
                    return False
                
                # Periksa dan buat direktori jika diperlukan
                for file in files:
                    if file.is_dir():
                        os.makedirs(output_dir / file.filename, exist_ok=True)
                
                # Ekstrak dengan progress
                progress = None
                if show_progress:
                    progress = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc=f"Extract {zip_path.name}"
                    )
                
                # Ekstrak file
                extracted_size = 0
                for file in files:
                    zip_ref.extract(file, output_dir)
                    
                    if progress:
                        extracted_size += file.file_size
                        progress.update(file.file_size)
                
                if progress:
                    progress.close()
            
            # Hapus zip jika diminta
            if remove_zip:
                zip_path.unlink()
                self.logger.info(f"🗑️ Zip dihapus: {zip_path}")
            
            self.logger.success(f"✅ Ekstraksi selesai: {output_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ekstraksi gagal: {str(e)}")
            return False
    
    def _download_with_progress(
        self,
        url: str,
        output_path: Path,
        show_progress: bool = True
    ) -> bool:
        """
        Download file dengan progress tracking dan dukungan resume.
        
        Args:
            url: URL file yang akan didownload
            output_path: Path untuk menyimpan file
            show_progress: Tampilkan progress bar
            
        Returns:
            Sukses atau tidak
        """
        # Header untuk request
        headers = {}
        existing_size = 0
        
        # Cek apakah file sudah ada (untuk resume)
        if output_path.exists():
            existing_size = output_path.stat().st_size
            
            if existing_size > 0:
                headers['Range'] = f'bytes={existing_size}-'
                self.logger.info(f"🔄 Melanjutkan download dari {existing_size / (1024*1024):.2f} MB")