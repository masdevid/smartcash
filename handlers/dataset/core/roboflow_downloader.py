"""
File: smartcash/handlers/dataset/core/download_core/roboflow_downloader.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk download dataset dari Roboflow
"""

import os
import json
import time
import zipfile
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Any
from tqdm.auto import tqdm

from smartcash.utils.observer import EventDispatcher, EventTopics


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
        """Inisialisasi RoboflowDownloader."""
        self.logger = logger
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
        temp_dir: Path
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
            metadata_path = temp_dir / f"{workspace}_{project}_{version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Log informasi dataset
            if 'project' in metadata and 'classes' in metadata['project']:
                classes = metadata['project']['classes']
                num_classes = len(classes)
                total_images = metadata['version']['images']
                
                # Tampilkan distribusi split
                splits = metadata.get('version', {}).get('splits', {})
                split_info = " | ".join([f"{s}: {c}" for s, c in splits.items() if c > 0])
                
                self.logger.info(f"📊 Dataset memiliki {num_classes} kelas dengan "
                                f"{total_images} gambar total ({split_info})")
                
                # Simpan informasi kelas untuk verifikasi label
                class_names = []
                for cls_name, count in classes.items():
                    class_names.append(f"{cls_name} ({count})")
                
                if class_names:
                    self.logger.info(f"📊 Kelas: {', '.join(class_names[:5])}" + 
                                    (f" dan {len(class_names)-5} lainnya" if len(class_names) > 5 else ""))
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan metadata: {str(e)}")
            raise
    
    def process_roboflow_download(self, download_url: str, output_path: Path, show_progress: bool = True) -> bool:
        """
        Proses download dari Roboflow dan ekstraksi.
        
        Args:
            download_url: URL download dataset
            output_path: Path output untuk dataset
            show_progress: Tampilkan progress bar
            
        Returns:
            Boolean yang menunjukkan keberhasilan download dan ekstraksi
        """
        # Siapkan path untuk file zip
        zip_path = output_path / "dataset.zip"
        
        try:
            # Download dengan progress tracking
            self.logger.info(f"📥 Downloading dataset dari Roboflow...")
            self._download_with_progress(download_url, zip_path)
            
            # Verifikasi ukuran file zip
            if not zip_path.exists() or zip_path.stat().st_size < 1000:
                self.logger.error(f"❌ File ZIP tidak valid: {zip_path}")
                return False
            
            # Ekstrak file zip
            self.logger.info(f"📦 Mengekstrak dataset...")
            self._extract_zip(zip_path, output_path, remove_zip=True, show_progress=show_progress)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Proses download gagal: {str(e)}")
            # Hapus file zip yang mungkin rusak
            if zip_path.exists():
                zip_path.unlink()
            return False
    
    def _download_with_progress(self, url: str, output_path: Path) -> None:
        """
        Download file dengan progress tracking.
        
        Args:
            url: URL file yang akan didownload
            output_path: Path untuk menyimpan file
        """
        try:
            self.logger.info(f"📥 Downloading dari: {url}")
            
            # Request dengan stream=True untuk download chunk by chunk
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            # Setup progress bar
            total_size = int(response.headers.get('content-length', 0))
            progress = tqdm(
                total=total_size, 
                unit='B', 
                unit_scale=True,
                desc=f"Downloading {output_path.name}"
            )
            
            # Download file chunks
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.update(len(chunk))
                        
                        # Notify progress
                        EventDispatcher.notify(
                            event_type=EventTopics.DOWNLOAD_PROGRESS,
                            sender=self,
                            progress=downloaded,
                            total=total_size
                        )
            
            progress.close()
            
            # Verifikasi file size
            if os.path.getsize(output_path) == 0:
                raise ValueError(f"❌ Download gagal: File size 0 bytes")
                
            self.logger.info(f"✅ Download selesai: {output_path}")
        
        except Exception as e:
            self.logger.error(f"❌ Download gagal: {str(e)}")
            # Hapus file yang mungkin rusak
            if output_path.exists():
                output_path.unlink()
            raise
    
    def _extract_zip(self, zip_path: Union[str, Path], output_dir: Path,
                    remove_zip: bool = True, show_progress: bool = True) -> Path:
        """
        Ekstrak file zip dengan progress bar.
        
        Args:
            zip_path: Path ke file zip
            output_dir: Direktori output untuk ekstraksi
            remove_zip: Hapus file zip setelah ekstraksi
            show_progress: Tampilkan progress bar
            
        Returns:
            Path direktori output
        """
        try:
            # Cek apakah file zip valid
            if not zipfile.is_zipfile(zip_path):
                raise ValueError(f"File tidak valid sebagai ZIP: {zip_path}")
                
            # Buka file zip dan periksa strukturnya terlebih dahulu
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                
                if not files:
                    raise ValueError(f"File ZIP kosong: {zip_path}")
                    
                total_size = sum(f.file_size for f in files)
                if total_size == 0:
                    raise ValueError(f"File ZIP tidak berisi data: {zip_path}")
                
                # Periksa dan buat direktori jika diperlukan
                for file in files:
                    if file.is_dir():
                        os.makedirs(output_dir / file.filename, exist_ok=True)
                
                # Ekstrak dengan progress
                progress = tqdm(total=total_size, unit='B', unit_scale=True,
                                desc=f"Extract {zip_path.name}") if show_progress else None
                
                # Ekstrak file secara paralel
                extracted_size = 0
                for file in files:
                    zip_ref.extract(file, output_dir)
                    
                    if progress:
                        extracted_size += file.file_size
                        progress.update(file.file_size)
                        
                        # Notifikasi progress
                        EventDispatcher.notify(
                            event_type=EventTopics.PREPROCESSING_PROGRESS,
                            sender=self,
                            progress=extracted_size,
                            total=total_size,
                            message=f"Mengekstrak {file.filename}"
                        )
                
                if progress: progress.close()
            
            # Hapus zip jika diminta
            if remove_zip:
                zip_path.unlink()
                self.logger.info(f"🗑️ Zip dihapus: {zip_path}")
            
            self.logger.success(f"✅ Ekstraksi selesai: {output_dir}")
            return output_dir
            
        except Exception as e:
            self.logger.error(f"❌ Ekstraksi gagal: {str(e)}")
            raise