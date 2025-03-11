"""
File: smartcash/handlers/dataset/core/download_core/roboflow_downloader.py
Author: Alfrida Sabar
Deskripsi: Komponen untuk download dataset dari Roboflow
"""

import os, json, time, zipfile, requests
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
        self.logger, self.timeout = logger, timeout
        self.chunk_size, self.retry_limit, self.retry_delay = chunk_size, retry_limit, retry_delay

    def _notify_progress(self, event_type: str, **kwargs) -> None:
        """Helper untuk progress notification."""
        EventDispatcher.notify(event_type=event_type, sender=self, **kwargs)

    def get_roboflow_metadata(self,workspace: str,project: str,version: str,api_key: str,format: str,temp_dir: Path) -> Dict:
        """Dapatkan metadata dari Roboflow sebelum download."""
        metadata_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{format}?api_key={api_key}"
        self.logger.info("📋 Mendapatkan metadata dataset dari Roboflow...")

        try:
            metadata = requests.get(metadata_url, timeout=self.timeout).json()
            if 'export' not in metadata or 'link' not in metadata['export']:
                raise ValueError("Format metadata tidak valid, tidak ada link download")

            # Simpan metadata
            metadata_path = temp_dir / f"{workspace}_{project}_{version}_metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            # Log informasi dataset
            if 'project' in metadata and 'classes' in metadata['project']:
                classes = metadata['project']['classes']
                splits = metadata.get('version', {}).get('splits', {})
                split_info = " | ".join(f"{s}: {c}" for s, c in splits.items() if c > 0)
                class_names = [f"{k} ({v})" for k, v in classes.items()]
                
                self.logger.info(
                    f"📊 Dataset memiliki {len(classes)} kelas dengan "
                    f"{metadata['version']['images']} gambar total ({split_info})\n"
                    f"📊 Kelas: {', '.join(class_names[:5])}"
                    + (f" dan {len(class_names)-5} lainnya" if len(class_names) > 5 else "")
                )
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan metadata: {str(e)}")
            raise

    def process_roboflow_download(self, download_url: str, output_path: Path, show_progress: bool = True) -> bool:
        """Proses download dari Roboflow dan ekstraksi."""
        zip_path = output_path / "dataset.zip"
        
        try:
            self.logger.info("📥 Downloading dataset dari Roboflow...")
            self._download_with_progress(download_url, zip_path)
            
            if not zip_path.exists() or zip_path.stat().st_size < 1000:
                self.logger.error(f"❌ File ZIP tidak valid: {zip_path}")
                return False

            self._extract_zip(zip_path, output_path, remove_zip=True, show_progress=show_progress)
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Proses download gagal: {str(e)}")
            if zip_path.exists(): zip_path.unlink()
            return False

    def _download_with_progress(self, url: str, output_path: Path) -> None:
        """Download file dengan progress tracking."""
        try:
            self.logger.info(f"📥 Downloading dari: {url}")
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with tqdm(total=total_size, unit='B', unit_scale=True,
                     desc=f"Downloading {output_path.name}") as progress, open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        f.write(chunk)
                        progress.update(len(chunk))
                        self._notify_progress(EventTopics.DOWNLOAD_PROGRESS, 
                                            progress=(downloaded := downloaded + len(chunk)),
                                            total=total_size)

            if output_path.stat().st_size == 0:
                raise ValueError("❌ Download gagal: File size 0 bytes")
            self.logger.info(f"✅ Download selesai: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Download gagal: {str(e)}")
            if output_path.exists(): output_path.unlink()
            raise

    def _extract_zip(self, zip_path: Union[str, Path], output_dir: Path,
                    remove_zip: bool = True, show_progress: bool = True) -> Path:
        """Ekstrak file zip dengan progress bar."""
        try:
            if not zipfile.is_zipfile(zip_path):
                raise ValueError(f"File tidak valid sebagai ZIP: {zip_path}")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = [f for f in zip_ref.infolist() if not f.is_dir()]
                total_size = sum(f.file_size for f in files)
                
                # Buat direktori untuk file yang memiliki path
                for file in files:
                    (output_dir / file.filename).parent.mkdir(parents=True, exist_ok=True)

                with tqdm(total=total_size, unit='B', unit_scale=True,
                          desc=f"Extract {zip_path.name}") as progress:
                    extracted_size = 0
                    for file in files:
                        zip_ref.extract(file, output_dir)
                        progress.update(file.file_size)
                        self._notify_progress(
                            EventTopics.PREPROCESSING_PROGRESS,
                            progress=(extracted_size := extracted_size + file.file_size),
                            total=total_size,
                            message=f"Mengekstrak {file.filename}"
                        )

            if remove_zip:
                zip_path.unlink()
                self.logger.info(f"🗑️ Zip dihapus: {zip_path}")

            self.logger.success(f"✅ Ekstraksi selesai: {output_dir}")
            return output_dir
            
        except Exception as e:
            self.logger.error(f"❌ Ekstraksi gagal: {str(e)}")
            raise