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
        self.file_locks = {}
        
        # Roboflow settings
        rf_config = self.config.get('data', {}).get('roboflow', {})
        self.api_key = api_key or rf_config.get('api_key') or os.getenv("ROBOFLOW_API_KEY")
        self.workspace = rf_config.get('workspace', 'smartcash-wo2us')
        self.project = rf_config.get('project', 'rupiah-emisi-2022')
        self.version = rf_config.get('version', '3')
        
        if not self.api_key:
            self.logger.warning("⚠️ Roboflow API key tidak ditemukan. Fitur download mungkin tidak berfungsi.")

    def download_roboflow(self, version_obj, format: str = "yolov5pytorch", output_dir: Optional[str] = None): 
        dataset = version_obj.download(model_format=format, location=output_dir)
        self._verify_download(output_dir)
        
        self.logger.success(f"✅ Dataset {version_obj.workspace}/{version_obj.project}:{version_obj.version} berhasil didownload ke {output_dir}")
        self._notify_download_complete(version_obj.workspace, version_obj.project, version_obj.version, output_dir)
        return dataset.location
    
    def download_dataset(
        self,
        format: str = "yolov5pytorch",
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        project: Optional[str] = None,
        version: Optional[str] = None,
        output_dir: Optional[str] = None,
        show_progress: bool = True,
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
        
        # Buat directory jika belum ada
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir)
        
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
            # Import Roboflow hanya jika diperlukan
            from roboflow import Roboflow
            
            # Inisialisasi Roboflow
            rf = Roboflow(api_key=api_key)
            
            # Akses project dan download dataset
            project_obj = rf.workspace(workspace).project(project)
            version_obj = project_obj.version(version)
            
            # Gunakan tqdm untuk progress tracking jika diminta
            if show_progress:
                self.logger.info(f"⏳ Downloading dataset dari Roboflow")
                
                # Implementasi download manual dengan progress tracking
                download_url = f"https://api.roboflow.com/{workspace}/{project}/{version}/{format}?api_key={api_key}"
                
                # Jika url tidak valid, gunakan implementasi default dari Roboflow
                if not download_url or not download_url.startswith('http'):
                    self.logger.warning("⚠️ URL download tidak valid, menggunakan metode default Roboflow")
                    return self.download_roboflow(version_obj, format, output_dir)
                
                # Download manual dengan progress tracking
                zip_path = output_path / "dataset.zip"
                self._download_with_progress(download_url, zip_path)
                
                # Ekstrak file zip
                self.logger.info(f"📦 Mengekstrak dataset...")
                self.extract_zip(zip_path, output_path, remove_zip=True, show_progress=True)
            else:
                return self.download_roboflow(version_obj, format, output_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Error download dataset: {str(e)}")
            EventDispatcher.notify(
                event_type=EventTopics.DOWNLOAD_ERROR,
                sender=self,
                error=str(e)
            )
            raise
    
    def _download_with_progress(self, url: str, output_path: Path) -> None:
        """
        Download file dengan progress tracking.
        
        Args:
            url: URL file yang akan didownload (Roboflow API URL)
            output_path: Path untuk menyimpan file
        """
        try:
            self.logger.info(f"📥 Mendapatkan link download dari: {url}")
            
            # Step 1: Get the export link from the Roboflow API
            api_response = requests.get(url, timeout=self.timeout)
            api_response.raise_for_status()  # Raise error if request fails
            
            # Extract the actual download URL from JSON
            export_data = api_response.json()
            export_link = export_data.get('export', {}).get('link')
            
            if not export_link:
                raise ValueError("No export link found in API response")

            self.logger.info(f"📥 Downloading dari: {export_link}")
            
            # Step 2: Download the file from the export_link with streaming
            response = requests.get(export_link, stream=True, timeout=self.timeout)
            response.raise_for_status()  # Raise error if download fails
            
            # Setup progress bar
            total_size = int(response.headers.get('content-length', 0))
            progress = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=f"Downloading {output_path.name}"
            )
            
            # Download file in chunks
            with open(output_path, 'wb') as f:
                downloaded = 0
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
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Download gagal: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            raise
        except ValueError as e:
            self.logger.error(f"❌ Error: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            raise
        except Exception as e:
            self.logger.error(f"❌ Unexpected error: {str(e)}")
            if output_path.exists():
                output_path.unlink()
            raise
    
    def _notify_download_complete(self, workspace: str, project: str, version: str, output_dir: str) -> None:
        """
        Kirim notifikasi download selesai.
        
        Args:
            workspace: Roboflow workspace
            project: Roboflow project
            version: Roboflow version
            output_dir: Directory dataset
        """
        stats = self._get_dataset_stats(output_dir)
        
        EventDispatcher.notify(
            event_type=EventTopics.DOWNLOAD_COMPLETE,
            sender=self,
            workspace=workspace,
            project=project,
            version=version,
            output_dir=output_dir,
            stats=stats
        )
    
    def _get_dataset_stats(self, dataset_dir: str) -> Dict[str, int]:
        """
        Dapatkan statistik dataset.
        
        Args:
            dataset_dir: Directory dataset
            
        Returns:
            Dictionary berisi statistik jumlah file per split
        """
        dataset_path = Path(dataset_dir)
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            split_path = dataset_path / split
            if not split_path.exists():
                stats[split] = 0
                continue
                
            images_dir = split_path / 'images'
            if not images_dir.exists():
                stats[split] = 0
                continue
                
            # Hitung jumlah file
            img_count = sum(1 for _ in images_dir.glob('*.jpg')) + \
                        sum(1 for _ in images_dir.glob('*.jpeg')) + \
                        sum(1 for _ in images_dir.glob('*.png'))
            
            stats[split] = img_count
        
        return stats
    
    def _verify_download(self, output_dir: Union[str, Path]) -> bool:
        """
        Verifikasi hasil download dataset.
        
        Args:
            output_dir: Direktori output download
            
        Returns:
            Boolean yang menunjukkan apakah download valid
        """
        output_path = Path(output_dir)
        
        # Periksa struktur direktori
        required_dirs = DEFAULT_SPLITS
        for subdir in required_dirs:
            subdir_path = output_path / subdir
            if not (subdir_path.exists() and 
                    (subdir_path / 'images').exists() and 
                    (subdir_path / 'labels').exists()):
                self.logger.warning(f"⚠️ Struktur direktori tidak lengkap: {subdir_path}")
                return False
                
        # Periksa minimal ada file di setiap direktori
        has_files = True
        for subdir in required_dirs:
            img_dir = output_path / subdir / 'images'
            label_dir = output_path / subdir / 'labels'
            
            img_files = list(img_dir.glob('*.[jJ][pP]*[gG]')) + list(img_dir.glob('*.png'))
            label_files = list(label_dir.glob('*.txt'))
            
            if not img_files:
                self.logger.warning(f"⚠️ Tidak ada gambar di {img_dir}")
                has_files = False
                
            if not label_files:
                self.logger.warning(f"⚠️ Tidak ada label di {label_dir}")
                has_files = False
                
        return has_files
    
    def export_to_local(self, roboflow_dir: Union[str, Path], show_progress: bool = True) -> Tuple[str, str, str]:
        """Export dataset Roboflow ke struktur folder lokal standar."""
        self.logger.start("📤 Mengexport dataset ke struktur folder lokal...")
        rf_path = Path(roboflow_dir)
        
        # Verifikasi sumber ada
        if not rf_path.exists():
            raise FileNotFoundError(f"❌ Direktori sumber tidak ditemukan: {rf_path}")
        
        # Persiapkan file yang akan di-copy
        file_copy_tasks = []
        for split in ['train', 'valid', 'test']:
            src_dir = rf_path / split
            target_dir = self.data_dir / split
            
            for item in ['images', 'labels']:
                os.makedirs(target_dir / item, exist_ok=True)
                src_item_dir = src_dir / item
                
                if src_item_dir.exists():
                    for file_path in src_item_dir.glob('*'):
                        if file_path.is_file():
                            target_path = target_dir / item / file_path.name
                            if not target_path.exists():
                                file_copy_tasks.append((file_path, target_path))
        
        # Copy file secara paralel
        total_files = len(file_copy_tasks)
        if total_files == 0:
            self.logger.info("✅ Tidak ada file baru yang perlu di-copy")
        else:
            self.logger.info(f"🔄 Memindahkan {total_files} file...")
            progress = tqdm(total=total_files, desc="Copying files", unit="file") if show_progress else None
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(shutil.copy2, src, dst) for src, dst in file_copy_tasks]
                for future in futures:
                    future.result()
                    if progress: progress.update(1)
            
            if progress: progress.close()
        
        # Verifikasi hasil export
        self._verify_local_dataset()
            
        # Return paths
        output_dirs = tuple(str(self.data_dir / split) for split in DEFAULT_SPLITS)
        self.logger.success(f"✅ Dataset berhasil diexport: {total_files} file → {self.data_dir}")
        return output_dirs
    
    def _verify_local_dataset(self) -> bool:
        """
        Verifikasi dataset lokal setelah export.
        
        Returns:
            Boolean yang menunjukkan apakah dataset valid
        """
        # Periksa struktur direktori
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_dir / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not (images_dir.exists() and labels_dir.exists()):
                self.logger.warning(f"⚠️ Struktur direktori tidak lengkap: {split_dir}")
                return False
                
            # Hitung file
            img_count = sum(1 for _ in images_dir.glob('*.[jJ][pP]*[gG]')) + sum(1 for _ in images_dir.glob('*.png'))
            label_count = sum(1 for _ in labels_dir.glob('*.txt'))
            
            self.logger.info(f"📊 Split {split}: {img_count} gambar, {label_count} label")
            
            if img_count == 0 or label_count == 0:
                self.logger.warning(f"⚠️ Split {split} tidak memiliki gambar/label")
                return False
                
        return True
            
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
            if self._is_dataset_available(verify_content=True):
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
        self.extract_zip(zip_path, extract_dir, remove_zip=False, show_progress=True)
        
        # Fix struktur dataset jika perlu
        valid_structure = self._fix_dataset_structure(extract_dir)
        if not valid_structure:
            self.logger.warning(f"⚠️ Struktur dataset tidak sesuai format YOLOv5")
        
        # Verifikasi hasil import
        if not self._verify_dataset_structure(extract_dir):
            self.logger.warning(f"⚠️ Dataset tidak lengkap setelah import")
        else:
            self.logger.success(f"✅ Dataset berhasil diimport ke {extract_dir}")
        
        # Copy ke data_dir jika berbeda
        if extract_dir != self.data_dir:
            self.logger.info(f"🔄 Menyalin dataset ke {self.data_dir}...")
            self._copy_dataset_to_data_dir(extract_dir)
            
        return str(extract_dir)
    
    def _fix_dataset_structure(self, dataset_dir: Path) -> bool:
        """
        Memperbaiki struktur dataset jika tidak sesuai format YOLOv5.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Boolean yang menunjukkan apakah struktur sudah diperbaiki
        """
        # Deteksi struktur
        has_splits = all([(dataset_dir / split).exists() for split in DEFAULT_SPLITS])
        has_images_labels = (dataset_dir / 'images').exists() and (dataset_dir / 'labels').exists()
        
        # Jika sudah ada struktur split, verifikasi sub-direktori
        if has_splits:
            fixed = True
            for split in ['train', 'valid', 'test']:
                split_dir = dataset_dir / split
                if not (split_dir / 'images').exists():
                    (split_dir / 'images').mkdir(parents=True, exist_ok=True)
                if not (split_dir / 'labels').exists():
                    (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
            return fixed
            
        # Jika ada struktur images/labels di root, buat struktur split
        if has_images_labels:
            self.logger.info("🔧 Mengkonversi struktur flat ke split...")
            
            # Buat direktori split
            for split in ['train', 'valid', 'test']:
                (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
                (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Pindahkan semua ke train untuk sementara
            image_files = list((dataset_dir / 'images').glob('*'))
            label_files = list((dataset_dir / 'labels').glob('*'))
            
            for img in image_files:
                shutil.copy2(img, dataset_dir / 'train' / 'images' / img.name)
            
            for label in label_files:
                shutil.copy2(label, dataset_dir / 'train' / 'labels' / label.name)
                
            self.logger.info(f"✅ Semua file dipindahkan ke train/ ({len(image_files)} gambar, {len(label_files)} label)")
            return True
            
        # Cari subdirektori yang mengandung images dan labels
        for subdir in dataset_dir.iterdir():
            if subdir.is_dir():
                if (subdir / 'images').exists() and (subdir / 'labels').exists():
                    return True
                    
        # Jika sampai sini belum return, berarti struktur tidak terdeteksi
        self.logger.warning("⚠️ Struktur dataset tidak terdeteksi, membuat struktur baru...")
        for split in DEFAULT_SPLITS:
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
        return False
    
    def _verify_dataset_structure(self, dataset_dir: Path) -> bool:
        """
        Verifikasi struktur dataset sesuai format YOLOv5.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Boolean yang menunjukkan apakah struktur valid
        """
        # Periksa struktur direktori
        for split in DEFAULT_SPLITS:
            if not (dataset_dir / split).exists():
                return False
                
            if not (dataset_dir / split / 'images').exists() or not (dataset_dir / split / 'labels').exists():
                return False
                
        return True
    
    def _copy_dataset_to_data_dir(self, source_dir: Path) -> None:
        """
        Copy dataset ke direktori data utama.
        
        Args:
            source_dir: Direktori sumber dataset
        """
        for split in DEFAULT_SPLITS:
            # Buat direktori tujuan
            (self.data_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.data_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
            # Copy file
            for subdir in ['images', 'labels']:
                src_dir = source_dir / split / subdir
                dst_dir = self.data_dir / split / subdir
                
                if src_dir.exists():
                    for file_path in src_dir.glob('*'):
                        if file_path.is_file():
                            dst_path = dst_dir / file_path.name
                            if not dst_path.exists():
                                shutil.copy2(file_path, dst_path)
    
    def get_dataset_info(self) -> Dict:
        """Mendapatkan informasi dataset dari konfigurasi dan status lokal."""
        is_available = self._is_dataset_available()
        local_stats = self._get_local_stats() if is_available else {}
        
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
    
    def download_file(self, url: str, output_path: Union[str, Path], 
                     progress_bar: bool = True) -> Path:
        """Download file tunggal dengan progress tracking."""
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Setup file locking
        file_id = hashlib.md5(f"{url}_{str(output_path)}".encode()).hexdigest()
        if file_id not in self.file_locks:
            self.file_locks[file_id] = threading.Lock()
        lock = self.file_locks[file_id]
        
        with lock:
            # Download dengan retry
            try:
                for attempt in range(self.retry_limit):
                    try:
                        with requests.get(url, stream=True, timeout=self.timeout) as response:
                            response.raise_for_status()
                            
                            # Setup progress bar
                            total_size = int(response.headers.get('content-length', 0))
                            progress = tqdm(total=total_size, unit='B', unit_scale=True,
                                          desc=f"Download {output_path.name}") if progress_bar else None
                            
                            # Download file chunks
                            with open(output_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=self.chunk_size):
                                    if chunk:
                                        f.write(chunk)
                                        if progress: progress.update(len(chunk))
                            
                            if progress: progress.close()
                            
                            self.logger.info(f"✅ Download selesai: {output_path}")
                            return output_path
                            
                    except (requests.RequestException, IOError) as e:
                        if attempt < self.retry_limit - 1:
                            delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                            self.logger.warning(f"⚠️ Download gagal, retry dalam {delay:.1f}s: {str(e)}")
                            time.sleep(delay)
                        else:
                            raise
                
                raise RuntimeError(f"Gagal download setelah {self.retry_limit} percobaan")
                
            except Exception as e:
                self.logger.error(f"❌ Download gagal: {str(e)}")
                raise
    
    def extract_zip(self, zip_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None,
                   remove_zip: bool = True, show_progress: bool = True) -> Path:
        """Ekstrak file zip dengan progress bar."""
        zip_path, output_dir = Path(zip_path), Path(output_dir or zip_path.parent)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Ekstrak dengan progress
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                files = zip_ref.infolist()
                total_size = sum(f.file_size for f in files)
                
                progress = tqdm(total=total_size, unit='B', unit_scale=True,
                               desc=f"Extract {zip_path.name}") if show_progress else None
                
                for file in files:
                    zip_ref.extract(file, output_dir)
                    if progress: progress.update(file.file_size)
                
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
    
    def _is_dataset_available(self, verify_content: bool = False) -> bool:
        """
        Cek apakah dataset sudah tersedia di lokal.
        
        Args:
            verify_content: Verifikasi juga isi dari setiap split
            
        Returns:
            Boolean yang menunjukkan apakah dataset tersedia
        """
        for split in DEFAULT_SPLITS:
            split_dir = self.data_dir / split
            if not (split_dir / 'images').exists() or not (split_dir / 'labels').exists():
                return False
                
            # Verifikasi isi jika diminta
            if verify_content:
                img_count = sum(1 for _ in (split_dir / 'images').glob('*.[jJ][pP]*[gG]')) + sum(1 for _ in (split_dir / 'images').glob('*.png'))
                label_count = sum(1 for _ in (split_dir / 'labels').glob('*.txt'))
                
                if img_count == 0 or label_count == 0:
                    return False
                    
        return True
    
    def _get_local_stats(self) -> Dict[str, int]:
        """Hitung statistik dataset lokal."""
        return {split: sum(1 for _ in (self.data_dir / split / 'images').glob('*.*')) 
                for split in DEFAULT_SPLITS 
                if (self.data_dir / split / 'images').exists()}