# File: smartcash/handlers/dataset/core/dataset_downloader.py
# Author: Alfrida Sabar
# Deskripsi: Downloader dataset terintegrasi dengan dukungan resume, retry, dan paralel

import os, json, time, hashlib, threading, zipfile, dotenv
import requests, shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.utils.logger import get_logger
from smartcash.configs import get_config_manager


class DatasetDownloader:
    """Downloader dataset dari berbagai sumber dengan dukungan paralel, resume, dan retry."""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        logger = None,
        num_workers: int = 4,
        chunk_size: int = 8192,
        retry_limit: int = 3
    ):
        """Inisialisasi DatasetDownloader."""
        self.logger = logger or get_logger("dataset_downloader")
        dotenv.load_dotenv()
        
        # Setup paths dan konfigurasi
        self.config = config or get_config_manager().get_config()
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
        self.api_key = api_key or self.config.get('data', {}).get('roboflow', {}).get('api_key') or os.getenv("ROBOFLOW_API_KEY")
        rf_config = self.config.get('data', {}).get('roboflow', {})
        self.workspace = rf_config.get('workspace', 'smartcash-wo2us')
        self.project = rf_config.get('project', 'rupiah-emisi-2022')
        self.version = rf_config.get('version', '3')
        
        if not self.api_key:
            self.logger.warning("⚠️ Roboflow API key tidak ditemukan. Fitur download mungkin tidak berfungsi.")
    
    def download_dataset(self, format: str = "yolov5", show_progress: bool = True, resume: bool = True) -> str:
        """Download dataset dari Roboflow dengan progress bar dan dukungan resume."""
        if not self.api_key:
            raise ValueError("Roboflow API key diperlukan untuk download dataset")
            
        self.logger.start(f"🔄 Mendownload dataset Roboflow {self.workspace}/{self.project}:{self.version} format {format}")
        
        # Cek apakah dataset sudah ada
        download_dir = self.data_dir / f"roboflow_{self.project}_{self.version}"
        dataset_info_path = download_dir / "dataset_info.json"
        os.makedirs(download_dir, exist_ok=True)
        
        if download_dir.exists() and dataset_info_path.exists():
            try:
                with open(dataset_info_path, 'r') as f:
                    info = json.load(f)
                
                if (info.get('completed', False) and info.get('workspace') == self.workspace and
                    info.get('project') == self.project and info.get('version') == self.version and
                    info.get('format') == format):
                    self.logger.info(f"✅ Dataset sudah ada di {download_dir}")
                    return str(download_dir)
            except:
                pass
        
        # Download dataset
        url = f"https://app.roboflow.com/ds/download?workspace={self.workspace}&project={self.project}&version={self.version}&format={format}&api_key={self.api_key}"
        zip_path = self.temp_dir / f"{self.project}_{self.version}_{format}.zip"
        
        self.download_file(
            url=url,
            output_path=zip_path,
            resume=resume,
            progress_bar=show_progress,
            metadata={'workspace': self.workspace, 'project': self.project, 'version': self.version, 'format': format}
        )
        
        # Ekstrak dataset
        self.logger.info(f"📦 Mengekstrak dataset...")
        self.extract_zip(zip_path=zip_path, output_dir=download_dir, remove_zip=True, show_progress=show_progress)
        
        # Simpan info dataset
        dataset_info = {
            'workspace': self.workspace, 'project': self.project, 'version': self.version, 'format': format,
            'timestamp': time.time(), 'completed': True, 'splits': self._count_files(download_dir)
        }
        
        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        # Log hasil
        splits_info = dataset_info.get('splits', {})
        self.logger.success(f"✅ Dataset berhasil diunduh ke {download_dir} 📥\n" +
                           f"   Train: {splits_info.get('train', 0)}, Valid: {splits_info.get('valid', 0)}, Test: {splits_info.get('test', 0)}")
        
        return str(download_dir)
    
    def export_to_local(self, roboflow_dir: Union[str, Path], show_progress: bool = True) -> Tuple[str, str, str]:
        """Export dataset Roboflow ke struktur folder lokal standar."""
        self.logger.start("📤 Mengexport dataset ke struktur folder lokal...")
        rf_path = Path(roboflow_dir)
        
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
        
        # Return paths
        output_dirs = tuple(str(self.data_dir / split) for split in ['train', 'valid', 'test'])
        self.logger.success(f"✅ Dataset berhasil diexport: {total_files} file → {self.data_dir}")
        return output_dirs
            
    def pull_dataset(self, format: str = "yolov5", show_progress: bool = True, resume: bool = True,
               api_key: str = None, workspace: str = None, project: str = None, version: str = None) -> tuple:
        """One-step untuk download dan setup dataset siap pakai."""
        try:
            # Override parameter sementara jika diberikan
            old_values = {}
            for param in ['api_key', 'workspace', 'project', 'version']:
                val = locals()[param]
                if val is not None:
                    old_values[param] = getattr(self, param)
                    setattr(self, param, val)
            
            # Jika dataset sudah ada
            if self._is_dataset_available():
                self.logger.info("✅ Dataset sudah tersedia di lokal")
                return tuple(str(self.data_dir / split) for split in ['train', 'valid', 'test'])
            
            # Download dari Roboflow dan export ke lokal
            roboflow_dir = self.download_dataset(format=format, show_progress=show_progress, resume=resume)
            return self.export_to_local(roboflow_dir, show_progress)
        except Exception as e:
            self.logger.error(f"❌ Gagal pull dataset: {str(e)}")
            raise
        finally:
            # Kembalikan nilai semula
            for param, val in old_values.items():
                setattr(self, param, val)
    
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
    
    # ===== File download methods =====
    
    def download_file(self, url: str, output_path: Union[str, Path], resume: bool = True,
                    progress_bar: bool = True, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Download file tunggal dengan dukungan resume."""
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # Setup file locking dan info
        file_id = hashlib.md5(f"{url}_{str(output_path)}".encode()).hexdigest()
        if file_id not in self.file_locks:
            self.file_locks[file_id] = threading.Lock()
        lock = self.file_locks[file_id]
        resume_info_path = self.temp_dir / f"{file_id}.json"
        
        with lock:
            # Cek file sudah ada dan lengkap
            if output_path.exists() and resume and resume_info_path.exists():
                try:
                    with open(resume_info_path) as f:
                        info = json.load(f)
                    if info.get('completed') and info.get('size') == output_path.stat().st_size:
                        self.logger.info(f"✅ File sudah ada: {output_path}")
                        return output_path
                except:
                    pass
            
            # Setup resume
            start_byte, temp_file_path = 0, self.temp_dir / f"{file_id}.part"
            if temp_file_path.exists() and resume and resume_info_path.exists():
                try:
                    with open(resume_info_path) as f:
                        info = json.load(f)
                    if time.time() - info.get('timestamp', 0) < 24*60*60 and temp_file_path.stat().st_size == info.get('downloaded', 0):
                        start_byte = temp_file_path.stat().st_size
                        self.logger.info(f"🔄 Resume download dari {start_byte/1024/1024:.1f} MB")
                except:
                    pass
            
            # Simpan info resume
            resume_info = {'url': url, 'output_path': str(output_path), 'timestamp': time.time(), 
                          'downloaded': start_byte, 'completed': False}
            if metadata:
                resume_info['metadata'] = metadata
            with open(resume_info_path, 'w') as f:
                json.dump(resume_info, f)
            
            # Download dengan retry
            try:
                headers = {'Range': f'bytes={start_byte}-'} if start_byte > 0 else {}
                
                for attempt in range(self.retry_limit):
                    try:
                        with requests.get(url, stream=True, headers=headers, timeout=self.timeout) as response:
                            response.raise_for_status()
                            
                            # Setup progress bar
                            total_size = int(response.headers.get('content-length', 0))
                            if start_byte > 0 and response.status_code == 206 and 'content-range' in response.headers:
                                try:
                                    total_size = int(response.headers['content-range'].split('/')[-1])
                                except:
                                    pass
                            
                            progress = tqdm(total=total_size + start_byte if total_size > 0 else None,
                                           initial=start_byte, unit='B', unit_scale=True,
                                           desc=f"Download {output_path.name}") if progress_bar else None
                            
                            # Download file chunks
                            mode = 'ab' if start_byte > 0 else 'wb'
                            with open(temp_file_path, mode) as f:
                                downloaded = start_byte
                                for chunk in response.iter_content(chunk_size=self.chunk_size):
                                    if chunk:
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        if progress: progress.update(len(chunk))
                                        
                                        # Update info periodik
                                        if downloaded % (self.chunk_size * 100) == 0:
                                            resume_info.update({'downloaded': downloaded, 'timestamp': time.time()})
                                            with open(resume_info_path, 'w') as info_file:
                                                json.dump(resume_info, info_file)
                            
                            if progress: progress.close()
                            
                            # Selesaikan download
                            shutil.move(temp_file_path, output_path)
                            resume_info.update({'completed': True, 'size': output_path.stat().st_size, 'timestamp': time.time()})
                            with open(resume_info_path, 'w') as info_file:
                                json.dump(resume_info, info_file)
                            
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
                resume_info.update({'error': str(e), 'timestamp': time.time()})
                with open(resume_info_path, 'w') as info_file:
                    json.dump(resume_info, info_file)
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
    
    # ===== Helper methods =====
    
    def _count_files(self, dir_path: Path) -> Dict[str, int]:
        """Hitung file pada setiap split dataset."""
        counts = {}
        for split in ['train', 'valid', 'test']:
            images_dir = dir_path / split / 'images'
            if images_dir.exists():
                counts[split] = sum(1 for _ in images_dir.glob('*.jpg')) + sum(1 for _ in images_dir.glob('*.jpeg')) + sum(1 for _ in images_dir.glob('*.png'))
        return counts
    
    def _is_dataset_available(self) -> bool:
        """Cek apakah dataset sudah tersedia di lokal."""
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_dir / split
            if not (split_dir / 'images').exists() or not (split_dir / 'labels').exists():
                return False
            if not list((split_dir / 'images').glob('*.*')):
                return False
        return True
    
    def _get_local_stats(self) -> Dict[str, int]:
        """Hitung statistik dataset lokal."""
        return {split: sum(1 for _ in (self.data_dir / split / 'images').glob('*.*')) 
                for split in ['train', 'valid', 'test'] 
                if (self.data_dir / split / 'images').exists()}