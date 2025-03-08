# File: smartcash/handlers/dataset/dataset_downloader.py
# Author: Alfrida Sabar
# Deskripsi: Downloader dataset dari Roboflow API

import os
import dotenv
import shutil
import requests
from typing import Dict, Optional, Tuple, Union
from pathlib import Path
from tqdm.auto import tqdm
import time

from smartcash.utils.logger import SmartCashLogger
from smartcash.config import get_config_manager

class DatasetDownloader:
    """
    Handler untuk download dan setup dataset dari Roboflow API.
    Menggunakan pola Strategy untuk mendukung berbagai sumber download.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi DatasetDownloader.
        
        Args:
            config: Konfigurasi untuk download
            data_dir: Direktori target dataset
            api_key: API key Roboflow
            logger: Logger kustom (opsional)
        """
        self.logger = logger or SmartCashLogger(__name__)
        
        # Load environment variables untuk API key
        dotenv.load_dotenv()
        
        # Prioritaskan config yang diberikan langsung, jika tidak ada baru load dari config manager
        if config:
            self.config = config
        else:
            config_manager = get_config_manager()
            self.config = config_manager.get_config()
            
        # Setup data directory
        self.data_dir = Path(data_dir) if data_dir else Path(self.config.get('data_dir', 'data'))
        
        # Set API Key dengan prioritas:
        # 1. Parameter api_key
        # 2. Config dari roboflow.api_key
        # 3. Environment variable ROBOFLOW_API_KEY
        self.api_key = api_key or \
                       self.config.get('data', {}).get('roboflow', {}).get('api_key') or \
                       os.getenv("ROBOFLOW_API_KEY")
        
        if not self.api_key:
            self.logger.warning("⚠️ Roboflow API key tidak ditemukan. Beberapa fitur mungkin tidak berfungsi.")
            
        # Siapkan informasi Roboflow
        roboflow_config = self.config.get('data', {}).get('roboflow', {})
        self.workspace = roboflow_config.get('workspace', 'smartcash-wo2us')
        self.project = roboflow_config.get('project', 'rupiah-emisi-2022')
        self.version = roboflow_config.get('version', '3')
        
        self.logger.info(f"🔧 DatasetDownloader diinisialisasi untuk {self.workspace}/{self.project}:{self.version}")
    
    def download_dataset(
        self, 
        format: str = "yolov5", 
        show_progress: bool = True
    ) -> Union[str, Dict]:
        """
        Download dataset dari Roboflow dengan progress bar.
        
        Args:
            format: Format dataset ('yolov5', 'coco', etc)
            show_progress: Tampilkan progress bar
            
        Returns:
            Path ke direktori dataset yang didownload
        """
        if not self.api_key:
            raise ValueError("Roboflow API key diperlukan untuk download dataset")
            
        self.logger.start(
            f"🔄 Mendownload dataset dari Roboflow...\n"
            f"   Workspace: {self.workspace}\n"
            f"   Project: {self.project}\n"
            f"   Version: {self.version}"
        )
        
        try:
            # Persiapkan direktori output
            download_dir = self.data_dir / f"roboflow_{self.project}_{self.version}"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            # Buat URL download menggunakan Roboflow API
            download_url = self._generate_download_url(format)
            
            # Download dataset dengan progress bar
            if show_progress:
                dataset_info = self._download_with_progress(download_url, download_dir)
            else:
                dataset_info = self._download_without_progress(download_url, download_dir)
            
            # Format hasil
            result = {
                'location': str(download_dir),
                'splits': dataset_info.get('splits', {})
            }
            
            self.logger.success(
                f"✅ Dataset berhasil diunduh ke {result['location']} 📥\n"
                f"   Train: {result['splits'].get('train', 'N/A')} gambar\n"
                f"   Valid: {result['splits'].get('valid', 'N/A')} gambar\n"
                f"   Test: {result['splits'].get('test', 'N/A')} gambar"
            )
            
            return result['location']
            
        except Exception as e:
            self.logger.error(f"❌ Gagal download dataset: {str(e)}")
            raise
    
    def _generate_download_url(self, format: str) -> str:
        """
        Generate URL download untuk Roboflow API.
        
        Args:
            format: Format dataset
            
        Returns:
            URL untuk download dataset
        """
        base_url = "https://app.roboflow.com/ds/download"
        url = f"{base_url}?workspace={self.workspace}&project={self.project}&version={self.version}&format={format}&api_key={self.api_key}"
        return url
    
    def _download_with_progress(self, download_url: str, output_dir: Path) -> Dict:
        """
        Download dataset dengan progress tracking.
        
        Args:
            download_url: URL untuk download
            output_dir: Direktori output
            
        Returns:
            Informasi dataset
        """
        try:
            # Path untuk file zip sementara
            zip_path = output_dir / "dataset.zip"
            
            # Download dengan progress bar
            response = requests.get(download_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            # Setup progress bar
            progress_bar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc="Downloading dataset"
            )
            
            # Download file
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            
            # Ekstrak file
            self.logger.info(f"📦 Mengekstrak dataset...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Hapus zip file
            zip_path.unlink()
            
            # Informasi dataset
            splits_info = self._count_dataset_files(output_dir)
            
            return {
                'location': str(output_dir),
                'splits': splits_info
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal download dengan progress: {str(e)}")
            raise
    
    def _download_without_progress(self, download_url: str, output_dir: Path) -> Dict:
        """
        Download dataset tanpa progress tracking.
        
        Args:
            download_url: URL untuk download
            output_dir: Direktori output
            
        Returns:
            Informasi dataset
        """
        try:
            # Path untuk file zip sementara
            zip_path = output_dir / "dataset.zip"
            
            # Download file
            response = requests.get(download_url)
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Ekstrak file
            self.logger.info(f"📦 Mengekstrak dataset...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            
            # Hapus zip file
            zip_path.unlink()
            
            # Informasi dataset
            splits_info = self._count_dataset_files(output_dir)
            
            return {
                'location': str(output_dir),
                'splits': splits_info
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal download tanpa progress: {str(e)}")
            raise
    
    def _count_dataset_files(self, dataset_dir: Path) -> Dict[str, int]:
        """
        Hitung jumlah file di setiap split dataset.
        
        Args:
            dataset_dir: Direktori dataset
            
        Returns:
            Dict berisi jumlah file per split
        """
        splits = {}
        
        for split in ['train', 'valid', 'test']:
            images_dir = dataset_dir / split / 'images'
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg'))) + \
                              len(list(images_dir.glob('*.jpeg'))) + \
                              len(list(images_dir.glob('*.png')))
                splits[split] = image_count
            else:
                splits[split] = 0
                
        return splits
    
    def export_to_local(
        self, 
        roboflow_dir: Union[str, Path],
        show_progress: bool = True
    ) -> Tuple[str, str, str]:
        """
        Export dataset Roboflow ke struktur folder lokal yang standar.
        
        Args:
            roboflow_dir: Direktori Roboflow
            show_progress: Tampilkan progress bar
            
        Returns:
            Tuple (train_dir, valid_dir, test_dir)
        """
        self.logger.start("📤 Mengexport dataset ke struktur folder lokal...")
        
        try:
            rf_path = Path(roboflow_dir)
            
            # Pastikan direktori tujuan ada
            self.data_dir.mkdir(exist_ok=True)
            
            # Mencari semua file untuk estimasi total
            total_files = 0
            for split in ['train', 'valid', 'test']:
                for item in ['images', 'labels']:
                    src_dir = rf_path / split / item
                    if src_dir.exists():
                        total_files += len(list(src_dir.glob('*')))
            
            # Copy untuk setiap split dataset
            processed_files = 0
            progress_bar = None
            
            if show_progress:
                progress_bar = tqdm(
                    total=total_files,
                    desc="Exporting files",
                    unit="file"
                )
            
            output_dirs = []
            for split in ['train', 'valid', 'test']:
                # Source dan target paths
                src_dir = rf_path / split
                target_dir = self.data_dir / split
                output_dirs.append(str(target_dir))
                
                # Buat direktori target
                target_dir.mkdir(exist_ok=True)
                (target_dir / 'images').mkdir(exist_ok=True)
                (target_dir / 'labels').mkdir(exist_ok=True)
                
                # Copy files
                for item in ['images', 'labels']:
                    src = src_dir / item
                    if src.exists():
                        for file in src.glob('*'):
                            target = target_dir / item / file.name
                            if not target.exists():
                                shutil.copy2(file, target)
                            
                            # Update progress
                            processed_files += 1
                            if progress_bar:
                                progress_bar.update(1)
            
            # Close progress bar
            if progress_bar:
                progress_bar.close()
                
            self.logger.success(
                f"✅ Dataset berhasil diexport ke struktur folder lokal\n"
                f"   Total file: {processed_files}\n"
                f"   Output: {self.data_dir}"
            )
            
            # Return paths untuk train, valid, dan test
            return tuple(output_dirs)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengexport dataset: {str(e)}")
            raise e
    
    def pull_dataset(self, show_progress: bool = True) -> Tuple[str, str, str]:
        """
        Download dan setup dataset dari Roboflow dengan progress tracking.
        Satu-step untuk mendapatkan dataset siap pakai.
        
        Args:
            show_progress: Tampilkan progress bar
            
        Returns:
            Tuple paths (train, valid, test)
        """
        try:
            # Tampilkan info dataset yang akan didownload
            self.get_dataset_info()
            
            # Jika dataset sudah ada, konfirmasi jika perlu redownload
            if self._is_dataset_available():
                self.logger.info("✅ Dataset sudah tersedia di lokal")
                return (
                    str(self.data_dir / 'train'),
                    str(self.data_dir / 'valid'),
                    str(self.data_dir / 'test')
                )
            
            # Download dari Roboflow
            roboflow_dir = self.download_dataset(show_progress=show_progress)
            
            # Export ke struktur folder lokal
            return self.export_to_local(roboflow_dir, show_progress=show_progress)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan pull dataset: {str(e)}")
            raise e
    
    def get_dataset_info(self) -> Dict:
        """
        Mendapatkan informasi dataset dari konfigurasi dan pengecekan lokal.
        
        Returns:
            Dict informasi dataset
        """
        try:
            # Dapatkan info lokal jika tersedia
            is_available = self._is_dataset_available()
            
            # Ukuran dataset lokal
            local_stats = self._get_local_stats() if is_available else {}
            
            # Gabungkan dengan informasi dari konfigurasi
            roboflow_config = self.config.get('data', {}).get('roboflow', {})
            
            info = {
                'name': self.project,
                'workspace': self.workspace,
                'version': self.version,
                'is_available_locally': is_available,
                'local_stats': local_stats,
                'config': {
                    'workspace': roboflow_config.get('workspace'),
                    'project': roboflow_config.get('project'),
                    'version': roboflow_config.get('version')
                }
            }
            
            # Log informasi dataset
            if is_available:
                self.logger.info(
                    f"🔍 Dataset Info (Lokal):\n"
                    f"   Nama: {info['name']} (Versi: {info['version']})\n"
                    f"   Workspace: {info['workspace']}\n"
                    f"   Total gambar: {sum(local_stats.values())}\n"
                    f"   Train: {local_stats.get('train', 0)} gambar\n"
                    f"   Valid: {local_stats.get('valid', 0)} gambar\n"
                    f"   Test: {local_stats.get('test', 0)} gambar"
                )
            else:
                self.logger.info(
                    f"🔍 Dataset Info (akan didownload):\n"
                    f"   Nama: {info['name']} (Versi: {info['version']})\n"
                    f"   Workspace: {info['workspace']}\n"
                    f"   Status: Belum tersedia, akan didownload dari Roboflow"
                )
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan info dataset: {str(e)}")
            return {
                'name': self.project,
                'workspace': self.workspace,
                'version': self.version,
                'is_available_locally': False,
                'error': str(e)
            }
    
    def _is_dataset_available(self) -> bool:
        """
        Cek apakah dataset sudah tersedia di lokal.
        
        Returns:
            True jika dataset tersedia
        """
        # Cek keberadaan direktori dan file minimal
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_dir / split
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not (split_dir.exists() and images_dir.exists() and labels_dir.exists()):
                return False
                
            # Cek keberadaan file gambar
            image_files = list(images_dir.glob('*.jpg')) + \
                         list(images_dir.glob('*.jpeg')) + \
                         list(images_dir.glob('*.png'))
                         
            if len(image_files) == 0:
                return False
        
        return True
    
    def _get_local_stats(self) -> Dict[str, int]:
        """
        Dapatkan statistik dataset lokal.
        
        Returns:
            Dict berisi jumlah file per split
        """
        stats = {}
        
        for split in ['train', 'valid', 'test']:
            images_dir = self.data_dir / split / 'images'
            if images_dir.exists():
                image_count = len(list(images_dir.glob('*.jpg'))) + \
                              len(list(images_dir.glob('*.jpeg'))) + \
                              len(list(images_dir.glob('*.png')))
                stats[split] = image_count
            else:
                stats[split] = 0
                
        return stats