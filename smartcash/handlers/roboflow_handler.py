# File: smartcash/handlers/roboflow_handler.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk download dan setup dataset dari Roboflow API dengan perbaikan progress bar

import os
import dotenv
from typing import Dict, Optional, Tuple, Union
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from roboflow import Roboflow
from tqdm import tqdm
import shutil
import requests

from smartcash.utils.logger import SmartCashLogger

class RoboflowHandler:
    """Handler untuk mengelola dataset dari Roboflow API"""
    
     
    def __init__(
        self,
        config: Optional[Dict] = None,  # 👈 Tambahkan parameter config
        config_path: str = "configs/base_config.yaml",
        data_dir: str = "data",
        api_key: Optional[str] = None,
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        
        # Load environment variables
        dotenv.load_dotenv()
        
        # Prioritaskan config yang diberikan langsung, jika tidak ada baru load dari file
        if config and isinstance(config, dict):
            # Ambil konfigurasi Roboflow dari parameter config yang diberikan
            self.config = self._extract_roboflow_config(config)
            self.logger.info("✅ Menggunakan konfigurasi dari parameter config")
        else:
            # Load config dari file konfigurasi
            self.config = self._load_config(config_path)
            
        # Setup data directory
        self.data_dir = Path(data_dir)
        
        # Set API Key dengan prioritas:
        # 1. Parameter api_key
        # 2. Config dari roboflow.api_key
        # 3. Environment variable ROBOFLOW_API_KEY
        self.api_key = api_key or self.config.get('api_key') or os.getenv("ROBOFLOW_API_KEY")
        
        if not self.api_key:
            err_msg = (
                "Roboflow API key tidak ditemukan. "
                "Mohon set ROBOFLOW_API_KEY di file .env atau berikan API key langsung"
            )
            self.logger.error(err_msg)
            raise ValueError(err_msg)
        
        # Inisialisasi Roboflow API client
        self.rf = Roboflow(api_key=self.api_key)
        self.logger.info("✅ Roboflow API client berhasil diinisialisasi")
    
    def _extract_roboflow_config(self, config: Dict) -> Dict:
        """Ekstrak konfigurasi Roboflow dari dictionary config"""
        roboflow_config = {}
        
        # Periksa beberapa lokasi yang mungkin menyimpan konfigurasi Roboflow
        if 'roboflow' in config and isinstance(config['roboflow'], dict):
            roboflow_config = config['roboflow']
        elif 'data' in config and 'roboflow' in config['data'] and isinstance(config['data']['roboflow'], dict):
            roboflow_config = config['data']['roboflow']
        
        # Pastikan konfigurasi minimal ada
        if not roboflow_config.get('workspace') or not roboflow_config.get('project'):
            # Gunakan nilai default
            roboflow_config.setdefault('workspace', 'smartcash-wo2us')
            roboflow_config.setdefault('project', 'rupiah-emisi-2022')
            roboflow_config.setdefault('version', '3')
            
        return roboflow_config
    
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi dataset dari file config"""
        try:
            # Load konfigurasi
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            # Ekstrak konfigurasi Roboflow, periksa beberapa lokasi umum
            for key in ['roboflow', 'dataset', 'data']:
                if key in full_config and isinstance(full_config[key], dict):
                    if 'workspace' in full_config[key] and 'project' in full_config[key]:
                        return full_config[key]
                    # Periksa nested roboflow config
                    elif 'roboflow' in full_config[key] and isinstance(full_config[key]['roboflow'], dict):
                        return full_config[key]['roboflow']
            
            # Jika tidak ada, return minimal default config
            return {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3'
            }
            
        except (FileNotFoundError, yaml.YAMLError) as e:
            self.logger.error(f"❌ Gagal memuat konfigurasi: {str(e)}")
            # Return default config
            return {
                'workspace': 'smartcash-wo2us',
                'project': 'rupiah-emisi-2022',
                'version': '3'
            }
    
    def download_dataset(self, format: str = "yolov5", show_progress: bool = True) -> Union[str, Dict]:
        """
        Download dataset dari Roboflow dengan progress bar
        
        Args:
            format: Format dataset ('yolov5', 'coco', etc)
            show_progress: Tampilkan progress bar
            
        Returns:
            Path to downloaded dataset atau informasi dataset
        """
        self.logger.start(
            f"🔄 Mendownload dataset dari Roboflow...\n"
            f"   Workspace: {self.config.get('workspace')}\n"
            f"   Project: {self.config.get('project')}\n"
            f"   Version: {self.config.get('version')}"
        )
        
        try:
            # Get project
            workspace = self.config.get('workspace', 'detection-twl6q')
            project_name = self.config.get('project', 'rupiah_emisi-baru')
            version = self.config.get('version', '3')
            
            project = self.rf.workspace(workspace).project(project_name)
            
            # Get version info untuk tampilan progress
            self.logger.info(f"🔍 Mendapatkan informasi dataset...")
            version_info = project.version(version)
            self.logger.info(f"📊 Dataset info: {version_info.id} ({version_info.created})")
            
            # Download dengan progress monitoring custom
            download_dir = self.data_dir / f"roboflow_{project_name}_{version}"
            download_dir.mkdir(parents=True, exist_ok=True)
            
            if show_progress:
                self.logger.info(f"📥 Memulai download dataset...")
                # Download dengan progress tracking custom
                dataset_info = self._download_with_progress(project, version, format, download_dir)
            else:
                # Download tanpa progress tracking
                dataset = project.version(version).download(format)
                dataset_info = {
                    'location': dataset.location,
                    'splits': {
                        'train': dataset.splits['train'],
                        'valid': dataset.splits['valid'],
                        'test': dataset.splits['test']
                    }
                }
            
            self.logger.success(
                f"✅ Dataset berhasil diunduh ke {dataset_info['location']} 📥\n"
                f"   Train: {dataset_info['splits']['train']} gambar\n"
                f"   Valid: {dataset_info['splits']['valid']} gambar\n"
                f"   Test: {dataset_info['splits']['test']} gambar"
            )
            
            return dataset_info['location']
            
        except Exception as e:
            self.logger.error(f"❌ Gagal download dataset: {str(e)}")
            raise
    
    def _download_with_progress(
        self, 
        project, 
        version: str, 
        format: str,
        output_dir: Path
    ) -> Dict:
        """
        Download dataset dengan progress tracking custom.
        
        Args:
            project: Instance project Roboflow
            version: Versi dataset
            format: Format dataset
            output_dir: Direktori output
            
        Returns:
            Dict info dataset
        """
        try:
            # Get version info
            version_obj = project.version(version)
            
            # Request download URL
            download_url = version_obj._generate_download_url(format=format)
            
            # Temporary zip file
            zip_path = output_dir / "dataset.zip"
            
            # Download dengan progress
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
            
            # Return info
            return {
                'location': str(output_dir),
                'splits': {
                    'train': version_obj.train_count,
                    'valid': version_obj.valid_count,
                    'test': version_obj.test_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Gagal download dengan progress: {str(e)}")
            raise
    
    def export_to_local(self, roboflow_dir: str, show_progress: bool = True) -> None:
        """
        Export dataset Roboflow ke struktur folder lokal dengan progress bar.
        
        Args:
            roboflow_dir: Path ke direktori Roboflow
            show_progress: Tampilkan progress bar
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
            
            for split in ['train', 'valid', 'test']:
                # Source dan target paths
                src_dir = rf_path / split
                target_dir = self.data_dir / split
                
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
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengexport dataset: {str(e)}")
            raise e
    
    def pull_dataset(self, show_progress: bool = True) -> Tuple[str, str, str]:
        """
        Download dan setup dataset dari Roboflow dengan progress tracking.
        
        Args:
            show_progress: Tampilkan progress bar
            
        Returns:
            Tuple paths (train, valid, test)
        """
        try:
            # Tampilkan info dataset
            self.get_dataset_info()
            
            # Download dari Roboflow
            roboflow_dir = self.download_dataset(show_progress=show_progress)
            
            # Export ke struktur folder lokal
            self.export_to_local(roboflow_dir, show_progress=show_progress)
            
            # Return paths
            return (
                str(self.data_dir / 'train'),
                str(self.data_dir / 'valid'),
                str(self.data_dir / 'test')
            )
            
        except Exception as e:
            self.logger.error(f"❌ Gagal melakukan pull dataset: {str(e)}")
            raise e
    
    def get_dataset_info(self) -> Dict:
        """
        Mendapatkan informasi dataset dari Roboflow.
        
        Returns:
            Dict informasi dataset
        """
        try:
            workspace = self.config.get('workspace', 'detection-twl6q')
            project_name = self.config.get('project', 'rupiah_emisi-baru')
            version = self.config.get('version', '3')
            
            project = self.rf.workspace(workspace).project(project_name)
            version_obj = project.version(version)
            
            info = {
                'name': project.name,
                'id': project.id,
                'version': version_obj.version,
                'created': version_obj.created,
                'classes': self.config.get('classes', []),
                'splits': {
                    'train': version_obj.train_count,
                    'valid': version_obj.valid_count,
                    'test': version_obj.test_count
                },
                'total_images': version_obj.train_count + version_obj.valid_count + version_obj.test_count
            }
            
            self.logger.info(
                f"🔍 Dataset Info:\n"
                f"   Nama: {info['name']} (ID: {info['id']})\n"
                f"   Versi: {info['version']} ({info['created']})\n"
                f"   Total gambar: {info['total_images']}\n"
                f"   Train: {info['splits']['train']} gambar\n"
                f"   Valid: {info['splits']['valid']} gambar\n"
                f"   Test: {info['splits']['test']} gambar"
            )
            
            return info
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mendapatkan info dataset: {str(e)}")
            raise e

    def get_train_loader(self, batch_size: int = 32, num_workers: int = 4, dataset_path: Optional[str] = None) -> DataLoader:
        """
        Dapatkan data loader untuk training.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            dataset_path: Custom dataset path (opsional)
            
        Returns:
            DataLoader untuk training
        """
        # Jika dataset belum ada, pull dataset terlebih dahulu
        train_path, _, _ = self._ensure_dataset_available()
        dataset_path = dataset_path or train_path
        
        # Gunakan DataHandler untuk membuat loader
        from smartcash.handlers.data_handler import DataHandler
        data_handler = DataHandler(
            config=self.config,
            data_dir=str(self.data_dir)
        )
        
        return data_handler.get_train_loader(batch_size, num_workers, dataset_path)

    def get_val_loader(self, batch_size: int = 32, num_workers: int = 4, dataset_path: Optional[str] = None) -> DataLoader:
        """
        Dapatkan data loader untuk validasi.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            dataset_path: Custom dataset path (opsional)
            
        Returns:
            DataLoader untuk validasi
        """
        # Jika dataset belum ada, pull dataset terlebih dahulu
        _, val_path, _ = self._ensure_dataset_available()
        dataset_path = dataset_path or val_path
        
        # Gunakan DataHandler untuk membuat loader
        from smartcash.handlers.data_handler import DataHandler
        data_handler = DataHandler(
            config=self.config,
            data_dir=str(self.data_dir)
        )
        
        return data_handler.get_val_loader(batch_size, num_workers, dataset_path)

    def get_test_loader(self, batch_size: int = 32, num_workers: int = 4, dataset_path: Optional[str] = None) -> DataLoader:
        """
        Dapatkan data loader untuk testing.
        
        Args:
            batch_size: Ukuran batch
            num_workers: Jumlah worker
            dataset_path: Custom dataset path (opsional)
            
        Returns:
            DataLoader untuk testing
        """
        # Jika dataset belum ada, pull dataset terlebih dahulu
        _, _, test_path = self._ensure_dataset_available()
        dataset_path = dataset_path or test_path
        
        # Gunakan DataHandler untuk membuat loader
        from smartcash.handlers.data_handler import DataHandler
        data_handler = DataHandler(
            config=self.config,
            data_dir=str(self.data_dir)
        )
        
        return data_handler.get_test_loader(batch_size, num_workers, dataset_path)

    def _ensure_dataset_available(self) -> Tuple[str, str, str]:
        """
        Pastikan dataset sudah tersedia, jika belum akan mendownload.
        
        Returns:
            Tuple (train_path, val_path, test_path)
        """
        # Cek apakah dataset sudah ada
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'valid'
        test_dir = self.data_dir / 'test'
        
        # Jika salah satu tidak ada, download dataset
        if not (train_dir.exists() and val_dir.exists() and test_dir.exists()):
            self.logger.info("🔍 Dataset belum tersedia, mendownload dari Roboflow...")
            train_path, val_path, test_path = self.pull_dataset(show_progress=True)
        else:
            train_path = str(train_dir)
            val_path = str(val_dir)
            test_path = str(test_dir)
            self.logger.info("✅ Dataset sudah tersedia di lokal")
        
        return train_path, val_path, test_path