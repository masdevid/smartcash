# File: smartcash/handlers/dataset/core/dataset_downloader.py
# Author: Alfrida Sabar
# Deskripsi: Downloader dataset dari berbagai sumber seperti Roboflow dengan dukungan paralel

import os
import dotenv
import shutil
import json
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.utils.logger import get_logger
from smartcash.config import get_config_manager
from smartcash.handlers.dataset.core.download_manager import DownloadManager

class DatasetDownloader:
    """
    Downloader dataset dari berbagai sumber seperti Roboflow, Kaggle, atau custom URL.
    Mendukung konversi format dan ekstraksi dari berbagai sumber.
    """
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        data_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        logger: Optional = None,
        num_workers: int = 4
    ):
        """
        Inisialisasi DatasetDownloader.
        
        Args:
            config: Konfigurasi untuk download
            data_dir: Direktori target dataset
            api_key: API key untuk layanan eksternal (misalnya Roboflow)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk download paralel
        """
        self.logger = logger or get_logger("dataset_downloader")
        
        # Load environment variables untuk API key
        dotenv.load_dotenv()
        
        # Prioritaskan config yang diberikan langsung, jika tidak ada baru load dari config manager
        if config:
            self.config = config
        else:
            config_manager = get_config_manager()
            self.config = config_manager.get_config()
            
        # Setup data directory
        self.data_dir = Path(data_dir if data_dir else self.config.get('data_dir', 'data'))
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Direktori untuk file sementara
        self.temp_dir = self.data_dir / ".temp"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Set API Key dengan prioritas:
        # 1. Parameter api_key
        # 2. Config dari roboflow.api_key
        # 3. Environment variable ROBOFLOW_API_KEY
        self.api_key = api_key or \
                      self.config.get('data', {}).get('roboflow', {}).get('api_key') or \
                      os.getenv("ROBOFLOW_API_KEY")
        
        if not self.api_key:
            self.logger.warning("⚠️ Roboflow API key tidak ditemukan. Beberapa fitur mungkin tidak berfungsi.")
            
        # Inisialisasi download manager
        self.download_manager = DownloadManager(
            output_dir=self.temp_dir,
            num_workers=num_workers,
            logger=self.logger
        )
        
        # Siapkan informasi Roboflow
        roboflow_config = self.config.get('data', {}).get('roboflow', {})
        self.workspace = roboflow_config.get('workspace', 'smartcash-wo2us')
        self.project = roboflow_config.get('project', 'rupiah-emisi-2022')
        self.version = roboflow_config.get('version', '3')
        
        self.logger.info(
            f"🔧 DatasetDownloader diinisialisasi dengan {num_workers} workers:\n"
            f"   • Data dir: {self.data_dir}\n"
            f"   • Roboflow: {self.workspace}/{self.project}:{self.version}"
        )
    
    def download_dataset(
        self, 
        format: str = "yolov5", 
        show_progress: bool = True,
        resume: bool = True
    ) -> str:
        """
        Download dataset dari Roboflow dengan progress bar dan dukungan resume.
        
        Args:
            format: Format dataset ('yolov5', 'coco', 'pascal', dll)
            show_progress: Tampilkan progress bar
            resume: Coba resume download jika sebelumnya gagal
            
        Returns:
            Path ke direktori dataset yang didownload
        """
        if not self.api_key:
            raise ValueError("Roboflow API key diperlukan untuk download dataset")
            
        self.logger.start(
            f"🔄 Mendownload dataset dari Roboflow...\n"
            f"   Workspace: {self.workspace}\n"
            f"   Project: {self.project}\n"
            f"   Version: {self.version}\n"
            f"   Format: {format}"
        )
        
        try:
            # Persiapkan direktori output
            download_dir = self.data_dir / f"roboflow_{self.project}_{self.version}"
            os.makedirs(download_dir, exist_ok=True)
            
            # Buat URL download menggunakan Roboflow API
            download_url = self._generate_download_url(format)
            
            # Path untuk file zip dataset
            zip_filename = f"{self.project}_{self.version}_{format}.zip"
            zip_path = self.temp_dir / zip_filename
            
            # Cek apakah dataset sudah pernah didownload dan diekstrak
            dataset_info_path = download_dir / "dataset_info.json"
            
            if download_dir.exists() and dataset_info_path.exists():
                try:
                    with open(dataset_info_path, 'r') as f:
                        info = json.load(f)
                    
                    if (
                        info.get('completed', False) and 
                        info.get('workspace') == self.workspace and
                        info.get('project') == self.project and
                        info.get('version') == self.version and
                        info.get('format') == format
                    ):
                        self.logger.info(f"✅ Dataset sudah ada di {download_dir}")
                        return str(download_dir)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Info file rusak atau tidak ada, lanjutkan download
                    pass
            
            # Download dataset dengan download manager
            self.download_manager.download_file(
                url=download_url,
                output_path=zip_path,
                resume=resume,
                progress_bar=show_progress,
                metadata={
                    'workspace': self.workspace,
                    'project': self.project,
                    'version': self.version,
                    'format': format
                }
            )
            
            # Ekstrak dataset
            self.logger.info(f"📦 Mengekstrak dataset...")
            self.download_manager.extract_zip(
                zip_path=zip_path,
                output_dir=download_dir,
                remove_zip=True,
                show_progress=show_progress
            )
            
            # Simpan info dataset
            dataset_info = {
                'workspace': self.workspace,
                'project': self.project,
                'version': self.version,
                'format': format,
                'timestamp': time.time(),
                'completed': True,
                'splits': self._count_dataset_files(download_dir)
            }
            
            with open(dataset_info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            # Log hasil
            splits_info = dataset_info.get('splits', {})
            self.logger.success(
                f"✅ Dataset berhasil diunduh ke {download_dir} 📥\n"
                f"   Train: {splits_info.get('train', 'N/A')} gambar\n"
                f"   Valid: {splits_info.get('valid', 'N/A')} gambar\n"
                f"   Test: {splits_info.get('test', 'N/A')} gambar"
            )
            
            return str(download_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal download dataset: {str(e)}")
            raise
    
    def download_dataset_format(
        self,
        format: str = "yolov5",
        output_dir: Optional[str] = None,
        show_progress: bool = True,
        resume: bool = True
    ) -> str:
        """
        Download dataset dalam format tertentu (yolov5, coco, pascal, dll).
        
        Args:
            format: Format dataset yang diinginkan
            output_dir: Direktori output kustom (opsional)
            show_progress: Tampilkan progress bar
            resume: Coba resume download jika sebelumnya gagal
            
        Returns:
            Path ke direktori dataset yang didownload
        """
        # Tentukan direktori output
        if output_dir:
            target_dir = Path(output_dir)
        else:
            target_dir = self.data_dir / f"{self.project}_{format}"
        
        # Berikan ke download_dataset dasar
        return self.download_dataset(
            format=format,
            show_progress=show_progress,
            resume=resume
        )
    
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
        show_progress: bool = True,
        num_workers: int = 4
    ) -> Tuple[str, str, str]:
        """
        Export dataset Roboflow ke struktur folder lokal yang standar.
        Mendukung copy atau symlink paralel.
        
        Args:
            roboflow_dir: Direktori dataset Roboflow
            show_progress: Tampilkan progress bar
            num_workers: Jumlah worker untuk proses paralel
            
        Returns:
            Tuple (train_dir, valid_dir, test_dir)
        """
        self.logger.start("📤 Mengexport dataset ke struktur folder lokal...")
        
        try:
            rf_path = Path(roboflow_dir)
            
            # Pastikan direktori tujuan ada
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Mencari semua file untuk estimasi total
            file_copy_tasks = []
            
            for split in ['train', 'valid', 'test']:
                # Source dan target paths
                src_dir = rf_path / split
                target_dir = self.data_dir / split
                
                # Buat direktori target
                os.makedirs(target_dir / 'images', exist_ok=True)
                os.makedirs(target_dir / 'labels', exist_ok=True)
                
                # Tambahkan task untuk copy gambar
                for item in ['images', 'labels']:
                    src_item_dir = src_dir / item
                    
                    if src_item_dir.exists():
                        for file_path in src_item_dir.glob('*'):
                            if file_path.is_file():
                                target_path = target_dir / item / file_path.name
                                
                                # Tambahkan ke task jika belum ada
                                if not target_path.exists():
                                    file_copy_tasks.append((file_path, target_path))
            
            # Copy file secara paralel
            total_files = len(file_copy_tasks)
            
            if total_files == 0:
                self.logger.info("✅ Tidak ada file baru yang perlu di-copy")
            else:
                self.logger.info(f"🔄 Memindahkan {total_files} file...")
                
                # Progress bar
                progress = None
                if show_progress:
                    progress = tqdm(total=total_files, desc="Copying files", unit="file")
                
                # Copy dengan ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    # Submit tasks
                    futures = []
                    for src_path, dest_path in file_copy_tasks:
                        future = executor.submit(shutil.copy2, src_path, dest_path)
                        futures.append(future)
                    
                    # Update progress saat selesai
                    for future in futures:
                        future.result()  # Tunggu selesai
                        if progress:
                            progress.update(1)
                
                # Tutup progress bar
                if progress:
                    progress.close()
            
            # Return paths untuk train, valid, dan test
            output_dirs = [
                str(self.data_dir / 'train'),
                str(self.data_dir / 'valid'),
                str(self.data_dir / 'test')
            ]
            
            self.logger.success(
                f"✅ Dataset berhasil diexport ke struktur folder lokal\n"
                f"   Total file: {total_files}\n"
                f"   Output: {self.data_dir}"
            )
            
            return tuple(output_dirs)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal mengexport dataset: {str(e)}")
            raise e
    
    def pull_dataset(self, 
                    format: str = "yolov5",
                    show_progress: bool = True, 
                    resume: bool = True) -> Tuple[str, str, str]:
        """
        Download dan setup dataset dari Roboflow dengan progress tracking.
        Satu-step untuk mendapatkan dataset siap pakai.
        
        Args:
            format: Format dataset ('yolov5', 'coco', dll)
            show_progress: Tampilkan progress bar
            resume: Coba resume download jika ada
            
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
            roboflow_dir = self.download_dataset(
                format=format,
                show_progress=show_progress,
                resume=resume
            )
            
            # Export ke struktur folder lokal
            return self.export_to_local(roboflow_dir, show_progress)
            
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
    
    def convert_dataset_format(
        self, 
        source_format: str, 
        target_format: str,
        source_dir: Optional[Union[str, Path]] = None,
        target_dir: Optional[Union[str, Path]] = None,
        show_progress: bool = True
    ) -> str:
        """
        Konversi dataset dari satu format ke format lain.
        
        Args:
            source_format: Format sumber ('yolov5', 'coco', dll)
            target_format: Format target
            source_dir: Direktori dataset sumber (opsional)
            target_dir: Direktori dataset target (opsional)
            show_progress: Tampilkan progress bar
            
        Returns:
            Path direktori hasil konversi
        """
        # Saat ini hanya support konversi dengan download ulang dari Roboflow
        # TODO: Implementasi konversi format lokal untuk format umum
        
        self.logger.info(f"🔄 Konversi dataset dari {source_format} ke {target_format}")
        
        # Download dalam format target
        return self.download_dataset(
            format=target_format,
            show_progress=show_progress
        )
    
    def download_dataset_from_custom_source(
        self,
        url: str,
        output_dir: Optional[str] = None,
        format: str = "custom",
        extract: bool = True,
        resume: bool = True,
        show_progress: bool = True
    ) -> str:
        """
        Download dataset dari URL kustom (non-Roboflow).
        
        Args:
            url: URL sumber dataset
            output_dir: Direktori output
            format: Format dataset untuk tracking
            extract: Ekstrak zip jika file merupakan zip
            resume: Coba resume download jika gagal
            show_progress: Tampilkan progress bar
            
        Returns:
            Path ke direktori dataset
        """
        # Tentukan direktori output
        if output_dir:
            target_dir = Path(output_dir)
        else:
            target_dir = self.data_dir / f"custom_{Path(url).name}"
        
        # Buat direktori output
        os.makedirs(target_dir, exist_ok=True)
        
        self.logger.info(f"🔄 Mendownload dataset dari URL kustom: {url}")
        
        # Download dengan download manager
        try:
            # Filename dari URL
            filename = Path(url).name
            if not filename:
                filename = "dataset.zip" if extract else "dataset.dat"
                
            # Download file
            downloaded_file = self.download_manager.download_file(
                url=url,
                output_path=target_dir / filename,
                resume=resume,
                progress_bar=show_progress
            )
            
            # Ekstrak jika diminta dan file adalah zip
            if extract and filename.lower().endswith('.zip'):
                self.logger.info(f"📦 Mengekstrak dataset...")
                target_dir = self.download_manager.extract_zip(
                    zip_path=downloaded_file,
                    output_dir=target_dir,
                    remove_zip=True,
                    show_progress=show_progress
                )
                
            # Cek struktur dataset dan buat info
            dataset_info = {
                'source_url': url,
                'format': format,
                'timestamp': time.time(),
                'completed': True,
                'splits': self._count_dataset_files(target_dir)
            }
            
            # Simpan info
            with open(target_dir / "dataset_info.json", 'w') as f:
                json.dump(dataset_info, f, indent=2)
                
            # Log hasil
            splits_info = dataset_info.get('splits', {})
            self.logger.success(
                f"✅ Dataset berhasil diunduh ke {target_dir} 📥\n"
                f"   Train: {splits_info.get('train', 'N/A')} gambar\n"
                f"   Valid: {splits_info.get('valid', 'N/A')} gambar\n"
                f"   Test: {splits_info.get('test', 'N/A')} gambar"
            )
            
            return str(target_dir)
            
        except Exception as e:
            self.logger.error(f"❌ Gagal download dataset dari URL kustom: {str(e)}")
            raise