"""
File: smartcash/utils/dataset/dataset_cleaner.py
Author: Alfrida Sabar
Deskripsi: Modul untuk membersihkan dataset, menghapus file augmentasi atau file yang tidak valid
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import multiprocessing as mp

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset.dataset_utils import DatasetUtils

class DatasetCleaner:
    """
    Kelas untuk membersihkan dataset dari file augmentasi atau yang tidak valid.
    """
    
    def __init__(
        self,
        config: Dict,
        data_dir: Optional[Union[str, Path]] = None,
        logger: Optional[SmartCashLogger] = None,
        num_workers: Optional[int] = None
    ):
        """
        Inisialisasi dataset cleaner.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori dataset
            logger: Logger kustom
            num_workers: Jumlah worker untuk paralelisasi
        """
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        
        # Setup path
        self.data_dir = Path(data_dir) if data_dir else Path(config.get('data_dir', 'data'))
        
        # Setup utilitas dataset
        self.utils = DatasetUtils(logger)
        
        # Setup multiprocessing
        if num_workers is None:
            self.num_workers = self._calculate_workers()
        else:
            self.num_workers = num_workers
            
        # Pattern pembersihan
        self.augmentation_patterns = self._get_cleanup_patterns(config)
        self.ignored_patterns = self._get_ignored_patterns(config)
        
        # Statistik pembersihan
        self.stats = {
            'before': {},
            'removed': {},
            'after': {}
        }
    
    def _calculate_workers(self) -> int:
        """
        Hitung jumlah optimal worker berdasarkan CPU dan memori.
        
        Returns:
            Jumlah worker optimal
        """
        cpu_count = mp.cpu_count()
        
        # Gunakan memori limit dari config jika ada
        if 'model' in self.config and 'memory_limit' in self.config['model']:
            memory_factor = self.config['model']['memory_limit']
        else:
            memory_factor = 0.75  # Default menggunakan 75% CPU
            
        return max(1, int(cpu_count * memory_factor))
    
    def _get_cleanup_patterns(self, config: Dict) -> List[str]:
        """
        Dapatkan pola file augmentasi dari config.
        
        Args:
            config: Konfigurasi aplikasi
            
        Returns:
            List pola file augmentasi
        """
        # Default patterns
        default_patterns = [
            r'aug_.*',
            r'.*_augmented.*',
            r'.*_modified.*'
        ]
        
        # Cek config
        if 'cleanup' in config and 'augmentation_patterns' in config['cleanup']:
            return config['cleanup']['augmentation_patterns']
        
        return default_patterns
    
    def _get_ignored_patterns(self, config: Dict) -> List[str]:
        """
        Dapatkan pola file yang harus diabaikan dari config.
        
        Args:
            config: Konfigurasi aplikasi
            
        Returns:
            List pola file yang diabaikan
        """
        # Default patterns
        default_patterns = [
            r'.*\.gitkeep',
            r'.*\.DS_Store'
        ]
        
        # Cek config
        if 'cleanup' in config and 'ignored_patterns' in config['cleanup']:
            return config['cleanup']['ignored_patterns']
        
        return default_patterns
    
    def _is_augmented_file(self, filename: str) -> bool:
        """
        Cek apakah file merupakan hasil augmentasi.
        
        Args:
            filename: Nama file
            
        Returns:
            Boolean yang menunjukkan apakah file adalah hasil augmentasi
        """
        return any(re.match(pattern, filename) for pattern in self.augmentation_patterns)
    
    def _should_ignore_file(self, filename: str) -> bool:
        """
        Cek apakah file perlu diabaikan.
        
        Args:
            filename: Nama file
            
        Returns:
            Boolean yang menunjukkan apakah file perlu diabaikan
        """
        return any(re.match(pattern, filename) for pattern in self.ignored_patterns)
    
    def _collect_files(self, augmented_only: bool = True) -> Dict[str, Set[Path]]:
        """
        Mengumpulkan semua file yang perlu dibersihkan.
        
        Args:
            augmented_only: Hanya file hasil augmentasi
            
        Returns:
            Dict file yang perlu dibersihkan per kategori
        """
        files_to_clean = {'images': set(), 'labels': set()}
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_dir / split
            
            # Cek file gambar
            img_dir = split_dir / 'images'
            if img_dir.exists():
                for img_path in img_dir.glob('*'):
                    # Skip direktori
                    if img_path.is_dir():
                        continue
                        
                    # Skip file yang diabaikan
                    if self._should_ignore_file(img_path.name):
                        continue
                    
                    # Filter berdasarkan augmented_only flag
                    if augmented_only and not self._is_augmented_file(img_path.name):
                        continue
                        
                    # Tambahkan file ke set yang akan dibersihkan
                    files_to_clean['images'].add(img_path)
                    
                    # Cari label yang sesuai
                    label_path = (split_dir / 'labels' / img_path.stem).with_suffix('.txt')
                    if label_path.exists():
                        files_to_clean['labels'].add(label_path)
        
        return files_to_clean
    
    def _update_stats(self, files_to_clean: Dict[str, Set[Path]]) -> None:
        """
        Update statistik pembersihan.
        
        Args:
            files_to_clean: File yang akan dibersihkan
        """
        # Statistik sebelum
        self.stats['before'] = {
            'images': len(list(self.data_dir.rglob('images/*.*'))),
            'labels': len(list(self.data_dir.rglob('labels/*.*')))
        }
        
        # Yang dihapus
        self.stats['removed'] = {
            'images': len(files_to_clean['images']),
            'labels': len(files_to_clean['labels'])
        }
        
        # Statistik setelah
        self.stats['after'] = {
            'images': self.stats['before']['images'] - self.stats['removed']['images'],
            'labels': self.stats['before']['labels'] - self.stats['removed']['labels']
        }
    
    def _remove_file(self, file_path: Path) -> bool:
        """
        Hapus satu file dengan aman.
        
        Args:
            file_path: Path file yang akan dihapus
            
        Returns:
            Boolean yang menunjukkan keberhasilan penghapusan
        """
        try:
            file_path.unlink()
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menghapus {file_path}: {str(e)}")
            return False
    
    def cleanup(
        self,
        augmented_only: bool = True,
        create_backup: bool = True,
        backup_dir: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Bersihkan dataset.
        
        Args:
            augmented_only: Hanya hapus file hasil augmentasi
            create_backup: Buat backup sebelum menghapus
            backup_dir: Direktori backup (opsional)
            
        Returns:
            Dict statistik pembersihan
        """
        self.logger.start("🧹 Memulai pembersihan dataset...")
        
        try:
            # Kumpulkan file yang akan dibersihkan
            files_to_clean = self._collect_files(augmented_only)
            
            # Update statistik awal
            self._update_stats(files_to_clean)
            
            # Buat backup jika diminta
            if create_backup:
                # Gabungkan semua file
                all_files = list(files_to_clean['images']) + list(files_to_clean['labels'])
                
                if all_files:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = Path(backup_dir) if backup_dir else self.data_dir / 'backup'
                    backup_subdir = backup_path / f"cleanup_{timestamp}"
                    
                    self.logger.info(f"📦 Membuat backup di {backup_subdir}")
                    
                    # Buat struktur direktori backup
                    for file_type in ['images', 'labels']:
                        (backup_subdir / file_type).mkdir(parents=True, exist_ok=True)
                    
                    # Salin semua file
                    for file_path in all_files:
                        file_type = file_path.parent.name  # 'images' or 'labels'
                        target_dir = backup_subdir / file_type
                        target_path = target_dir / file_path.name
                        
                        # Salin file
                        target_path.write_bytes(file_path.read_bytes())
            
            # Hapus file dengan multiprocessing
            total_files = sum(len(files) for files in files_to_clean.values())
            
            if total_files == 0:
                self.logger.info("ℹ️ Tidak ada file yang perlu dibersihkan")
                return self.stats
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                with tqdm(total=total_files, desc="🗑️ Menghapus file") as pbar:
                    # Proses images
                    for success in executor.map(
                        self._remove_file,
                        files_to_clean['images']
                    ):
                        pbar.update(1)
                    
                    # Proses labels
                    for success in executor.map(
                        self._remove_file,
                        files_to_clean['labels']
                    ):
                        pbar.update(1)
            
            # Log statistik
            self.logger.success(
                f"✨ Pembersihan selesai!\n"
                f"📊 Statistik:\n"
                f"   Sebelum: {self.stats['before']['images']} gambar, "
                f"{self.stats['before']['labels']} label\n"
                f"   Dihapus: {self.stats['removed']['images']} gambar, "
                f"{self.stats['removed']['labels']} label\n"
                f"   Setelah: {self.stats['after']['images']} gambar, "
                f"{self.stats['after']['labels']} label"
            )
            
            return self.stats
            
        except Exception as e:
            self.logger.error(f"❌ Pembersihan gagal: {str(e)}")
            raise