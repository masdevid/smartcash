# File: handlers/dataset_cleanup.py
# Author: Alfrida Sabar
# Deskripsi: Handler untuk membersihkan dataset, menghapus file augmentasi atau file yang tidak valid

import re
import shutil
from pathlib import Path
from typing import Dict, Optional, Set
from datetime import datetime
import yaml
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import multiprocessing as mp

from utils.logger import SmartCashLogger

class DatasetCleanupHandler:
    """Handler untuk membersihkan dan memvalidasi dataset"""
    
    def __init__(
        self,
        config_path: str,
        data_dir: str = "data",
        backup_dir: Optional[str] = "backup",
        logger: Optional[SmartCashLogger] = None
    ):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else None
        
        # Statistik pembersihan
        self.stats = {
            'before': {},
            'removed': {},
            'after': {}
        }
        
        # Setup multiprocessing
        self.n_workers = self._calculate_workers()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load konfigurasi pembersihan"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Default cleanup patterns jika tidak ada di config
        if 'cleanup' not in config:
            config['cleanup'] = {
                'augmentation_patterns': [
                    r'aug_.*',
                    r'.*_augmented.*',
                    r'.*_modified.*'
                ],
                'ignored_patterns': [
                    r'.*\.gitkeep',
                    r'.*\.DS_Store'
                ]
            }
            
        return config
        
    def _calculate_workers(self) -> int:
        """Hitung jumlah optimal worker"""
        cpu_count = mp.cpu_count()
        memory_limit = self.config['model']['memory_limit']
        return max(1, int(cpu_count * memory_limit))
        
    def _is_augmented_file(self, filename: str) -> bool:
        """Cek apakah file merupakan hasil augmentasi"""
        patterns = self.config['cleanup']['augmentation_patterns']
        return any(re.match(pattern, filename) for pattern in patterns)
        
    def _should_ignore_file(self, filename: str) -> bool:
        """Cek apakah file perlu diabaikan"""
        patterns = self.config['cleanup']['ignored_patterns']
        return any(re.match(pattern, filename) for pattern in patterns)
        
    def _collect_files(self) -> Dict[str, Set[Path]]:
        """Mengumpulkan semua file yang perlu dibersihkan"""
        files_to_clean = {'images': set(), 'labels': set()}
        
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_dir / split
            
            # Cek file gambar
            img_dir = split_dir / 'images'
            if img_dir.exists():
                for img_path in img_dir.glob('*'):
                    if (self._is_augmented_file(img_path.name) and 
                        not self._should_ignore_file(img_path.name)):
                        files_to_clean['images'].add(img_path)
                        
                        # Cari label yang sesuai
                        label_path = (split_dir / 'labels' / 
                                    img_path.stem).with_suffix('.txt')
                        if label_path.exists():
                            files_to_clean['labels'].add(label_path)
                            
        return files_to_clean
        
    def _backup_files(self, files: Dict[str, Set[Path]]) -> None:
        """Backup file sebelum dihapus"""
        if not self.backup_dir:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"cleanup_{timestamp}"
        
        self.logger.info(f"📦 Membuat backup di {backup_path}")
        
        for file_type, file_paths in files.items():
            target_dir = backup_path / file_type
            target_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in file_paths:
                shutil.copy2(file_path, target_dir / file_path.name)
                
    def _remove_file(self, file_path: Path) -> bool:
        """Hapus satu file dengan aman"""
        try:
            file_path.unlink()
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menghapus {file_path}: {str(e)}")
            return False
            
    def _update_stats(
        self,
        files_to_clean: Dict[str, Set[Path]]
    ) -> None:
        """Update statistik pembersihan"""
        # Statistik sebelum
        self.stats['before'] = {
            'images': len(list(self.data_dir.rglob('images/*'))),
            'labels': len(list(self.data_dir.rglob('labels/*')))
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
        
    def cleanup(
        self,
        augmented_only: bool = True,
        create_backup: bool = True
    ) -> Dict:
        """
        Bersihkan dataset
        Args:
            augmented_only: Hanya hapus file hasil augmentasi
            create_backup: Buat backup sebelum menghapus
        Returns:
            Dict statistik pembersihan
        """
        self.logger.start("🧹 Memulai pembersihan dataset...")
        
        try:
            # Kumpulkan file yang akan dibersihkan
            files_to_clean = self._collect_files()
            
            # Backup jika diperlukan
            if create_backup and self.backup_dir:
                self._backup_files(files_to_clean)
                
            # Update statistik awal
            self._update_stats(files_to_clean)
            
            # Hapus file dengan multiprocessing
            total_files = sum(len(files) for files in files_to_clean.values())
            
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                with tqdm(total=total_files, 
                         desc="🗑️ Menghapus file") as pbar:
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
            raise e