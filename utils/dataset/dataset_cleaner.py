"""
File: smartcash/utils/dataset/dataset_cleaner.py
Author: Alfrida Sabar
Deskripsi: Modul untuk pembersihan dataset dan penghapusan file yang tidak diperlukan
"""

import re
from pathlib import Path
from typing import Dict, Optional, Union, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import multiprocessing as mp

from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset.dataset_utils import DatasetUtils, DEFAULT_SPLITS

class DatasetCleaner:
    """Kelas untuk pembersihan dataset dan file augmentasi."""
    
    def __init__(self, config: Dict, data_dir: Optional[Union[str, Path]] = None, 
                logger: Optional[SmartCashLogger] = None, num_workers: Optional[int] = None):
        """Inisialisasi DatasetCleaner."""
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.utils = DatasetUtils(config, data_dir, logger)
        
        # Setup workers dan pattern
        self.num_workers = num_workers or max(1, int(mp.cpu_count() * 0.75))
        self.augmentation_patterns = config.get('cleanup', {}).get('augmentation_patterns', 
                                              [r'aug_.*', r'.*_augmented.*', r'.*_modified.*'])
        self.ignored_patterns = config.get('cleanup', {}).get('ignored_patterns', 
                                         [r'.*\.gitkeep', r'.*\.DS_Store'])
        
        # Statistik pembersihan
        self.stats = {'before': {}, 'removed': {}, 'after': {}}

    def _is_augmented_file(self, filename: str) -> bool:
        """Cek apakah file termasuk hasil augmentasi berdasarkan pattern."""
        return any(re.match(p, filename) for p in self.augmentation_patterns)

    def _should_ignore_file(self, filename: str) -> bool:
        """Cek apakah file harus diabaikan berdasarkan pattern."""
        return any(re.match(p, filename) for p in self.ignored_patterns)

    def _collect_files(self, augmented_only: bool = True) -> Dict[str, Set[Path]]:
        """Kumpulkan file yang akan dibersihkan."""
        files = {'images': set(), 'labels': set()}
        
        for split in DEFAULT_SPLITS:
            split_dir = self.utils.get_split_path(split)
            img_dir = split_dir / 'images'
            
            if img_dir.exists():
                for img_path in img_dir.glob('*'):
                    # Skip jika direktori, file yang diabaikan, atau bukan augmentasi (jika augmented_only=True)
                    if (img_path.is_dir() or 
                        self._should_ignore_file(img_path.name) or 
                        (augmented_only and not self._is_augmented_file(img_path.name))):
                        continue
                        
                    files['images'].add(img_path)
                    
                    # Tambahkan label yang sesuai jika ada
                    label_path = (split_dir / 'labels' / img_path.stem).with_suffix('.txt')
                    if label_path.exists():
                        files['labels'].add(label_path)
        
        return files

    def _update_stats(self, files_to_clean: Dict[str, Set[Path]]) -> None:
        """Update statistik sebelum dan sesudah pembersihan."""
        self.stats['before'] = {
            'images': len(list(self.data_dir.rglob('images/*.*'))), 
            'labels': len(list(self.data_dir.rglob('labels/*.*')))
        }
        self.stats['removed'] = {
            'images': len(files_to_clean['images']), 
            'labels': len(files_to_clean['labels'])
        }
        self.stats['after'] = {
            k: self.stats['before'][k] - self.stats['removed'][k] 
            for k in self.stats['before']
        }

    def _remove_file(self, file_path: Path) -> bool:
        """Hapus file dengan handling error."""
        try:
            file_path.unlink()
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menghapus {file_path}: {str(e)}")
            return False

    def cleanup(self, augmented_only: bool = True, create_backup: bool = True, 
              backup_dir: Optional[Union[str, Path]] = None) -> Dict:
        """
        Bersihkan dataset dengan menghapus file yang tidak diperlukan.
        
        Args:
            augmented_only: Hanya hapus file hasil augmentasi
            create_backup: Buat backup sebelum menghapus
            backup_dir: Direktori untuk backup
            
        Returns:
            Dict statistik pembersihan
        """
        self.logger.start("🧹 Memulai pembersihan dataset...")
        
        # Kumpulkan file yang akan dibersihkan
        files_to_clean = self._collect_files(augmented_only)
        self._update_stats(files_to_clean)
        
        # Buat backup jika diminta
        if create_backup and any(len(files) > 0 for files in files_to_clean.values()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(backup_dir or self.data_dir / 'backup') / f"cleanup_{timestamp}"
            self.logger.info(f"📦 Membuat backup di {backup_path}")
            
            # Buat direktori backup dan salin file
            for file_type in ['images', 'labels']:
                if files_to_clean[file_type]:
                    (backup_path / file_type).mkdir(parents=True, exist_ok=True)
                    for file_path in files_to_clean[file_type]:
                        try:
                            (backup_path / file_type / file_path.name).write_bytes(file_path.read_bytes())
                        except Exception as e:
                            self.logger.warning(f"⚠️ Gagal backup {file_path.name}: {str(e)}")

        # Hitung total file dan lakukan pembersihan
        total_files = sum(len(files) for files in files_to_clean.values())
        
        if total_files == 0:
            self.logger.info("ℹ️ Tidak ada file yang perlu dibersihkan")
            return self.stats

        # Hapus file secara paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=total_files, desc="🗑️ Menghapus file") as pbar:
                for _ in executor.map(self._remove_file, 
                                     list(files_to_clean['images']) + list(files_to_clean['labels'])):
                    pbar.update(1)

        # Log hasil
        self.logger.success(
            f"✨ Pembersihan selesai!\n"
            f"📊 Statistik: "
            f"Sebelum: {self.stats['before']}, "
            f"Dihapus: {self.stats['removed']}, "
            f"Setelah: {self.stats['after']}"
        )
        
        return self.stats