import re, yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm
import multiprocessing as mp
from smartcash.utils.logger import SmartCashLogger
from smartcash.utils.dataset.dataset_utils import DatasetUtils

class DatasetCleaner:
    def __init__(self, config: Dict, data_dir: Optional[Union[str, Path]] = None, logger: Optional[SmartCashLogger] = None, num_workers: Optional[int] = None):
        self.logger = logger or SmartCashLogger(__name__)
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.utils = DatasetUtils(logger)
        self.num_workers = num_workers or max(1, int(mp.cpu_count() * (config.get('model', {}).get('memory_limit', 0.75))))
        self.augmentation_patterns = config.get('cleanup', {}).get('augmentation_patterns', [r'aug_.*', r'.*_augmented.*', r'.*_modified.*'])
        self.ignored_patterns = config.get('cleanup', {}).get('ignored_patterns', [r'.*\.gitkeep', r'.*\.DS_Store'])
        self.stats = {'before': {}, 'removed': {}, 'after': {}}

    def _is_augmented_file(self, filename: str) -> bool:
        return any(re.match(p, filename) for p in self.augmentation_patterns)

    def _should_ignore_file(self, filename: str) -> bool:
        return any(re.match(p, filename) for p in self.ignored_patterns)

    def _collect_files(self, augmented_only: bool = True) -> Dict[str, Set[Path]]:
        files = {'images': set(), 'labels': set()}
        for split in ['train', 'valid', 'test']:
            split_dir = self.data_dir / split
            img_dir = split_dir / 'images'
            if img_dir.exists():
                for img_path in img_dir.glob('*'):
                    if img_path.is_dir() or self._should_ignore_file(img_path.name) or (augmented_only and not self._is_augmented_file(img_path.name)):
                        continue
                    files['images'].add(img_path)
                    label_path = (split_dir / 'labels' / img_path.stem).with_suffix('.txt')
                    if label_path.exists():
                        files['labels'].add(label_path)
        return files

    def _update_stats(self, files_to_clean: Dict[str, Set[Path]]) -> None:
        self.stats['before'] = {'images': len(list(self.data_dir.rglob('images/*.*'))), 'labels': len(list(self.data_dir.rglob('labels/*.*')))}
        self.stats['removed'] = {'images': len(files_to_clean['images']), 'labels': len(files_to_clean['labels'])}
        self.stats['after'] = {k: self.stats['before'][k] - self.stats['removed'][k] for k in self.stats['before']}

    def _remove_file(self, file_path: Path) -> bool:
        try:
            file_path.unlink()
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menghapus {file_path}: {str(e)}")
            return False

    def cleanup(self, augmented_only: bool = True, create_backup: bool = True, backup_dir: Optional[Union[str, Path]] = None) -> Dict:
        self.logger.start("🧹 Memulai pembersihan dataset...")
        files_to_clean = self._collect_files(augmented_only)
        self._update_stats(files_to_clean)
        
        if create_backup and any(files_to_clean.values()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = Path(backup_dir or self.data_dir / 'backup') / f"cleanup_{timestamp}"
            self.logger.info(f"📦 Membuat backup di {backup_path}")
            for file_type in ['images', 'labels']:
                (backup_path / file_type).mkdir(parents=True, exist_ok=True)
                for file_path in files_to_clean[file_type]:
                    (backup_path / file_type / file_path.name).write_bytes(file_path.read_bytes())

        total_files = sum(len(files) for files in files_to_clean.values())
        if total_files == 0:
            self.logger.info("ℹ️ Tidak ada file yang perlu dibersihkan")
            return self.stats

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            with tqdm(total=total_files, desc="🗑️ Menghapus file") as pbar:
                for _ in executor.map(self._remove_file, files_to_clean['images'] | files_to_clean['labels']):
                    pbar.update(1)

        self.logger.success(f"✨ Pembersihan selesai!\n📊 Statistik: Sebelum: {self.stats['before']}, Dihapus: {self.stats['removed']}, Setelah: {self.stats['after']}")
        return self.stats