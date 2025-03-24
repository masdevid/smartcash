"""
File: smartcash/dataset/utils/split/merger.py
Deskripsi: Utilitas untuk menggabungkan beberapa dataset menjadi satu dataset
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Union, Optional
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS


class DatasetMerger:
    """Utilitas untuk menggabungkan beberapa dataset menjadi satu."""
    
    def __init__(self, config: Dict, data_dir: Optional[str] = None, logger=None):
        """
        Inisialisasi DatasetMerger.
        
        Args:
            config: Konfigurasi aplikasi
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.data_dir = Path(data_dir or config.get('data_dir', 'data'))
        self.logger = logger or get_logger("dataset_merger")
        
        self.logger.info(f"🔄 DatasetMerger diinisialisasi dengan data_dir: {self.data_dir}")
    
    def merge_datasets(
        self, 
        source_dirs: List[Union[str, Path]], 
        output_dir: Optional[Union[str, Path]] = None,
        prefix_filenames: bool = True,
        splits: List[str] = None,
        rename_duplicates: bool = True
    ) -> Dict[str, Dict[str, int]]:
        """
        Gabungkan beberapa dataset menjadi satu.
        
        Args:
            source_dirs: List direktori sumber
            output_dir: Direktori output (opsional)
            prefix_filenames: Apakah memberi prefix pada nama file
            splits: List split yang akan digabung (default: train, valid, test)
            rename_duplicates: Apakah memberi nama baru untuk file duplikat
            
        Returns:
            Statistik penggabungan per split
        """
        # Gunakan data_dir sebagai output jika tidak disediakan
        output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Gunakan semua split jika tidak dispesifikasi
        if splits is None:
            splits = DEFAULT_SPLITS
            
        # Buat direktori output
        for split in splits:
            for subdir in ['images', 'labels']:
                (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
                
        # Statistik penggabungan
        stats = {split: {'copied': 0, 'skipped': 0, 'renamed': 0, 'errors': 0} for split in splits}
        
        # Log informasi sumber
        self.logger.info(f"🔄 Menggabungkan {len(source_dirs)} dataset")
        for i, src_dir in enumerate(source_dirs):
            self.logger.info(f"   • Source {i+1}: {src_dir}")
            
        # Proses setiap direktori sumber
        for src_idx, src_dir in enumerate(source_dirs):
            src_dir = Path(src_dir)
            src_name = src_dir.name
            
            self.logger.info(f"🔄 Memproses dataset: {src_name}")
            
            # Prefix untuk nama file
            prefix = f"{src_name}_" if prefix_filenames else ""
            
            # Proses setiap split
            for split in splits:
                src_split_dir = src_dir / split
                
                # Skip jika direktori split tidak ada
                if not src_split_dir.exists():
                    self.logger.warning(f"⚠️ Split '{split}' tidak ditemukan di {src_name}")
                    continue
                    
                # Cek direktori gambar dan label
                src_images_dir = src_split_dir / 'images'
                src_labels_dir = src_split_dir / 'labels'
                
                if not src_images_dir.exists() or not src_labels_dir.exists():
                    self.logger.warning(f"⚠️ Struktur direktori tidak lengkap untuk split '{split}' di {src_name}")
                    continue
                
                # Direktori output untuk split ini
                dst_images_dir = output_dir / split / 'images'
                dst_labels_dir = output_dir / split / 'labels'
                
                # Daftar semua gambar
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(list(src_images_dir.glob(ext)))
                    
                # Proses setiap gambar dan label
                for img_path in tqdm(image_files, desc=f"Merging {split} from {src_name}"):
                    label_path = src_labels_dir / f"{img_path.stem}.txt"
                    
                    # Skip jika label tidak ada
                    if not label_path.exists():
                        stats[split]['skipped'] += 1
                        continue
                        
                    # Nama file dengan prefix
                    dst_img_name = f"{prefix}{img_path.name}"
                    dst_label_name = f"{prefix}{img_path.stem}.txt"
                    
                    # Path output
                    dst_img_path = dst_images_dir / dst_img_name
                    dst_label_path = dst_labels_dir / dst_label_name
                    
                    # Cek apakah file sudah ada
                    if dst_img_path.exists() or dst_label_path.exists():
                        if rename_duplicates:
                            # Buat nama baru
                            counter = 1
                            while True:
                                new_img_name = f"{prefix}{img_path.stem}_{counter}{img_path.suffix}"
                                new_label_name = f"{prefix}{img_path.stem}_{counter}.txt"
                                
                                new_img_path = dst_images_dir / new_img_name
                                new_label_path = dst_labels_dir / new_label_name
                                
                                if not new_img_path.exists() and not new_label_path.exists():
                                    dst_img_path = new_img_path
                                    dst_label_path = new_label_path
                                    break
                                    
                                counter += 1
                                
                            stats[split]['renamed'] += 1
                        else:
                            # Skip file
                            stats[split]['skipped'] += 1
                            continue
                            
                    try:
                        # Salin gambar dan label
                        shutil.copy2(img_path, dst_img_path)
                        shutil.copy2(label_path, dst_label_path)
                        stats[split]['copied'] += 1
                    except Exception as e:
                        self.logger.warning(f"⚠️ Gagal menyalin {img_path.name}: {str(e)}")
                        stats[split]['errors'] += 1
                        
        # Log hasil
        self.logger.success(f"✅ Penggabungan dataset selesai ke {output_dir}")
        for split in splits:
            s = stats[split]
            self.logger.info(
                f"   • Split {split}: {s['copied']} disalin, {s['renamed']} diberi nama baru, "
                f"{s['skipped']} dilewati, {s['errors']} error"
            )
            
        return stats
    
    def merge_splits(
        self, 
        source_dir: Union[str, Path], 
        output_dir: Optional[Union[str, Path]] = None,
        splits_to_merge: List[str] = None,
        output_split: str = 'merged'
    ) -> Dict[str, int]:
        """
        Gabungkan beberapa split dalam satu dataset menjadi satu split.
        
        Args:
            source_dir: Direktori dataset sumber
            output_dir: Direktori output (opsional)
            splits_to_merge: List split yang akan digabung (default: train, valid, test)
            output_split: Nama split output
            
        Returns:
            Statistik penggabungan
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir) if output_dir else self.data_dir
        
        # Gunakan semua split jika tidak dispesifikasi
        if splits_to_merge is None:
            splits_to_merge = DEFAULT_SPLITS
            
        # Buat direktori output
        output_images_dir = output_dir / output_split / 'images'
        output_labels_dir = output_dir / output_split / 'labels'
        
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistik
        stats = {'copied': 0, 'skipped': 0, 'errors': 0}
        
        self.logger.info(f"🔄 Menggabungkan splits: {', '.join(splits_to_merge)} menjadi '{output_split}'")
        
        # Proses setiap split
        for split in splits_to_merge:
            src_split_dir = source_dir / split
            
            # Skip jika direktori split tidak ada
            if not src_split_dir.exists():
                self.logger.warning(f"⚠️ Split '{split}' tidak ditemukan di {source_dir}")
                continue
                
            # Cek direktori gambar dan label
            src_images_dir = src_split_dir / 'images'
            src_labels_dir = src_split_dir / 'labels'
            
            if not src_images_dir.exists() or not src_labels_dir.exists():
                self.logger.warning(f"⚠️ Struktur direktori tidak lengkap untuk split '{split}'")
                continue
                
            # Daftar semua gambar
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(src_images_dir.glob(ext)))
                
            # Tambahkan prefix dengan nama split
            prefix = f"{split}_"
                
            # Proses setiap gambar dan label
            for img_path in tqdm(image_files, desc=f"Merging {split}"):
                label_path = src_labels_dir / f"{img_path.stem}.txt"
                
                # Skip jika label tidak ada
                if not label_path.exists():
                    stats['skipped'] += 1
                    continue
                    
                # Nama file dengan prefix
                dst_img_name = f"{prefix}{img_path.name}"
                dst_label_name = f"{prefix}{img_path.stem}.txt"
                
                # Path output
                dst_img_path = output_images_dir / dst_img_name
                dst_label_path = output_labels_dir / dst_label_name
                
                # Cek apakah file sudah ada
                if dst_img_path.exists() or dst_label_path.exists():
                    # Buat nama baru
                    counter = 1
                    while True:
                        new_img_name = f"{prefix}{img_path.stem}_{counter}{img_path.suffix}"
                        new_label_name = f"{prefix}{img_path.stem}_{counter}.txt"
                        
                        new_img_path = output_images_dir / new_img_name
                        new_label_path = output_labels_dir / new_label_name
                        
                        if not new_img_path.exists() and not new_label_path.exists():
                            dst_img_path = new_img_path
                            dst_label_path = new_label_path
                            break
                            
                        counter += 1
                
                try:
                    # Salin gambar dan label
                    shutil.copy2(img_path, dst_img_path)
                    shutil.copy2(label_path, dst_label_path)
                    stats['copied'] += 1
                except Exception as e:
                    self.logger.warning(f"⚠️ Gagal menyalin {img_path.name}: {str(e)}")
                    stats['errors'] += 1
                    
        # Log hasil
        total_files = stats['copied'] + stats['skipped'] + stats['errors']
        self.logger.success(
            f"✅ Penggabungan splits selesai ke {output_dir / output_split}:\n"
            f"   • Total: {total_files} file\n"
            f"   • Disalin: {stats['copied']}\n"
            f"   • Dilewati: {stats['skipped']}\n"
            f"   • Error: {stats['errors']}"
        )
        
        return stats