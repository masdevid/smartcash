"""
File: smartcash/dataset/utils/file/file_processor.py
Deskripsi: Utilitas untuk memproses file dan direktori dalam dataset dengan pendekatan DRY
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.common.io import (
    find_corrupted_images, copy_files, 
    move_files, extract_zip, ensure_dir
)
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS
from smartcash.dataset.utils.data_utils import find_image_files

class FileProcessor:
    """Wrapper untuk operasi file dataset menggunakan common/file_utils."""
    
    def __init__(self, config: Dict = None, data_dir: Optional[str] = None, logger=None, num_workers: int = 4):
        """
        Inisialisasi FileProcessor.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            data_dir: Direktori utama data (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config or {}
        self.data_dir = Path(data_dir or self.config.get('data_dir', 'data'))
        self.logger = logger or get_logger()
        self.num_workers = num_workers
        
        self.logger.info(f"📂 FileProcessor diinisialisasi dengan data_dir: {self.data_dir}")
    
    def count_files(self, directory: Union[str, Path], extensions: List[str] = None) -> Dict[str, int]:
        """
        Hitung jumlah file dalam direktori berdasarkan ekstensi.
        
        Args:
            directory: Direktori yang akan dihitung
            extensions: Daftar ekstensi file (default: ['.jpg', '.jpeg', '.png', '.txt'])
            
        Returns:
            Dictionary dengan jumlah file per ekstensi
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.warning(f"⚠️ Direktori {directory} tidak ditemukan")
            return {}
            
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.txt']
        
        # Konversi ekstensi ke lowercase dan tambahkan titik jika perlu
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in extensions]
        
        # Inisialisasi dictionary hasil
        counts = {ext: 0 for ext in extensions}
        counts['total'] = 0
        
        # Lakukan penghitungan
        for ext in extensions:
            file_count = sum(1 for _ in directory.glob(f'**/*{ext}'))
            counts[ext] = file_count
            counts['total'] += file_count
        
        # Log hasil
        self.logger.info(f"📊 Jumlah file di {directory}: {counts['total']} file")
        for ext, count in counts.items():
            if ext != 'total':
                self.logger.info(f"   • {ext}: {count} file")
                
        return counts
    
    def copy_files(self, source_dir: Union[str, Path], target_dir: Union[str, Path], 
                   file_list: Optional[List[Union[str, Path]]] = None, 
                   extensions: Optional[List[str]] = None, 
                   flatten: bool = False, 
                   show_progress: bool = True) -> Dict[str, int]:
        """
        Salin file dari direktori sumber ke direktori target.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            file_list: Daftar file yang akan disalin (opsional)
            extensions: Daftar ekstensi file yang akan disalin (opsional)
            flatten: Apakah meratakan struktur direktori
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik penyalinan
        """
        # Konversi pattern extensions ke format yang diharapkan file_wrapper
        patterns = [f"*{ext}" for ext in extensions] if extensions else None
        
        # Gunakan fungsi dari file_wrapper
        stats = copy_files(
            source_dir=source_dir, 
            target_dir=target_dir, 
            file_list=file_list,
            patterns=patterns, 
            flatten=flatten, 
            show_progress=show_progress
        )
        
        return stats
    
    def move_files(self, source_dir: Union[str, Path], target_dir: Union[str, Path],
                  file_list: Optional[List[Union[str, Path]]] = None,
                  extensions: Optional[List[str]] = None,
                  flatten: bool = False,
                  show_progress: bool = True) -> Dict[str, int]:
        """
        Pindahkan file dari direktori sumber ke direktori target.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            file_list: Daftar file yang akan dipindahkan (opsional)
            extensions: Daftar ekstensi file yang akan dipindahkan (opsional)
            flatten: Apakah meratakan struktur direktori
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik pemindahan
        """
        # Konversi pattern extensions ke format yang diharapkan file_wrapper
        patterns = [f"*{ext}" for ext in extensions] if extensions else None
        
        # Gunakan fungsi dari file_wrapper
        stats = move_files(
            source_dir=source_dir, 
            target_dir=target_dir, 
            file_list=file_list,
            patterns=patterns, 
            flatten=flatten, 
            show_progress=show_progress
        )
        
        return stats
    
    def extract_zip(self, zip_path: Union[str, Path], output_dir: Union[str, Path],
                   include_patterns: Optional[List[str]] = None,
                   exclude_patterns: Optional[List[str]] = None,
                   remove_zip: bool = False,
                   show_progress: bool = True) -> Dict[str, int]:
        """
        Ekstrak file zip ke direktori output.
        
        Args:
            zip_path: Path ke file zip
            output_dir: Direktori output
            include_patterns: Pola file yang akan diinclude (opsional)
            exclude_patterns: Pola file yang akan diexclude (opsional)
            remove_zip: Hapus file zip setelah ekstraksi
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik ekstraksi
        """
        # Gunakan fungsi dari file_wrapper
        stats = extract_zip(
            zip_path=zip_path, 
            output_dir=output_dir, 
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns, 
            remove_zip=remove_zip, 
            show_progress=show_progress
        )
        
        return stats
    
    def merge_splits(self, source_dir: Union[str, Path], target_dir: Union[str, Path],
                    splits: Optional[List[str]] = None,
                    include_patterns: Optional[List[str]] = None,
                    show_progress: bool = True) -> Dict[str, int]:
        """
        Gabungkan beberapa split dataset ke dalam satu direktori.
        
        Args:
            source_dir: Direktori sumber yang berisi split
            target_dir: Direktori target untuk hasil penggabungan
            splits: Daftar split yang akan digabungkan (default: train, valid, test)
            include_patterns: Pola file yang akan diinclude (opsional)
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik penggabungan
        """
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        
        if not source_dir.exists():
            self.logger.error(f"❌ Direktori sumber {source_dir} tidak ditemukan")
            return {'merged': 0, 'errors': 0}
            
        # Buat direktori target jika belum ada
        ensure_dir(target_dir / 'images')
        ensure_dir(target_dir / 'labels')
        
        # Default splits
        if splits is None:
            splits = DEFAULT_SPLITS
            
        # Statistik penggabungan
        stats = {'merged': 0, 'skipped': 0, 'errors': 0}
        
        for split in splits:
            split_dir = source_dir / split
            
            if not split_dir.exists():
                self.logger.warning(f"⚠️ Split {split} tidak ditemukan di {source_dir}")
                continue
                
            images_dir = split_dir / 'images'
            labels_dir = split_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                self.logger.warning(f"⚠️ Direktori images/labels tidak lengkap di {split_dir}")
                continue
                
            # List semua file gambar
            image_files = find_image_files(images_dir) if not include_patterns else [f for f in find_image_files(images_dir) if any(pattern in f.name for pattern in include_patterns)]
                
            # Salin file dengan prefiks split menggunakan metode copy_file dari file_utils
            merge_stats = self._copy_with_prefix(image_files, labels_dir, target_dir, split, show_progress)
            
            # Update statistik
            for key in stats:
                stats[key] += merge_stats.get(key, 0)
        
        self.logger.info(
            f"✅ Penggabungan split dataset selesai:\n"
            f"   • Merged: {stats['merged']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def _copy_with_prefix(self, image_files: List[Path], labels_dir: Path, target_dir: Path, prefix: str, show_progress: bool) -> Dict[str, int]:
        """
        Salin file dengan prefiks ke direktori target.
        
        Args:
            image_files: Daftar file gambar
            labels_dir: Direktori label sumber
            target_dir: Direktori target
            prefix: Prefiks untuk nama file
            show_progress: Tampilkan progress bar
            
        Returns:
            Dictionary dengan statistik penyalinan
        """
        from tqdm import tqdm
        stats = {'merged': 0, 'skipped': 0, 'errors': 0}
        
        for img_file in tqdm(image_files, desc=f"🔄 Menggabungkan split {prefix}", disable=not show_progress):
            try:
                # Salin gambar dengan prefix split
                new_img_name = f"{prefix}_{img_file.name}"
                import shutil
                shutil.copy2(img_file, target_dir / 'images' / new_img_name)
                
                # Salin label jika ada
                label_file = labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    new_label_name = f"{prefix}_{label_file.name}"
                    shutil.copy2(label_file, target_dir / 'labels' / new_label_name)
                    
                stats['merged'] += 1
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal menggabungkan {img_file}: {str(e)}")
                stats['errors'] += 1
                
        return stats
    
    def find_corrupted_images(self, directory: Union[str, Path], recursive: bool = True, 
                             show_progress: bool = True) -> List[Path]:
        """
        Temukan gambar yang rusak dalam direktori.
        
        Args:
            directory: Direktori yang akan diperiksa
            recursive: Apakah memeriksa subdirektori secara rekursif
            show_progress: Tampilkan progress bar
            
        Returns:
            List path gambar yang rusak
        """
        # Gunakan fungsi dari file_wrapper
        return find_corrupted_images(directory, recursive, show_progress)