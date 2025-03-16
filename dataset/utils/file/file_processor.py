"""
File: smartcash/dataset/utils/file/file_processor.py
Deskripsi: Utilitas untuk memproses file dan direktori dalam dataset
"""

import os
import shutil
import zipfile
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_utils import DEFAULT_SPLITS


class FileProcessor:
    """Utilitas untuk pemrosesan file dataset."""
    
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
        self.logger = logger or get_logger("file_processor")
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
            
        # Konversi ekstensi ke lowercase
        extensions = [ext.lower() if not ext.startswith('.') else ext.lower() for ext in extensions]
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        counts = {ext: 0 for ext in extensions}
        total = 0
        
        # Hitung file
        for ext in extensions:
            for file in directory.glob(f'**/*{ext}'):
                if file.is_file():
                    counts[ext] += 1
                    total += 1
                    
        # Tambahkan total
        counts['total'] = total
        
        self.logger.info(f"📊 Jumlah file di {directory}: {total} file")
        for ext, count in counts.items():
            if ext != 'total':
                self.logger.info(f"   • {ext}: {count} file")
                
        return counts
    
    def copy_files(
        self, 
        source_dir: Union[str, Path], 
        target_dir: Union[str, Path],
        file_list: Optional[List[Union[str, Path]]] = None,
        extensions: Optional[List[str]] = None,
        flatten: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
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
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        
        if not source_dir.exists():
            self.logger.error(f"❌ Direktori sumber {source_dir} tidak ditemukan")
            return {'copied': 0, 'skipped': 0, 'errors': 0}
            
        # Buat direktori target jika belum ada
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Dapatkan daftar file yang akan disalin
        if file_list is None:
            file_list = []
            if extensions:
                for ext in extensions:
                    ext = ext if ext.startswith('.') else f'.{ext}'
                    file_list.extend(list(source_dir.glob(f'**/*{ext}')))
            else:
                file_list = list(source_dir.glob('**/*.*'))
        else:
            # Konversi ke objek Path
            file_list = [Path(f) if not isinstance(f, Path) else f for f in file_list]
            
        if not file_list:
            self.logger.warning(f"⚠️ Tidak ada file yang akan disalin dari {source_dir}")
            return {'copied': 0, 'skipped': 0, 'errors': 0}
            
        # Statistik penyalinan
        stats = {'copied': 0, 'skipped': 0, 'errors': 0}
        
        # Fungsi untuk menyalin satu file
        def copy_file(file_path):
            try:
                rel_path = file_path.relative_to(source_dir)
                
                if flatten:
                    dest_path = target_dir / file_path.name
                else:
                    dest_path = target_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not dest_path.exists():
                    shutil.copy2(file_path, dest_path)
                    return 'copied'
                else:
                    return 'skipped'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal menyalin {file_path}: {str(e)}")
                return 'error'
        
        # Progress bar
        desc = f"📋 Menyalin {len(file_list)} file" if show_progress else None
        pbar = tqdm(total=len(file_list), desc=desc) if show_progress else None
        
        # Salin file secara paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for file_path in file_list:
                futures.append(executor.submit(copy_file, file_path))
            
            # Process results
            for future in futures:
                result = future.result()
                stats[f'{result}{"s" if result == "error" else ""}'] += 1
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()
        
        self.logger.info(
            f"✅ Penyalinan file selesai:\n"
            f"   • Copied: {stats['copied']}\n"
            f"   • Skipped: {stats['skipped']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def move_files(
        self, 
        source_dir: Union[str, Path], 
        target_dir: Union[str, Path],
        file_list: Optional[List[Union[str, Path]]] = None,
        extensions: Optional[List[str]] = None,
        flatten: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
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
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        
        if not source_dir.exists():
            self.logger.error(f"❌ Direktori sumber {source_dir} tidak ditemukan")
            return {'moved': 0, 'skipped': 0, 'errors': 0}
            
        # Buat direktori target jika belum ada
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Dapatkan daftar file yang akan dipindahkan
        if file_list is None:
            file_list = []
            if extensions:
                for ext in extensions:
                    ext = ext if ext.startswith('.') else f'.{ext}'
                    file_list.extend(list(source_dir.glob(f'**/*{ext}')))
            else:
                file_list = list(source_dir.glob('**/*.*'))
        else:
            # Konversi ke objek Path
            file_list = [Path(f) if not isinstance(f, Path) else f for f in file_list]
            
        if not file_list:
            self.logger.warning(f"⚠️ Tidak ada file yang akan dipindahkan dari {source_dir}")
            return {'moved': 0, 'skipped': 0, 'errors': 0}
            
        # Statistik pemindahan
        stats = {'moved': 0, 'skipped': 0, 'errors': 0}
        
        # Fungsi untuk memindahkan satu file
        def move_file(file_path):
            try:
                rel_path = file_path.relative_to(source_dir)
                
                if flatten:
                    dest_path = target_dir / file_path.name
                else:
                    dest_path = target_dir / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not dest_path.exists():
                    shutil.move(file_path, dest_path)
                    return 'moved'
                else:
                    return 'skipped'
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal memindahkan {file_path}: {str(e)}")
                return 'error'
        
        # Progress bar
        desc = f"📋 Memindahkan {len(file_list)} file" if show_progress else None
        pbar = tqdm(total=len(file_list), desc=desc) if show_progress else None
        
        # Pindahkan file secara paralel
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for file_path in file_list:
                futures.append(executor.submit(move_file, file_path))
            
            # Process results
            for future in futures:
                result = future.result()
                stats[f'{result}{"s" if result == "error" else ""}'] += 1
                if pbar:
                    pbar.update(1)
        
        if pbar:
            pbar.close()
        
        self.logger.info(
            f"✅ Pemindahan file selesai:\n"
            f"   • Moved: {stats['moved']}\n"
            f"   • Skipped: {stats['skipped']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def extract_zip(
        self, 
        zip_path: Union[str, Path], 
        output_dir: Union[str, Path],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        remove_zip: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
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
        zip_path = Path(zip_path)
        output_dir = Path(output_dir)
        
        if not zip_path.exists():
            self.logger.error(f"❌ File zip {zip_path} tidak ditemukan")
            return {'extracted': 0, 'skipped': 0, 'errors': 0}
            
        if not zipfile.is_zipfile(zip_path):
            self.logger.error(f"❌ File {zip_path} bukan file zip yang valid")
            return {'extracted': 0, 'skipped': 0, 'errors': 0}
            
        # Buat direktori output jika belum ada
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistik ekstraksi
        stats = {'extracted': 0, 'skipped': 0, 'errors': 0}
        
        # Ekstrak file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            files = zip_ref.infolist()
            
            # Filter file berdasarkan pola
            if include_patterns or exclude_patterns:
                filtered_files = []
                for file in files:
                    filename = file.filename
                    
                    # Cek apakah file diinclude
                    include_file = True
                    if include_patterns:
                        include_file = any(pattern in filename for pattern in include_patterns)
                    
                    # Cek apakah file diexclude
                    if exclude_patterns and include_file:
                        include_file = not any(pattern in filename for pattern in exclude_patterns)
                    
                    if include_file:
                        filtered_files.append(file)
                
                files = filtered_files
            
            # Progress bar
            total_size = sum(file.file_size for file in files)
            desc = f"📦 Mengekstrak {zip_path.name}" if show_progress else None
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) if show_progress else None
            
            # Ekstrak file
            for file in files:
                try:
                    if file.is_dir():
                        continue
                        
                    zip_ref.extract(file, output_dir)
                    stats['extracted'] += 1
                    
                    if pbar:
                        pbar.update(file.file_size)
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal mengekstrak {file.filename}: {str(e)}")
                    stats['errors'] += 1
        
        if pbar:
            pbar.close()
        
        # Hapus file zip jika diminta
        if remove_zip:
            try:
                zip_path.unlink()
                self.logger.info(f"🗑️ File zip {zip_path} dihapus")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menghapus file zip {zip_path}: {str(e)}")
        
        self.logger.info(
            f"✅ Ekstraksi file zip selesai:\n"
            f"   • Extracted: {stats['extracted']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def merge_splits(
        self, 
        source_dir: Union[str, Path], 
        target_dir: Union[str, Path],
        splits: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
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
        os.makedirs(target_dir / 'images', exist_ok=True)
        os.makedirs(target_dir / 'labels', exist_ok=True)
        
        # Default splits
        if splits is None:
            splits = DEFAULT_SPLITS
            
        # Statistik penggabungan
        stats = {'merged': 0, 'errors': 0}
        
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
            image_files = list(images_dir.glob('*.*'))
            if include_patterns:
                image_files = [f for f in image_files if any(pattern in f.name for pattern in include_patterns)]
                
            # Salin file
            for img_file in tqdm(image_files, desc=f"🔄 Menggabungkan split {split}", disable=not show_progress):
                try:
                    # Salin gambar dengan prefix split
                    new_img_name = f"{split}_{img_file.name}"
                    shutil.copy2(img_file, target_dir / 'images' / new_img_name)
                    
                    # Salin label jika ada
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    if label_file.exists():
                        new_label_name = f"{split}_{label_file.name}"
                        shutil.copy2(label_file, target_dir / 'labels' / new_label_name)
                        
                    stats['merged'] += 1
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal menggabungkan {img_file}: {str(e)}")
                    stats['errors'] += 1
        
        self.logger.info(
            f"✅ Penggabungan split dataset selesai:\n"
            f"   • Merged: {stats['merged']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def find_corrupted_images(
        self, 
        directory: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> List[Path]:
        """
        Temukan gambar yang rusak dalam direktori.
        
        Args:
            directory: Direktori yang akan diperiksa
            recursive: Apakah memeriksa subdirektori secara rekursif
            extensions: Daftar ekstensi file gambar (default: ['.jpg', '.jpeg', '.png', '.bmp'])
            show_progress: Tampilkan progress bar
            
        Returns:
            List path gambar yang rusak
        """
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")