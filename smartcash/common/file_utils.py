"""
File: smartcash/common/file_utils.py
Deskripsi: Utilitas terpadu untuk operasi file dan path dengan pendekatan DRY
"""

import os
import shutil
import glob
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from tqdm.auto import tqdm

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import process_in_parallel, process_with_stats

class FileUtils:
    """Utilitas terpadu untuk operasi file dan path."""
    
    def __init__(self, config: Dict = None, logger = None, num_workers: int = None):
        """
        Inisialisasi FileUtils.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
            num_workers: Jumlah worker untuk operasi paralel
        """
        self.config = config or {}
        self.logger = logger or get_logger("file_utils")
        self.num_workers = num_workers
        
    def ensure_dir(self, path: Union[str, Path]) -> Path:
        """Pastikan direktori ada, jika tidak buat."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def file_exists(self, path: Union[str, Path]) -> bool:
        """Cek apakah file ada."""
        return Path(path).exists()
    
    def file_size(self, path: Union[str, Path]) -> int:
        """Dapatkan ukuran file dalam bytes."""
        return Path(path).stat().st_size
    
    def format_size(self, size_bytes: int) -> str:
        """Format ukuran dalam bytes ke format yang lebih mudah dibaca."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} TB"
    
    def find_files(self, directory: Union[str, Path], patterns: List[str] = None, recursive: bool = False) -> List[Path]:
        """
        Cari file dalam direktori berdasarkan pattern.
        
        Args:
            directory: Direktori yang akan dicari
            patterns: List pattern file (default: ['*.*'])
            recursive: Cari di subdirektori
            
        Returns:
            List path file
        """
        patterns = patterns or ['*.*']
        directory = Path(directory)
        files = []
        
        for pattern in patterns:
            if recursive:
                files.extend(list(directory.glob(f"**/{pattern}")))
            else:
                files.extend(list(directory.glob(pattern)))
        
        return sorted(files)
    
    def find_image_files(self, directory: Union[str, Path], recursive: bool = False) -> List[Path]:
        """
        Cari file gambar dalam direktori.
        
        Args:
            directory: Direktori yang akan dicari
            recursive: Cari di subdirektori
            
        Returns:
            List path file gambar
        """
        from smartcash.dataset.utils.dataset_constants import IMG_EXTENSIONS
        patterns = [f"*.{ext[1:]}" for ext in IMG_EXTENSIONS]
        return self.find_files(directory, patterns, recursive)
    
    def find_matching_label(self, image_path: Union[str, Path], labels_dir: Union[str, Path]) -> Optional[Path]:
        """
        Cari file label yang sesuai dengan gambar.
        
        Args:
            image_path: Path file gambar
            labels_dir: Direktori label
            
        Returns:
            Path file label atau None jika tidak ditemukan
        """
        img_name = Path(image_path).stem
        label_path = Path(labels_dir) / f"{img_name}.txt"
        return label_path if label_path.exists() else None
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path], overwrite: bool = False) -> bool:
        """
        Copy file dari src ke dst.
        
        Args:
            src: Path sumber
            dst: Path tujuan
            overwrite: Flag untuk overwrite jika file tujuan sudah ada
            
        Returns:
            True jika berhasil, False jika gagal
        """
        src, dst = Path(src), Path(dst)
        
        # Cek jika file sumber ada
        if not src.exists(): return False
        
        # Cek jika file tujuan sudah ada dan overwrite = False
        if dst.exists() and not overwrite: return False
        
        # Buat direktori tujuan jika belum ada
        if not dst.parent.exists(): dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            self.logger.debug(f"⚠️ Gagal menyalin {src}: {str(e)}")
            return False
    
    def copy_files(
        self, 
        source_dir: Union[str, Path], 
        target_dir: Union[str, Path],
        file_list: Optional[List[Union[str, Path]]] = None,
        patterns: Optional[List[str]] = None,
        flatten: bool = False,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Salin file dari direktori sumber ke direktori target.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            file_list: Daftar file yang akan disalin (opsional)
            patterns: Daftar pattern file yang akan disalin (opsional)
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
            file_list = self.find_files(source_dir, patterns, recursive=True) if patterns else list(source_dir.glob('**/*.*'))
            
        # Konversi ke objek Path
        file_list = [Path(f) if not isinstance(f, Path) else f for f in file_list]
            
        if not file_list:
            self.logger.warning(f"⚠️ Tidak ada file yang akan disalin dari {source_dir}")
            return {'copied': 0, 'skipped': 0, 'errors': 0}
        
        # Fungsi untuk menyalin satu file
        def copy_file(file_path):
            try:
                rel_path = file_path.relative_to(source_dir)
                dst_path = target_dir / (file_path.name if flatten else rel_path)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                if not dst_path.exists():
                    shutil.copy2(file_path, dst_path)
                    return {'copied': 1}
                else:
                    return {'skipped': 1}
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal menyalin {file_path}: {str(e)}")
                return {'errors': 1}
        
        # Salin file secara paralel
        stats = process_with_stats(
            file_list,
            copy_file,
            max_workers=self.num_workers,
            desc="📋 Menyalin file",
            show_progress=show_progress
        )
        
        # Default stats jika kosong
        stats.setdefault('copied', 0)
        stats.setdefault('skipped', 0)
        stats.setdefault('errors', 0)
        
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
        patterns: Optional[List[str]] = None,
        flatten: bool = False,
        show_progress: bool = True,
        overwrite: bool = False
    ) -> Dict[str, int]:
        """
        Pindahkan file dari direktori sumber ke direktori target.
        
        Args:
            source_dir: Direktori sumber
            target_dir: Direktori target
            file_list: Daftar file yang akan dipindahkan (opsional)
            patterns: Daftar pattern file yang akan dipindahkan (opsional)
            flatten: Apakah meratakan struktur direktori
            show_progress: Tampilkan progress bar
            overwrite: Flag untuk overwrite jika file tujuan sudah ada
            
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
            file_list = self.find_files(source_dir, patterns, recursive=True) if patterns else list(source_dir.glob('**/*.*'))
            
        # Konversi ke objek Path
        file_list = [Path(f) if not isinstance(f, Path) else f for f in file_list]
            
        if not file_list:
            self.logger.warning(f"⚠️ Tidak ada file yang akan dipindahkan dari {source_dir}")
            return {'moved': 0, 'skipped': 0, 'errors': 0}
        
        # Fungsi untuk memindahkan satu file
        def move_file(file_path):
            try:
                rel_path = file_path.relative_to(source_dir)
                dst_path = target_dir / (file_path.name if flatten else rel_path)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                if dst_path.exists():
                    if overwrite:
                        dst_path.unlink()
                    else:
                        return {'skipped': 1}
                
                shutil.move(str(file_path), str(dst_path))
                return {'moved': 1}
            except Exception as e:
                self.logger.debug(f"⚠️ Gagal memindahkan {file_path}: {str(e)}")
                return {'errors': 1}
        
        # Pindahkan file secara paralel
        stats = process_with_stats(
            file_list,
            move_file,
            max_workers=self.num_workers,
            desc="📋 Memindahkan file",
            show_progress=show_progress
        )
        
        # Default stats jika kosong
        stats.setdefault('moved', 0)
        stats.setdefault('skipped', 0)
        stats.setdefault('errors', 0)
        
        self.logger.info(
            f"✅ Pemindahan file selesai:\n"
            f"   • Moved: {stats['moved']}\n"
            f"   • Skipped: {stats['skipped']}\n"
            f"   • Errors: {stats['errors']}"
        )
        
        return stats
    
    def backup_directory(self, source_dir: Union[str, Path], suffix: Optional[str] = None) -> Optional[Path]:
        """
        Buat backup direktori.
        
        Args:
            source_dir: Direktori yang akan di-backup
            suffix: Suffix untuk nama direktori backup (opsional)
            
        Returns:
            Path ke direktori backup atau None jika gagal
        """
        from datetime import datetime
        
        source_path = Path(source_dir)
        if not source_path.exists():
            self.logger.warning(f"⚠️ Direktori sumber tidak ditemukan: {source_path}")
            return None
        
        suffix = suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = source_path.parent
        backup_path = parent_dir / f"{source_path.name}_backup_{suffix}"
        
        # Apply increment untuk backup jika sudah ada
        i = 1
        while backup_path.exists():
            backup_path = parent_dir / f"{source_path.name}_backup_{suffix}_{i}"
            i += 1
        
        try:
            shutil.copytree(source_path, backup_path)
            self.logger.success(f"✅ Direktori berhasil dibackup ke: {backup_path}")
            return backup_path
        except Exception as e:
            self.logger.error(f"❌ Gagal membuat backup {source_path}: {str(e)}")
            return None
    
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
        import zipfile
        
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
                filtered_files = [file for file in files if any(pattern in file.filename for pattern in (include_patterns or [])) 
                                  and not any(pattern in file.filename for pattern in (exclude_patterns or []))]
                files = filtered_files if filtered_files else files
            
            # Progress bar
            total_size = sum(file.file_size for file in files)
            desc = f"📦 Mengekstrak {zip_path.name}" if show_progress else None
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) if show_progress else None
            
            # Ekstrak file
            for file in files:
                try:
                    if file.is_dir(): continue
                    zip_ref.extract(file, output_dir)
                    stats['extracted'] += 1
                    if pbar: pbar.update(file.file_size)
                except Exception as e:
                    self.logger.debug(f"⚠️ Gagal mengekstrak {file.filename}: {str(e)}")
                    stats['errors'] += 1
        
        if pbar: pbar.close()
        
        # Hapus file zip jika diminta
        if remove_zip and stats['errors'] == 0:
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
    
    def find_corrupted_images(
        self, 
        directory: Union[str, Path],
        recursive: bool = True,
        show_progress: bool = True
    ) -> List[Path]:
        """
        Temukan gambar yang rusak dalam direktori.
        
        Args:
            directory: Direktori yang akan diperiksa
            recursive: Apakah memeriksa subdirektori secara rekursif
            show_progress: Tampilkan progress bar
            
        Returns:
            List path gambar yang rusak
        """
        import cv2
        
        directory = Path(directory)
        
        if not directory.exists():
            self.logger.error(f"❌ Direktori {directory} tidak ditemukan")
            return []
            
        # Cari semua file gambar
        image_files = self.find_image_files(directory, recursive)
                
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada gambar ditemukan di {directory}")
            return []
            
        self.logger.info(f"🔍 Memeriksa {len(image_files)} gambar")
        
        # Fungsi untuk memeriksa satu gambar
        def check_image(img_path):
            try:
                img = cv2.imread(str(img_path))
                return img_path if img is None else None
            except Exception:
                return img_path
                
        # Periksa gambar secara paralel
        corrupted_images = [result for result in process_in_parallel(
            image_files,
            check_image,
            max_workers=self.num_workers,
            desc="🔍 Memeriksa integritas gambar",
            show_progress=show_progress
        ) if result is not None]
        
        self.logger.info(
            f"✅ Pemeriksaan selesai:\n"
            f"   • Total gambar: {len(image_files)}\n"
            f"   • Gambar rusak: {len(corrupted_images)}"
        )
        
        return corrupted_images

# Singleton instance
_file_utils = None

def get_file_utils(config: Dict = None, logger = None, num_workers: int = None) -> FileUtils:
    """
    Dapatkan instance FileUtils (singleton).
    
    Args:
        config: Konfigurasi aplikasi (opsional)
        logger: Logger kustom (opsional)
        num_workers: Jumlah worker untuk operasi paralel
        
    Returns:
        Instance FileUtils
    """
    global _file_utils
    if _file_utils is None:
        _file_utils = FileUtils(config, logger, num_workers)
    return _file_utils