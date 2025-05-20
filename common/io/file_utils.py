"""
File: smartcash/common/io/file_utils.py
Deskripsi: Utilitas untuk operasi file seperti copy, move dan extract
"""

import os
import shutil
import zipfile
import cv2
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from tqdm.auto import tqdm

from smartcash.common.io.path_utils import ensure_dir, file_exists, standardize_path
from smartcash.common.threadpools import process_in_parallel, process_with_stats

def copy_file(
    src: Union[str, Path], 
    dst: Union[str, Path], 
    overwrite: bool = False,
    create_dirs: bool = True
) -> bool:
    """
    Copy file dari src ke dst.
    
    Args:
        src: Path sumber
        dst: Path tujuan
        overwrite: Flag untuk overwrite jika file tujuan sudah ada
        create_dirs: Buat direktori tujuan jika belum ada
        
    Returns:
        True jika berhasil, False jika gagal
    """
    src, dst = Path(src), Path(dst)
    
    # Validasi
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return False
    
    # Buat direktori jika perlu
    if create_dirs:
        ensure_dir(dst.parent)
    
    # Copy file
    try:
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False

def copy_files(
    source_dir: Union[str, Path], 
    target_dir: Union[str, Path],
    file_list: Optional[List[Union[str, Path]]] = None,
    patterns: Optional[List[str]] = None,
    flatten: bool = False,
    show_progress: bool = True,
    max_workers: Optional[int] = None,
    logger = None
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
        max_workers: Jumlah maksimum worker thread
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary dengan statistik penyalinan
    """
    from smartcash.common.io.path_utils import find_files
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Setup logger jika belum ada
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger()
        except ImportError:
            import logging
            logger = logging.getLogger("file_utils")
    
    # Validasi source directory
    if not source_dir.exists():
        logger.error(f"❌ Direktori sumber {source_dir} tidak ditemukan")
        return {'copied': 0, 'skipped': 0, 'errors': 0}
    
    # Buat direktori target
    ensure_dir(target_dir)
    
    # Dapatkan file list jika belum ada
    if file_list is None:
        file_list = find_files(source_dir, patterns, recursive=True)
    
    # Konversi ke Path objects
    file_list = [Path(f) if not isinstance(f, Path) else f for f in file_list]
    
    if not file_list:
        logger.warning(f"⚠️ Tidak ada file untuk disalin dari {source_dir}")
        return {'copied': 0, 'skipped': 0, 'errors': 0}
    
    # Fungsi untuk menyalin satu file
    def copy_single_file(file_path: Path) -> Dict[str, int]:
        try:
            # Tentukan path tujuan
            if flatten:
                dest_path = target_dir / file_path.name
            else:
                rel_path = file_path.relative_to(source_dir)
                dest_path = target_dir / rel_path
            
            # Buat direktori tujuan
            if not dest_path.parent.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Salin file jika belum ada
            if not dest_path.exists():
                shutil.copy2(file_path, dest_path)
                return {'copied': 1, 'skipped': 0, 'errors': 0}
            else:
                return {'copied': 0, 'skipped': 1, 'errors': 0}
        except Exception as e:
            logger.debug(f"⚠️ Error saat menyalin {file_path}: {str(e)}")
            return {'copied': 0, 'skipped': 0, 'errors': 1}
    
    # Process files in parallel
    stats = process_with_stats(
        file_list,
        copy_single_file,
        max_workers=max_workers,
        desc="📋 Menyalin file" if show_progress else None,
        show_progress=show_progress
    )
    
    # Default stats jika tidak ada
    stats.setdefault('copied', 0)
    stats.setdefault('skipped', 0)
    stats.setdefault('errors', 0)
    
    # Log results
    if logger:
        logger.info(
            f"✅ Penyalinan selesai: {stats['copied']} disalin, "
            f"{stats['skipped']} dilewati, {stats['errors']} error"
        )
    
    return stats

def move_files(
    source_dir: Union[str, Path], 
    target_dir: Union[str, Path],
    file_list: Optional[List[Union[str, Path]]] = None,
    patterns: Optional[List[str]] = None,
    flatten: bool = False,
    overwrite: bool = False,
    show_progress: bool = True,
    max_workers: Optional[int] = None,
    logger = None
) -> Dict[str, int]:
    """
    Pindahkan file dari direktori sumber ke direktori target.
    
    Args:
        source_dir: Direktori sumber
        target_dir: Direktori target
        file_list: Daftar file yang akan dipindahkan (opsional)
        patterns: Daftar pattern file yang akan dipindahkan (opsional)
        flatten: Apakah meratakan struktur direktori
        overwrite: Timpa file yang sudah ada
        show_progress: Tampilkan progress bar
        max_workers: Jumlah maksimum worker thread
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary dengan statistik pemindahan
    """
    from smartcash.common.io.path_utils import find_files
    
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Setup logger jika belum ada
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger()
        except ImportError:
            import logging
            logger = logging.getLogger("file_utils")
    
    # Validasi source directory
    if not source_dir.exists():
        logger.error(f"❌ Direktori sumber {source_dir} tidak ditemukan")
        return {'moved': 0, 'skipped': 0, 'errors': 0}
    
    # Buat direktori target
    ensure_dir(target_dir)
    
    # Dapatkan file list jika belum ada
    if file_list is None:
        file_list = find_files(source_dir, patterns, recursive=True)
    
    # Konversi ke Path objects
    file_list = [Path(f) if not isinstance(f, Path) else f for f in file_list]
    
    if not file_list:
        logger.warning(f"⚠️ Tidak ada file untuk dipindahkan dari {source_dir}")
        return {'moved': 0, 'skipped': 0, 'errors': 0}
    
    # Fungsi untuk memindahkan satu file
    def move_single_file(file_path: Path) -> Dict[str, int]:
        try:
            # Tentukan path tujuan
            if flatten:
                dest_path = target_dir / file_path.name
            else:
                rel_path = file_path.relative_to(source_dir)
                dest_path = target_dir / rel_path
            
            # Buat direktori tujuan
            if not dest_path.parent.exists():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Periksa jika file tujuan sudah ada
            if dest_path.exists():
                if overwrite:
                    dest_path.unlink()
                else:
                    return {'moved': 0, 'skipped': 1, 'errors': 0}
            
            # Pindahkan file
            shutil.move(str(file_path), str(dest_path))
            return {'moved': 1, 'skipped': 0, 'errors': 0}
        except Exception as e:
            logger.debug(f"⚠️ Error saat memindahkan {file_path}: {str(e)}")
            return {'moved': 0, 'skipped': 0, 'errors': 1}
    
    # Process files in parallel - terbatas untuk move operation
    stats = process_with_stats(
        file_list,
        move_single_file,
        max_workers=max_workers if max_workers else 4,  # Limit workers for move
        desc="📋 Memindahkan file" if show_progress else None,
        show_progress=show_progress
    )
    
    # Default stats jika tidak ada
    stats.setdefault('moved', 0)
    stats.setdefault('skipped', 0)
    stats.setdefault('errors', 0)
    
    # Log results
    if logger:
        logger.info(
            f"✅ Pemindahan selesai: {stats['moved']} dipindahkan, "
            f"{stats['skipped']} dilewati, {stats['errors']} error"
        )
    
    return stats

def backup_directory(
    source_dir: Union[str, Path], 
    suffix: Optional[str] = None,
    backup_dir: Optional[Union[str, Path]] = None,
    logger = None
) -> Optional[Path]:
    """
    Buat backup direktori.
    
    Args:
        source_dir: Direktori yang akan di-backup
        suffix: Suffix untuk nama direktori backup (opsional)
        backup_dir: Direktori untuk menaruh backup (opsional)
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Path ke direktori backup atau None jika gagal
    """
    source_path = Path(source_dir)
    
    # Setup logger jika belum ada
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger()
        except ImportError:
            import logging
            logger = logging.getLogger("file_utils")
    
    # Validasi
    if not source_path.exists():
        logger.warning(f"⚠️ Direktori sumber {source_dir} tidak ditemukan")
        return None
    
    # Generate timestamp jika suffix tidak ada
    suffix = suffix or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tentukan path backup
    if backup_dir:
        backup_parent = Path(backup_dir)
        ensure_dir(backup_parent)
        backup_path = backup_parent / f"{source_path.name}_backup_{suffix}"
    else:
        # Default: buat di direktori yang sama
        backup_path = source_path.parent / f"{source_path.name}_backup_{suffix}"
    
    # Tangani jika backup path sudah ada
    counter = 1
    original_backup_path = backup_path
    while backup_path.exists():
        backup_path = original_backup_path.parent / f"{original_backup_path.name}_{counter}"
        counter += 1
    
    # Lakukan backup
    try:
        shutil.copytree(source_path, backup_path)
        logger.info(f"✅ Direktori berhasil dibackup ke: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Gagal membuat backup {source_path}: {str(e)}")
        return None

def extract_zip(
    zip_path: Union[str, Path], 
    output_dir: Union[str, Path],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    remove_zip: bool = False,
    show_progress: bool = True,
    logger = None
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
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        Dictionary dengan statistik ekstraksi
    """
    zip_path = Path(zip_path)
    output_dir = Path(output_dir)
    
    # Setup logger jika belum ada
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger()
        except ImportError:
            import logging
            logger = logging.getLogger("file_utils")
    
    # Validasi
    if not zip_path.exists():
        logger.error(f"❌ File zip {zip_path} tidak ditemukan")
        return {'extracted': 0, 'skipped': 0, 'errors': 0}
    
    if not zipfile.is_zipfile(zip_path):
        logger.error(f"❌ File {zip_path} bukan file zip yang valid")
        return {'extracted': 0, 'skipped': 0, 'errors': 0}
    
    # Buat direktori output
    ensure_dir(output_dir)
    
    # Fungsi bantuan filter pattern
    def matches_pattern(filename: str, patterns: List[str]) -> bool:
        """Check if filename matches any pattern in the list."""
        import fnmatch
        return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)
    
    # Statistik
    stats = {'extracted': 0, 'skipped': 0, 'errors': 0}
    
    # Ekstrak file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Daftar semua file dalam zip
        all_files = zip_ref.infolist()
        
        # Filter berdasarkan patterns
        files_to_extract = []
        for file_info in all_files:
            filename = file_info.filename
            
            # Skip direktori
            if filename.endswith('/'):
                continue
                
            # Periksa include patterns
            if include_patterns and not matches_pattern(filename, include_patterns):
                stats['skipped'] += 1
                continue
                
            # Periksa exclude patterns
            if exclude_patterns and matches_pattern(filename, exclude_patterns):
                stats['skipped'] += 1
                continue
                
            files_to_extract.append(file_info)
        
        # Setup progress bar
        total_size = sum(file_info.file_size for file_info in files_to_extract)
        pbar = None
        if show_progress:
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"📦 Mengekstrak {zip_path.name}")
        
        # Ekstrak file
        for file_info in files_to_extract:
            try:
                zip_ref.extract(file_info, output_dir)
                stats['extracted'] += 1
                
                # Update progress bar
                if pbar:
                    pbar.update(file_info.file_size)
            except Exception as e:
                logger.debug(f"⚠️ Error saat mengekstrak {file_info.filename}: {str(e)}")
                stats['errors'] += 1
        
        # Tutup progress bar
        if pbar:
            pbar.close()
    
    # Hapus file zip jika diminta dan tidak ada error
    if remove_zip and stats['errors'] == 0:
        try:
            zip_path.unlink()
            logger.info(f"🗑️ File zip {zip_path} dihapus")
        except Exception as e:
            logger.warning(f"⚠️ Gagal menghapus file zip {zip_path}: {str(e)}")
    
    # Log hasil
    logger.info(
        f"✅ Ekstraksi file zip selesai: {stats['extracted']} diekstrak, "
        f"{stats['skipped']} dilewati, {stats['errors']} error"
    )
    
    return stats

def find_corrupted_images(
    directory: Union[str, Path],
    recursive: bool = True,
    show_progress: bool = True,
    max_workers: Optional[int] = None,
    logger = None
) -> List[Path]:
    """
    Temukan gambar yang rusak dalam direktori.
    
    Args:
        directory: Direktori yang akan diperiksa
        recursive: Apakah memeriksa subdirektori secara rekursif
        show_progress: Tampilkan progress bar
        max_workers: Jumlah maksimum worker thread
        logger: Logger untuk mencatat aktivitas
        
    Returns:
        List path gambar yang rusak
    """
    from smartcash.common.io.path_utils import find_files
    
    directory = Path(directory)
    
    # Setup logger jika belum ada
    if logger is None:
        try:
            from smartcash.common.logger import get_logger
            logger = get_logger()
        except ImportError:
            import logging
            logger = logging.getLogger("file_utils")
    
    # Validasi
    if not directory.exists():
        logger.error(f"❌ Direktori {directory} tidak ditemukan")
        return []
    
    # Cari gambar
    image_patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    image_files = find_files(directory, image_patterns, recursive)
    
    if not image_files:
        logger.warning(f"⚠️ Tidak ada gambar ditemukan di {directory}")
        return []
    
    # Informasi jumlah file
    logger.info(f"🔍 Memeriksa {len(image_files)} gambar untuk korupsi")
    
    # Fungsi untuk memeriksa satu gambar
    def check_image(img_path: Path) -> Optional[Path]:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                return img_path
            return None
        except Exception:
            return img_path
    
    # Process in parallel
    desc = "🔍 Memeriksa integritas gambar" if show_progress else None
    results = process_in_parallel(
        image_files,
        check_image,
        max_workers=max_workers,
        desc=desc,
        show_progress=show_progress
    )
    
    # Filter hasil yang tidak None (gambar rusak)
    corrupted_images = [path for path in results if path is not None]
    
    # Log hasil
    logger.info(
        f"✅ Pemeriksaan selesai: {len(corrupted_images)} gambar rusak "
        f"dari total {len(image_files)} gambar"
    )
    
    return corrupted_images
