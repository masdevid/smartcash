"""
File: smartcash/common/io/path_utils.py
Deskripsi: Utilitas untuk operasi path dan file system
"""

import os
import fnmatch
from pathlib import Path
from typing import List, Union, Optional, Callable

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Pastikan direktori ada, jika tidak buat.
    
    Args:
        path: Path direktori yang akan dibuat
        
    Returns:
        Path direktori yang dibuat
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def file_exists(path: Union[str, Path]) -> bool:
    """
    Cek apakah file ada.
    
    Args:
        path: Path file yang dicek
        
    Returns:
        Boolean yang menunjukkan keberadaan file
    """
    return Path(path).exists()

def file_size(path: Union[str, Path]) -> int:
    """
    Dapatkan ukuran file dalam bytes.
    
    Args:
        path: Path file
        
    Returns:
        Ukuran file dalam bytes
    """
    return Path(path).stat().st_size

def format_size(size_bytes: int) -> str:
    """
    Format ukuran dalam bytes ke format yang lebih mudah dibaca.
    
    Args:
        size_bytes: Ukuran dalam bytes
        
    Returns:
        String format ukuran yang mudah dibaca
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_file_extension(path: Union[str, Path]) -> str:
    """
    Dapatkan ekstensi file (lowercase).
    
    Args:
        path: Path file
        
    Returns:
        Ekstensi file tanpa dot
    """
    return Path(path).suffix.lower().lstrip('.')

def is_file_type(path: Union[str, Path], extensions: List[str]) -> bool:
    """
    Periksa apakah file memiliki ekstensi tertentu.
    
    Args:
        path: Path file
        extensions: List ekstensi yang valid (tanpa dot)
        
    Returns:
        Boolean yang menunjukkan apakah file memiliki ekstensi yang valid
    """
    ext = get_file_extension(path)
    return ext in [e.lower().lstrip('.') for e in extensions]

def standardize_path(path: Union[str, Path]) -> Path:
    """
    Standardisasi path ke absolute path.
    
    Args:
        path: Path untuk distandardisasi
        
    Returns:
        Path yang distandardisasi
    """
    return Path(path).expanduser().resolve()

def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> Path:
    """
    Dapatkan path relatif terhadap base.
    
    Args:
        path: Path yang akan direlativekan
        base: Path base
        
    Returns:
        Path relatif
    """
    return Path(path).relative_to(Path(base))

def list_dir_recursively(
    directory: Union[str, Path], 
    filter_func: Optional[Callable[[Path], bool]] = None
) -> List[Path]:
    """
    List direktori secara rekursif dengan filter opsional.
    
    Args:
        directory: Direktori untuk di-list
        filter_func: Fungsi filter untuk file (opsional)
        
    Returns:
        List path file
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    result = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if filter_func is None or filter_func(file_path):
                result.append(file_path)
    
    return result

def find_files(
    directory: Union[str, Path], 
    patterns: Optional[List[str]] = None, 
    recursive: bool = False
) -> List[Path]:
    """
    Cari file dalam direktori berdasarkan pattern.
    
    Args:
        directory: Direktori yang akan dicari
        patterns: List pattern file (default: ['*.*'])
        recursive: Cari di subdirektori
        
    Returns:
        List path file
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    patterns = patterns or ['*.*']
    result = []
    
    if recursive:
        for pattern in patterns:
            result.extend(list(directory.glob(f"**/{pattern}")))
    else:
        for pattern in patterns:
            result.extend(list(directory.glob(pattern)))
    
    return sorted(result)

def find_directories(
    directory: Union[str, Path], 
    patterns: Optional[List[str]] = None, 
    recursive: bool = False
) -> List[Path]:
    """
    Cari direktori dalam direktori berdasarkan pattern.
    
    Args:
        directory: Direktori yang akan dicari
        patterns: List pattern direktori (default: ['*'])
        recursive: Cari di subdirektori
        
    Returns:
        List path direktori
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    patterns = patterns or ['*']
    result = []
    
    # Fungsi filter untuk direktori
    def is_dir(path: Path) -> bool:
        return path.is_dir()
    
    if recursive:
        for pattern in patterns:
            # Temukan semua direktori yang cocok dengan pattern
            for item in directory.glob(f"**/{pattern}"):
                if item.is_dir():
                    result.append(item)
    else:
        for pattern in patterns:
            # Temukan direktori di level saat ini
            for item in directory.glob(pattern):
                if item.is_dir():
                    result.append(item)
    
    return sorted(result)

def file_matches_pattern(file_path: Union[str, Path], patterns: List[str]) -> bool:
    """
    Cek apakah file cocok dengan pattern.
    
    Args:
        file_path: Path file
        patterns: List pattern
        
    Returns:
        Boolean yang menunjukkan kecocokan
    """
    filename = Path(file_path).name
    return any(fnmatch.fnmatch(filename, pattern) for pattern in patterns)

def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Hitung ukuran total direktori dalam bytes.
    
    Args:
        directory: Path direktori
        
    Returns:
        Ukuran direktori dalam bytes
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
        
    total_size = 0
    for path in directory.glob('**/*'):
        if path.is_file():
            total_size += path.stat().st_size
            
    return total_size

def get_file_count(directory: Union[str, Path], recursive: bool = True) -> int:
    """
    Hitung jumlah file dalam direktori.
    
    Args:
        directory: Path direktori
        recursive: Cari di subdirektori
        
    Returns:
        Jumlah file
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
        
    if recursive:
        return len(list(directory.glob('**/*')))
    else:
        return len([f for f in directory.iterdir() if f.is_file()])

def get_directory_count(directory: Union[str, Path], recursive: bool = True) -> int:
    """
    Hitung jumlah direktori dalam direktori.
    
    Args:
        directory: Path direktori
        recursive: Cari di subdirektori
        
    Returns:
        Jumlah direktori
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
        
    if recursive:
        return len([d for d in directory.glob('**/*') if d.is_dir()])
    else:
        return len([d for d in directory.iterdir() if d.is_dir()])

def create_unique_filename(
    directory: Union[str, Path], 
    basename: str, 
    extension: str, 
    max_attempts: int = 1000
) -> Path:
    """
    Buat nama file unik dengan menambahkan counter jika diperlukan.
    
    Args:
        directory: Direktori untuk file
        basename: Nama dasar file
        extension: Ekstensi file
        max_attempts: Jumlah maksimum percobaan
        
    Returns:
        Path file unik
    """
    directory = Path(directory)
    ensure_dir(directory)
    
    extension = extension.lstrip('.')
    
    # Coba nama asli dulu
    filename = f"{basename}.{extension}"
    path = directory / filename
    
    if not path.exists():
        return path
    
    # Tambahkan counter
    for i in range(1, max_attempts + 1):
        filename = f"{basename}_{i}.{extension}"
        path = directory / filename
        
        if not path.exists():
            return path
    
    # Jika masih gagal, gunakan timestamp
    import time
    timestamp = int(time.time())
    filename = f"{basename}_{timestamp}.{extension}"
    return directory / filename

def get_parent_directory(path: Union[str, Path], levels: int = 1) -> Path:
    """
    Dapatkan direktori induk dari path.
    
    Args:
        path: Path yang akan dicari induknya
        levels: Jumlah level naik
        
    Returns:
        Path direktori induk
    """
    path = Path(path)
    result = path
    
    for _ in range(levels):
        result = result.parent
    
    return result