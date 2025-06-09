"""
File: smartcash/dataset/preprocessor/utils/file_scanner.py
Deskripsi: Modul untuk melakukan scanning direktori dan file
"""
import os
from pathlib import Path
from typing import List, Set, Optional

class FileScanner:
    """Kelas untuk melakukan scanning direktori dan file"""
    
    def __init__(self):
        """Inisialisasi FileScanner"""
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.supported_label_extensions = {'.txt'}
    
    def scan_directory(self, directory: Path, extensions: Optional[Set[str]] = None) -> List[Path]:
        """Scan direktori untuk file dengan ekstensi tertentu
        
        Args:
            directory: Path ke direktori yang akan di-scan
            extensions: Set ekstensi file yang dicari (contoh: {'.jpg', '.png'})
                      Jika None, kembalikan semua file
                      
        Returns:
            List Path ke file yang sesuai
        """
        if not directory.exists() or not directory.is_dir():
            return []
            
        file_paths = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if extensions is None or file_path.suffix.lower() in extensions:
                    file_paths.append(file_path)
        
        return sorted(file_paths)
    
    def find_matching_label(self, image_path: Path, label_dir: Path) -> Optional[Path]:
        """Cari file label yang sesuai dengan nama file gambar
        
        Args:
            image_path: Path ke file gambar
            label_dir: Direktori yang berisi file label
            
        Returns:
            Path ke file label yang sesuai atau None jika tidak ditemukan
        """
        if not label_dir.exists() or not label_dir.is_dir():
            return None
            
        # Format: nama_file_sama_dengan_gambar.txt
        label_path = label_dir / f"{image_path.stem}.txt"
        
        if label_path.exists() and label_path.is_file():
            return label_path
            
        return None
    
    def find_matching_image(self, label_path: Path, image_dir: Path) -> Optional[Path]:
        """Cari file gambar yang sesuai dengan nama file label
        
        Args:
            label_path: Path ke file label
            image_dir: Direktori yang berisi file gambar
            
        Returns:
            Path ke file gambar yang sesuai atau None jika tidak ditemukan
        """
        if not image_dir.exists() or not image_dir.is_dir():
            return None
            
        # Coba berbagai ekstensi gambar yang didukung
        for ext in self.supported_image_extensions:
            image_path = image_dir / f"{label_path.stem}{ext}"
            if image_path.exists() and image_path.is_file():
                return image_path
                
        return None
    
    def find_image_label_pairs(self, image_dir: Path, label_dir: Path) -> List[tuple[Path, Optional[Path]]]:
        """Temukan pasangan gambar dan label yang sesuai
        
        Args:
            image_dir: Direktori yang berisi file gambar
            label_dir: Direktori yang berisi file label
            
        Returns:
            List tuple berisi (path_gambar, path_label) atau (path_gambar, None) jika tidak ada label
        """
        image_files = self.scan_directory(image_dir, self.supported_image_extensions)
        pairs = []
        
        for img_path in image_files:
            label_path = self.find_matching_label(img_path, label_dir)
            pairs.append((img_path, label_path))
            
        return pairs
    
    def filter_valid_pairs(self, pairs: List[tuple[Path, Optional[Path]]]) -> List[tuple[Path, Path]]:
        """Filter pasangan yang valid (memiliki gambar dan label)
        
        Args:
            pairs: List tuple berisi (path_gambar, path_label)
            
        Returns:
            List tuple berisi (path_gambar, path_label) yang keduanya valid
        """
        return [(img, lbl) for img, lbl in pairs if lbl is not None and img.exists() and lbl.exists()]
    
    def get_unique_directories(self, file_paths: List[Path]) -> List[Path]:
        """Dapatkan daftar direktori unik dari kumpulan path file
        
        Args:
            file_paths: List path file
            
        Returns:
            List path direktori unik
        """
        dirs = set()
        for file_path in file_paths:
            if file_path.is_file():
                dirs.add(file_path.parent)
        return sorted(dirs)
