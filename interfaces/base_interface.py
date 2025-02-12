# File: src/interfaces/base_interface.py
# Author: Alfrida Sabar
# Deskripsi: Interface dasar untuk manajemen data dan interaksi pengguna

from pathlib import Path
from typing import Optional, Dict, List, Tuple
from termcolor import colored
from utils.logging import ColoredLogger

class BaseInterface:
    """Interface dasar untuk manajemen data dengan utilitas interaktif"""
    def __init__(self):
        self.logger = ColoredLogger(self.__class__.__name__)

    def prompt(self, message: str, default: Optional[str] = None, 
              color: str = 'yellow') -> str:
        """
        Tampilkan prompt dengan nilai default dan warna
        
        Args:
            message: Pesan prompt
            default: Nilai default jika tidak ada input
            color: Warna teks (default: yellow)
            
        Returns:
            Input pengguna atau nilai default
        """
        if default:
            message = f"{message} (default: {default}): "
        else:
            message = f"{message}: "
            
        value = input(colored(message, color)) or default
        return value

    def confirm(self, message: str) -> bool:
        """
        Tampilkan konfirmasi yes/no
        
        Args:
            message: Pesan konfirmasi
            
        Returns:
            True jika konfirmasi positif (y/Y)
        """
        response = self.prompt(f"{message} (y/n)", default='n')
        return response.lower() == 'y'

    def show_error(self, message: str):
        """
        Tampilkan pesan error dengan format yang konsisten
        
        Args:
            message: Pesan error
        """
        self.logger.error(message)

    def show_success(self, message: str):
        """
        Tampilkan pesan sukses dengan format yang konsisten
        
        Args:
            message: Pesan sukses
        """
        self.logger.info(message)

    def display_table(self, headers: List[str], rows: List[List[str]]):
        """
        Tampilkan data dalam format tabel
        
        Args:
            headers: List judul kolom
            rows: List baris data
        """
        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(str(cell)))

        # Print headers
        header_str = " | ".join(
            colored(h.ljust(w), 'cyan') 
            for h, w in zip(headers, widths)
        )
        print(header_str)
        print("-" * len(header_str))

        # Print rows
        for row in rows:
            print(" | ".join(str(cell).ljust(w) 
                           for cell, w in zip(row, widths)))

    def display_progress(self, current: int, total: int, 
                        prefix: str = '', suffix: str = ''):
        """
        Tampilkan progress bar sederhana
        
        Args:
            current: Nilai progres saat ini
            total: Nilai total
            prefix: Teks sebelum progress bar
            suffix: Teks setelah progress bar
        """
        percent = int(100.0 * current / total)
        bar_length = 50
        filled_length = int(bar_length * current / total)
        
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        print(f'\r{prefix} [{bar}] {percent}% {suffix}', end='')
        
        if current == total:
            print()

    def format_size(self, size_bytes: int) -> str:
        """
        Format ukuran file ke format yang mudah dibaca
        
        Args:
            size_bytes: Ukuran dalam bytes
            
        Returns:
            String ukuran yang diformat (e.g., "1.23 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    def validate_path(self, path: Path, 
                     create: bool = False, 
                     is_dir: bool = True) -> bool:
        """
        Validasi path dengan opsi pembuatan
        
        Args:
            path: Path yang akan divalidasi
            create: Buat direktori jika belum ada
            is_dir: True jika path adalah direktori
            
        Returns:
            True jika path valid
        """
        try:
            if not path.exists():
                if create and is_dir:
                    path.mkdir(parents=True)
                    return True
                return False
            
            return path.is_dir() if is_dir else path.is_file()
            
        except Exception as e:
            self.show_error(f"Error validating path: {str(e)}")
            return False