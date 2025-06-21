"""
File: smartcash/common/utils/progress_utils.py
Deskripsi: Utilitas untuk progress reporting tanpa tqdm, menggunakan callback untuk UI
"""

import os
import sys
from typing import Callable, Optional, Dict, Any, Union
import time

class ProgressCallback:
    """
    Kelas utilitas untuk melaporkan progress tanpa tqdm.
    Menggunakan callback function untuk melaporkan progress ke UI.
    """
    
    def __init__(
        self, 
        total: int = 100, 
        desc: str = "Progress", 
        unit: str = "%",
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        logger = None
    ):
        """
        Inisialisasi progress callback.
        
        Args:
            total: Total langkah/ukuran untuk progress
            desc: Deskripsi progress
            unit: Unit pengukuran (%, B, KB, dll)
            callback: Fungsi callback untuk melaporkan progress ke UI
            logger: Logger untuk mencatat progress jika tidak ada callback
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.callback = callback
        self.logger = logger
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = 0
        self.min_update_interval = 0.1  # Minimal interval update (detik)
        
        # Laporkan progress awal
        self._report_progress(0)
    
    def update(self, increment: int = 1):
        """Update progress dengan increment tertentu"""
        self.current += increment
        
        # Batasi frekuensi update untuk menghindari flooding
        current_time = time.time()
        if (current_time - self.last_update_time) >= self.min_update_interval:
            self._report_progress(self.current)
            self.last_update_time = current_time
    
    def update_to(self, current: int):
        """Update progress ke nilai tertentu"""
        self.current = current
        
        # Batasi frekuensi update untuk menghindari flooding
        current_time = time.time()
        if (current_time - self.last_update_time) >= self.min_update_interval:
            self._report_progress(self.current)
            self.last_update_time = current_time
    
    def _report_progress(self, current: int):
        """Laporkan progress ke callback atau logger"""
        percentage = min(100, int(current / self.total * 100)) if self.total > 0 else 0
        elapsed = time.time() - self.start_time
        
        # Buat info progress
        progress_info = {
            "desc": self.desc,
            "current": current,
            "total": self.total,
            "percentage": percentage,
            "elapsed": elapsed,
            "unit": self.unit
        }
        
        # Panggil callback jika ada
        if self.callback:
            self.callback(progress_info)
        # Fallback ke logger jika tidak ada callback
        elif self.logger:
            self.logger.info(f"{self.desc}: {percentage}% ({current}/{self.total} {self.unit})")
    
    def __enter__(self):
        """Support untuk context manager"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finalisasi progress saat keluar dari context"""
        # Pastikan progress terakhir dilaporkan
        self._report_progress(self.current)


class DownloadProgressCallback(ProgressCallback):
    """
    Kelas khusus untuk melaporkan progress download.
    Kompatibel dengan urllib.request.urlretrieve.
    """
    
    def __init__(
        self, 
        desc: str = "Downloading", 
        unit: str = "B",
        unit_scale: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        logger = None
    ):
        """
        Inisialisasi download progress callback.
        
        Args:
            desc: Deskripsi download
            unit: Unit pengukuran (B, KB, dll)
            unit_scale: Skala unit otomatis (B -> KB -> MB)
            callback: Fungsi callback untuk melaporkan progress ke UI
            logger: Logger untuk mencatat progress jika tidak ada callback
        """
        super().__init__(total=100, desc=desc, unit=unit, callback=callback, logger=logger)
        self.unit_scale = unit_scale
        self.total_size = 0
        
    def update_to(self, blocknum: int, bs: int, tsize: int):
        """
        Update progress download, kompatibel dengan urllib.request.urlretrieve.
        
        Args:
            blocknum: Jumlah blok yang sudah didownload
            bs: Ukuran blok dalam bytes
            tsize: Total ukuran file dalam bytes
        """
        if tsize > 0:
            self.total = tsize
        
        current = blocknum * bs
        
        # Format unit yang sesuai
        if self.unit_scale:
            if current >= 1024*1024*1024:
                formatted_current = f"{current/1024/1024/1024:.2f} GB"
                formatted_total = f"{self.total/1024/1024/1024:.2f} GB"
            elif current >= 1024*1024:
                formatted_current = f"{current/1024/1024:.2f} MB"
                formatted_total = f"{self.total/1024/1024:.2f} MB"
            elif current >= 1024:
                formatted_current = f"{current/1024:.2f} KB"
                formatted_total = f"{self.total/1024:.2f} KB"
            else:
                formatted_current = f"{current} B"
                formatted_total = f"{self.total} B"
        else:
            formatted_current = f"{current} {self.unit}"
            formatted_total = f"{self.total} {self.unit}"
        
        # Update deskripsi dengan informasi ukuran
        self.desc = f"{self.desc.split(':')[0]}: {formatted_current}/{formatted_total}"
        
        # Update progress
        super().update_to(current)


def download_with_progress(
    url: str, 
    output_path: str, 
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    logger = None
):
    """
    Download file dengan progress reporting.
    
    Args:
        url: URL file yang akan didownload
        output_path: Path untuk menyimpan file
        callback: Fungsi callback untuk melaporkan progress ke UI
        logger: Logger untuk mencatat progress jika tidak ada callback
    
    Returns:
        Path file yang didownload
    """
    import urllib.request
    
    # Buat direktori jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Ekstrak nama file dari URL
    filename = url.split('/')[-1]
    
    # Download dengan progress callback
    with DownloadProgressCallback(
        desc=f"⬇️ Downloading {filename}", 
        unit="B", 
        unit_scale=True, 
        callback=callback,
        logger=logger
    ) as progress:
        urllib.request.urlretrieve(url, output_path, reporthook=progress.update_to)
    
    return output_path
