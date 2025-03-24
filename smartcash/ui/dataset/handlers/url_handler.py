"""
File: smartcash/ui/dataset/handlers/url_handler.py
Deskripsi: Handler untuk download dataset dari URL
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def download_from_url(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Download dataset dari URL.
    
    Args:
        ui_components: Dictionary komponen UI
        config: Konfigurasi download dataset
        
    Returns:
        Tuple (success, message)
    """
    logger = ui_components.get('logger')
    
    try:
        # Get konfigurasi
        url = config.get('url')
        output_dir = config.get('output_dir', 'data')
        
        # Pastikan direktori output ada
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Nama file dari URL
        file_name = os.path.basename(url)
        if not file_name:
            file_name = "dataset.zip"
        
        # Path file output
        file_path = output_path / file_name
        
        # Update progress
        _update_progress(ui_components, 10, f"Memulai download dari URL...")
        
        # Download file dengan progress
        success = _download_file_with_progress(url, file_path, ui_components)
        
        if not success:
            return False, "Gagal download file dari URL"
        
        # Proses file yang didownload
        _update_progress(ui_components, 80, "Memproses file dataset...")
        
        # Untuk file arsip, ekstrak kontennya
        if _is_archive_file(file_path):
            try:
                from smartcash.ui.dataset.handlers.zip_handler import extract_dataset
                result, message = extract_dataset(file_path, output_path, ui_components)
                return result, message
            except ImportError:
                # Fallback: Gunakan extract_archive sederhana
                extract_path = _extract_archive(file_path, output_path)
                if extract_path:
                    return True, f"Dataset berhasil didownload dan diekstrak ke {extract_path}"
                else:
                    return False, "Gagal mengekstrak file dataset"
        
        # Jika bukan arsip, selesai download
        return True, f"File berhasil didownload ke {file_path}"
    
    except Exception as e:
        if logger: logger.error(f"❌ Error saat download dari URL: {str(e)}")
        return False, f"Error saat download dari URL: {str(e)}"

def _download_file_with_progress(url: str, file_path: Path, ui_components: Dict[str, Any]) -> bool:
    """
    Download file dengan progress tracking.
    
    Args:
        url: URL file yang akan didownload
        file_path: Path tujuan file
        ui_components: Dictionary komponen UI
        
    Returns:
        Boolean menunjukkan keberhasilan
    """
    try:
        # Buka request untuk download
        with urllib.request.urlopen(url) as response:
            # Dapatkan total size jika tersedia
            file_size = int(response.info().get('Content-Length', -1))
            
            # Update progress
            if file_size > 0:
                _update_progress(ui_components, 20, f"Downloading file ({file_size//(1024*1024)} MB)...")
            else:
                _update_progress(ui_components, 20, "Downloading file (unknown size)...")
            
            # Dapatkan chunk size
            chunk_size = 1024 * 1024  # 1 MB
            
            # Download dengan progress
            downloaded = 0
            with open(file_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress jika file size tersedia
                    if file_size > 0:
                        progress = 20 + int(60 * downloaded / file_size)
                        if progress % 10 == 0:  # Update setiap 10%
                            _update_progress(ui_components, progress, f"Downloaded {downloaded//(1024*1024)} MB ({int(downloaded/file_size*100)}%)...")
        
        # Download selesai
        _update_progress(ui_components, 80, "Download selesai")
        return True
    
    except Exception as e:
        logger = ui_components.get('logger')
        if logger: logger.error(f"❌ Error saat download file: {str(e)}")
        return False

def _is_archive_file(file_path: Path) -> bool:
    """
    Cek apakah file adalah arsip.
    
    Args:
        file_path: Path file
        
    Returns:
        Boolean menunjukkan apakah file arsip
    """
    archive_exts = ['.zip', '.tar', '.gz', '.tar.gz', '.tgz']
    return file_path.suffix.lower() in archive_exts

def _extract_archive(file_path: Path, output_path: Path) -> Optional[Path]:
    """
    Extract file arsip.
    
    Args:
        file_path: Path file arsip
        output_path: Path direktori output
        
    Returns:
        Path direktori hasil ekstraksi atau None jika gagal
    """
    try:
        import zipfile
        import tarfile
        
        # Buat direktori ekstraksi
        extract_dir = output_path / file_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract berdasarkan tipe file
        if file_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_path.suffix.lower() in ['.tar', '.gz', '.tgz']:
            with tarfile.open(file_path) as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            # Tipe file tidak didukung
            return None
        
        return extract_dir
    
    except Exception:
        return None

def _update_progress(ui_components: Dict[str, Any], value: int, message: str) -> None:
    """
    Update progress bar dan progress tracker.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress
    """
    # Update progress bar
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    if progress_bar:
        progress_bar.value = value
    
    if progress_message:
        progress_message.value = message
    
    # Update progress tracker
    tracker_key = 'dataset_downloader_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)
        
    # Log ke logger
    logger = ui_components.get('logger')
    if logger and value % 20 == 0:  # Log setiap 20%
        logger.info(f"🔄 {message} ({value}%)")