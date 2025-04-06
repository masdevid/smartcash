"""
File: smartcash/ui/dataset/download/handlers/url_handler.py
Deskripsi: Handler untuk download dataset dari URL
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

def process_url_download(ui_components: Dict[str, Any]) -> None:
    """
    Proses download dataset dari URL.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    # Ambil parameter dari UI components
    url = ui_components['url_input'].value.strip()
    output_dir = ui_components['output_dir'].value
    
    # Jalankan proses download
    try:
        # Dapatkan endpoint config
        from smartcash.ui.dataset.download.handlers.endpoint_handler import get_endpoint_config
        config = get_endpoint_config(ui_components)
        
        # Jalankan download dari URL
        success, message = download_from_url(config, ui_components)
        
        # Update UI berdasarkan hasil
        from smartcash.ui.utils.fallback_utils import show_status
        from smartcash.ui.utils.constants import ALERT_STYLES
        
        if success:
            # Sukses
            show_status(message, "success", ui_components)
            ui_components['status_panel'].value = f"""
            <div style="padding:10px; background-color:{ALERT_STYLES['success']['bg_color']}; 
                      color:{ALERT_STYLES['success']['text_color']}; border-radius:4px; margin:5px 0;
                      border-left:4px solid {ALERT_STYLES['success']['text_color']};">
                <p style="margin:5px 0">{ALERT_STYLES['success']['icon']} {message}</p>
            </div>
            """
        else:
            # Error
            show_status(message, "error", ui_components)
            ui_components['status_panel'].value = f"""
            <div style="padding:10px; background-color:{ALERT_STYLES['error']['bg_color']}; 
                      color:{ALERT_STYLES['error']['text_color']}; border-radius:4px; margin:5px 0;
                      border-left:4px solid {ALERT_STYLES['error']['text_color']};">
                <p style="margin:5px 0">{ALERT_STYLES['error']['icon']} {message}</p>
            </div>
            """
    except Exception as e:
        error_msg = f"Error saat download dari URL: {str(e)}"
        if logger:
            logger.error(f"❌ {error_msg}")
        
        # Update UI dengan error
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(error_msg, "error", ui_components)
    finally:
        # Reset UI setelah selesai
        _reset_ui_after_download(ui_components)

def download_from_url(config: Dict[str, Any], ui_components: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Download dataset dari URL.
    
    Args:
        config: Konfigurasi download
        ui_components: Dictionary komponen UI
        
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
                # Coba dapatkan service dataset
                from smartcash.ui.utils.fallback_utils import get_dataset_manager
                download_service = get_dataset_manager({'data': {'dir': output_dir}}, logger)
                
                if download_service:
                    # Gunakan service untuk import dari ZIP
                    result = download_service.import_from_zip(
                        zip_file=str(file_path),
                        target_dir=output_dir,
                        remove_zip=False,
                        show_progress=True
                    )
                    
                    stats = result.get('stats', {})
                    total_images = stats.get('total_images', 0)
                    return True, f"Dataset berhasil didownload dan diimport: {total_images} gambar"
            except ImportError:
                pass
                
            # Fallback: ekstrak file secara manual
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
            # Dapatkan total size
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
    tracker_key = 'download_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(value, message)
        
    # Log ke logger
    logger = ui_components.get('logger')
    if logger and value % 20 == 0:  # Log setiap 20%
        logger.info(f"🔄 {message} ({value}%)")
        
def _reset_ui_after_download(ui_components: Dict[str, Any]) -> None:
    """Reset UI setelah proses download selesai."""
    # Sembunyikan progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
    if 'progress_message' in ui_components:
        ui_components['progress_message'].layout.visibility = 'hidden'