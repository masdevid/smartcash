"""
File: smartcash/ui/pretrained_model/services/download_service.py
Deskripsi: Layanan untuk mengunduh model pretrained untuk SmartCash
"""

from typing import Dict, Any, Optional
from pathlib import Path
import requests
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor

from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger
from smartcash.ui.pretrained_model.utils.model_utils import ModelManager

# Gunakan logger dari utils dengan namespace yang konsisten
logger = get_module_logger("PRETRAINED_MODEL", "download_service")

def download_with_progress(url: str, save_path: Path, ui_components: Dict[str, Any] = None, 
                           min_size: int = 0, metadata: Dict[str, Any] = None) -> bool:
    """
    Download file dengan progress bar dan validasi ukuran.
    
    Args:
        url: URL file yang akan diunduh
        save_path: Path tujuan penyimpanan
        ui_components: Komponen UI untuk update progress
        min_size: Ukuran minimum file yang valid (bytes)
        metadata: Metadata model untuk disimpan setelah download berhasil
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    # Pastikan direktori tujuan ada
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Mulai download dengan streaming untuk progress bar
        logger.info(f"📥 Mengunduh {url} ke {save_path}")
        
        # Buat koneksi dengan streaming
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise exception jika status bukan 200 OK
        
        # Dapatkan ukuran file jika tersedia
        total_size = int(response.headers.get('content-length', 0))
        
        # Validasi ukuran file
        if total_size < min_size:
            logger.warning(f"⚠️ Ukuran file ({total_size} bytes) lebih kecil dari minimum ({min_size} bytes)")
        
        # Buat progress bar
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {save_path.name}")
        
        # Download langsung ke file tujuan (tanpa temporary file)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))
                    
                    # Update UI progress jika tersedia
                    if ui_components and 'progress_bar' in ui_components and total_size > 0:
                        progress_percent = min(progress_bar.n / total_size * 100, 100)
                        if hasattr(ui_components['progress_bar'], 'value'):
                            ui_components['progress_bar'].value = progress_percent
        
        progress_bar.close()
        
        # Validasi ukuran file hasil download
        actual_size = save_path.stat().st_size
        if actual_size < min_size:
            logger.error(f"❌ Download gagal: Ukuran file ({actual_size} bytes) lebih kecil dari minimum ({min_size} bytes)")
            return False
        
        # Simpan metadata jika tersedia
        if metadata:
            try:
                model_manager = ModelManager(save_path.parent)
                model_manager.save_metadata(save_path.name, metadata)
                logger.info(f"✅ Metadata untuk {save_path.name} berhasil disimpan")
            except Exception as e:
                logger.warning(f"⚠️ Gagal menyimpan metadata: {str(e)}")
        
        logger.success(f"✅ Download {save_path.name} selesai ({actual_size} bytes)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Download gagal: {str(e)}")
        # Hapus file yang tidak lengkap jika ada
        if save_path.exists():
            save_path.unlink()
        return False


def download_with_thread(url: str, save_path: Path, ui_components: Dict[str, Any] = None,
                        min_size: int = 0, metadata: Dict[str, Any] = None) -> None:
    """
    Download file dalam thread terpisah menggunakan ThreadPoolExecutor.
    Cocok untuk I/O bound tasks seperti download.
    
    Args:
        url: URL file yang akan diunduh
        save_path: Path tujuan penyimpanan
        ui_components: Komponen UI untuk update progress
        min_size: Ukuran minimum file yang valid (bytes)
        metadata: Metadata model untuk disimpan setelah download berhasil
    """
    with ThreadPoolExecutor() as executor:
        executor.submit(download_with_progress, url, save_path, ui_components, min_size, metadata)
