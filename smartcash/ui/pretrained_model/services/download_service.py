"""
File: smartcash/ui/pretrained_model/services/download_service.py
Deskripsi: Layanan untuk mengunduh model pretrained untuk SmartCash
"""

from typing import Dict, Any, Callable, Optional, Union
from pathlib import Path
import tempfile
import urllib.request
import concurrent.futures

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui

logger = get_logger(__name__)

def download_with_progress(url: str, target_path: Union[str, Path], log_func: Callable, 
                           ui_components: Dict[str, Any], model_idx: int, total_models: int) -> None:
    """
    Download file dengan progress tracking menggunakan ThreadPoolExecutor agar tidak blocking
    
    Args:
        url: URL sumber file
        target_path: Path tujuan file
        log_func: Fungsi untuk logging pesan
        ui_components: Komponen UI dengan progress bar
        model_idx: Indeks model dalam daftar
        total_models: Total jumlah model
    """
    # Konversi target_path ke Path jika string
    if isinstance(target_path, str):
        target_path = Path(target_path)
    
    # Buat temporary file untuk download
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    
    # Fungsi untuk melakukan download secara asynchronous
    def download_task():
        try:
            # Dapatkan ukuran file
            file_size = 0
            try:
                with urllib.request.urlopen(url) as response:
                    file_size = int(response.info().get('Content-Length', 0))
            except Exception as e:
                logger.warning(f"⚠️ Tidak dapat mendapatkan ukuran file: {str(e)}")
                file_size = 0
            
            # Fungsi untuk melaporkan progress download
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                
                # Update progress tracking di UI jika tersedia
                if 'progress_bar' in ui_components and 'progress_label' in ui_components:
                    percent = min(100, int(100 * downloaded / total_size if total_size > 0 else 0))
                    message = f"Mengunduh {target_path.name}: {percent}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)"
                    
                    # Jika ada update_progress_bar, gunakan itu
                    if 'update_progress_ui' in ui_components and callable(ui_components['update_progress_ui']):
                        ui_components['update_progress_ui'](
                            ui_components,
                            model_idx + (percent/100), 
                            total_models, 
                            message
                        )
                    else:
                        # Update komponen secara langsung
                        if hasattr(ui_components['progress_bar'], 'value'):
                            ui_components['progress_bar'].value = model_idx + (percent/100)
                            ui_components['progress_bar'].max = total_models
                        if hasattr(ui_components['progress_label'], 'value'):
                            ui_components['progress_label'].value = message
            
            # Download file
            urllib.request.urlretrieve(url, temp_path, reporthook=report_progress)
            
            # Setelah download selesai, pindahkan ke target
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.rename(target_path)
            
            # Log sukses
            size_mb = target_path.stat().st_size / (1024 * 1024)
            log_func(f"{target_path.name} berhasil diunduh (<span style='color:{COLORS.get('alert_success_text', '#155724')}'>{size_mb:.1f} MB</span>)", "success")
            
            return True
            
        except Exception as e:
            error_msg = f"Gagal mengunduh {target_path.name}: {str(e)}"
            logger.error(error_msg)
            log_func(error_msg, "error")
            
            if temp_path.exists():
                temp_path.unlink()  # Hapus file temporary jika terjadi error
            
            return False
    
    # Mulai download dalam thread terpisah menggunakan ThreadPoolExecutor
    # ThreadPoolExecutor cocok untuk I/O bound tasks seperti download
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(download_task)
        
    # Return langsung tanpa menunggu download selesai
    return None
