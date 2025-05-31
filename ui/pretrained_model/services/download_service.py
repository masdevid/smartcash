"""
File: smartcash/ui/pretrained_model/services/download_service.py
Deskripsi: Layanan untuk mengunduh model pretrained untuk SmartCash
"""

from typing import Dict, Any, Callable, Optional, Union
from pathlib import Path
import urllib.request
import concurrent.futures
import os

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger, log_message
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui
from smartcash.ui.pretrained_model.utils.download_utils import check_model_exists
from smartcash.ui.pretrained_model.utils.model_utils import ModelManager

# Gunakan logger dari utils
logger = get_module_logger()

def download_with_progress(url: str, target_path: Union[str, Path], log_func: Callable, 
                           ui_components: Dict[str, Any], model_idx: int, total_models: int,
                           model_info: Dict[str, Any] = None) -> None:
    """
    Download file dengan progress tracking menggunakan ThreadPoolExecutor agar tidak blocking
    
    Args:
        url: URL sumber file
        target_path: Path tujuan file
        log_func: Fungsi untuk logging pesan
        ui_components: Komponen UI dengan progress bar
        model_idx: Indeks model dalam daftar
        total_models: Total jumlah model
        model_info: Informasi tambahan tentang model (opsional)
    """
    # Konversi target_path ke Path jika string
    if isinstance(target_path, str):
        target_path = Path(target_path)
    
    # Buat direktori target terlebih dahulu
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Fungsi untuk melakukan download langsung ke target path
    def download_task():
        try:
            # Inisialisasi ModelManager untuk metadata
            model_manager = ModelManager(str(target_path.parent))
            
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
                    # Hitung persentase dengan aman
                    if total_size > 0:
                        percent = min(100, int(100 * downloaded / total_size))
                    else:
                        percent = 0
                        
                    # Format pesan dengan ukuran yang diunduh
                    downloaded_mb = downloaded/(1024*1024)
                    total_mb = total_size/(1024*1024) if total_size > 0 else 0
                    message = f"Mengunduh {target_path.name}: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)"
                    
                    # Hitung progress global berdasarkan indeks model dan kemajuan saat ini
                    global_progress = (model_idx + (percent/100)) / total_models
                    
                    # Update UI dengan progress
                    update_progress_ui(ui_components, global_progress, message)
            
            # Download file langsung ke target path
            logger.info(f"🔄 Mulai mengunduh {target_path.name} dari {url}")
            urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
            
            # Update metadata model jika info tersedia
            if model_info:
                model_id = model_info.get('id', f"{target_path.name}_downloaded")
                version = model_info.get('version', '1.0')
                source = model_info.get('source', url)
                model_manager.update_model_metadata(target_path, model_id, version, source)
            
            # Log sukses dengan styling yang konsisten
            size_mb = target_path.stat().st_size / (1024 * 1024)
            success_msg = f"✅ {target_path.name} berhasil diunduh ({size_mb:.1f} MB)"
            logger.info(success_msg)
            log_func(success_msg, "success")
            
            # Update progress ke 100% untuk model ini
            global_progress = (model_idx + 1) / total_models
            update_progress_ui(ui_components, global_progress, f"Selesai mengunduh {target_path.name}")
            
            return True
            
        except Exception as e:
            error_msg = f"❌ Gagal mengunduh {target_path.name}: {str(e)}"
            logger.error(error_msg)
            log_func(error_msg, "error")
            
            # Update progress untuk menunjukkan error
            global_progress = (model_idx + 0.5) / total_models
            update_progress_ui(ui_components, global_progress, f"Error: {target_path.name}")
            
            return False
    
    # Mulai download dalam thread terpisah menggunakan ThreadPoolExecutor
    # ThreadPoolExecutor cocok untuk I/O bound tasks seperti download
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(download_task)
        
    # Return langsung tanpa menunggu download selesai
    return None
