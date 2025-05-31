"""
File: smartcash/ui/pretrained_model/services/process_orchestrator.py
Deskripsi: Orchestrator untuk proses download dan sinkronisasi model pretrained
"""

import os
import time
from typing import Dict, Any, Optional
from pathlib import Path
from IPython.display import display, HTML
from enum import Enum

from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger, log_message
from smartcash.ui.pretrained_model.utils.download_utils import get_models_to_download, check_model_exists
from smartcash.ui.pretrained_model.utils.progress import update_progress_ui
from smartcash.ui.pretrained_model.services.subprocess_download import DownloadProcess
from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.ui.pretrained_model.config.model_config import get_all_models, get_model_info_for_download

# Definisi tahapan proses sebagai Enum untuk tracking progress
class ProcessStage(Enum):
    INIT = (0, "Inisialisasi dan persiapan")
    CHECK_MODELS = (20, "Memeriksa model yang tersedia")
    DOWNLOAD_YOLO = (50, "Download model YOLOv5")
    DOWNLOAD_EFFICIENTNET = (70, "Download model EfficientNet")
    COMPLETE = (100, "Proses selesai")
    ERROR = (-1, "Error dalam proses")
    
    def __init__(self, progress: int, description: str):
        self.progress = progress
        self.description = description

# Gunakan logger dari utils
logger = get_module_logger()

def update_stage(ui_components: Dict[str, Any], stage: ProcessStage) -> None:
    """
    Update tahapan proses dan progress bar
    
    Args:
        ui_components: Komponen UI untuk interaksi
        stage: Tahapan proses yang akan diupdate
    """
    # Dapatkan progress bar
    progress_bar = ui_components.get('progress_bar')
    
    # Update progress bar jika tersedia
    if progress_bar and hasattr(progress_bar, 'value'):
        # Set progress value
        progress_bar.value = stage.progress
        
        # Update description jika tersedia
        if hasattr(progress_bar, 'description'):
            progress_bar.description = stage.description
    
    # Log update stage
    log_message(ui_components, f"🔄 {stage.description} ({stage.progress}%)", "info")

def process_reset(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Proses reset model pretrained (menghapus semua model)
    
    Args:
        ui_components: Komponen UI untuk interaksi
        config: Konfigurasi untuk proses reset
    """
    # Dapatkan komponen UI
    status_panel = ui_components.get('status_panel')
    progress_bar = ui_components.get('progress_bar')
    log_accordion = ui_components.get('log_accordion')
    
    # Pastikan log accordion terbuka
    if log_accordion and hasattr(log_accordion, 'open'):
        log_accordion.open = True
        
    # Fungsi helper untuk update status panel
    def update_status(message: str, level: str = "info") -> None:
        if status_panel:
            icon = "⏳" if level == "info" else "✅" if level == "success" else "⚠️" if level == "warning" else "❌"
            status_panel.value = f"<h3>{icon} {message}</h3>"
    
    # Update status awal
    update_status("Menghapus semua model...")
        
    # Reset progress bar
    if progress_bar and hasattr(progress_bar, 'reset'):
        progress_bar.reset(0, "Menghapus model...")
    
    # Update progress stage
    update_stage(ui_components, ProcessStage.INIT)
    
    # Log start
    log_message(ui_components, "🗑️ Menghapus semua model pretrained", "info")
    
    try:
        # Dapatkan direktori model
        models_dir = config.get('models_dir', '')
        if not models_dir or not os.path.exists(models_dir):
            log_message(ui_components, "⚠️ Direktori model tidak ditemukan", "warning")
            update_status("Direktori model tidak ditemukan", "warning")
                
            # Reset progress bar
            if progress_bar and hasattr(progress_bar, 'reset'):
                progress_bar.reset(100, "Proses selesai")
                
            return
        
        # Update progress
        update_progress_ui(ui_components, 30, 100, "Menghapus file model...")
        
        # Update progress stage
        update_stage(ui_components, ProcessStage.CHECK_MODELS)
        
        # Dapatkan semua file model
        model_files = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
        
        # Hapus setiap file
        for i, file_name in enumerate(model_files):
            file_path = os.path.join(models_dir, file_name)
            os.remove(file_path)
            
            # Update progress
            progress = 30 + int((i / len(model_files)) * 70) if model_files else 100
            update_progress_ui(ui_components, progress, 100, f"Menghapus {file_name}...")
            
        # Log success
        log_message(ui_components, f"✅ Berhasil menghapus {len(model_files)} file model", "success")
        update_status("Semua model berhasil dihapus", "success")
            
        # Update progress stage
        update_stage(ui_components, ProcessStage.COMPLETE)
            
        # Reset progress bar
        if progress_bar and hasattr(progress_bar, 'reset'):
            progress_bar.reset(100, "Proses selesai")
            
    except Exception as e:
        # Log error
        log_message(ui_components, f"❌ Error saat menghapus model: {str(e)}", "error")
        update_status(f"Gagal menghapus model: {str(e)}", "error")
            
        # Update progress stage
        update_stage(ui_components, ProcessStage.ERROR)
            
        # Reset progress bar
        if progress_bar and hasattr(progress_bar, 'reset'):
            progress_bar.reset(100, "Proses gagal")

def process_download_sync(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Proses download dan sinkronisasi model pretrained
    
    Args:
        ui_components: Komponen UI untuk interaksi
        config: Konfigurasi untuk proses download
    """
    # Dapatkan komponen UI
    status_panel = ui_components.get('status_panel')
    progress_bar = ui_components.get('progress_bar')
    log_accordion = ui_components.get('log_accordion')
    
    # Pastikan log accordion terbuka
    if log_accordion and hasattr(log_accordion, 'open'):
        log_accordion.open = True
        
    # Fungsi helper untuk update status panel
    def update_status(message: str, level: str = "info") -> None:
        if status_panel:
            icon = "⏳" if level == "info" else "✅" if level == "success" else "⚠️" if level == "warning" else "❌"
            status_panel.value = f"<h3>{icon} {message}</h3>"
            
    # Update status awal
    update_status("Memeriksa model yang perlu diunduh...")
        
    # Reset progress bar
    if progress_bar and hasattr(progress_bar, 'reset'):
        progress_bar.reset(0, "Memeriksa model...")
    
    # Log start
    log_message(ui_components, "🔍 Memeriksa model yang perlu diunduh", "info")
    
    try:
        # Dapatkan model yang perlu diunduh
        models_to_download = get_models_to_download(config, ui_components)
        
        # Jika tidak ada model yang perlu diunduh
        if not models_to_download:
            log_message(ui_components, "✅ Semua model sudah tersedia di lokal", "success")
            update_status("Semua model sudah tersedia di lokal", "success")
                
            # Reset progress bar
            if progress_bar and hasattr(progress_bar, 'reset'):
                progress_bar.reset(100, "Semua model sudah tersedia")
                
            return
            
        # Update status untuk proses download
        update_status(f"Mengunduh {len(models_to_download)} model...")
            
        # Jalankan proses download
        download_process = DownloadProcess(ui_components)
        
        # Callback untuk update tahapan proses
        def on_model_download_start(model_name: str) -> None:
            # Update status panel
            update_status(f"Mengunduh model {model_name}...")
            
            # Update progress stage berdasarkan model
            if model_name == 'yolov5s':
                update_stage(ui_components, ProcessStage.DOWNLOAD_YOLO)
            elif 'efficientnet' in model_name.lower():
                update_stage(ui_components, ProcessStage.DOWNLOAD_EFFICIENTNET)
        
        # Tambahkan callback ke UI components
        ui_components['on_model_download_start'] = on_model_download_start
        
        # Mulai proses download
        download_process.start_download(models_to_download)
        
        # Tunggu proses selesai
        download_process.wait_until_complete()
        
        # Simpan referensi download_process untuk digunakan nanti
        ui_components['download_process'] = download_process
        
        # Cek keberhasilan download berdasarkan success_count dari download_process
        all_success = download_process.success_count == len(models_to_download)
        
        # Update status panel berdasarkan hasil
        if all_success:
            log_message(ui_components, f"✅ Semua model ({download_process.success_count}/{len(models_to_download)}) berhasil diunduh", "success")
            update_status("Semua model berhasil diunduh", "success")
        else:
            log_message(ui_components, f"⚠️ Beberapa model gagal diunduh ({download_process.success_count}/{len(models_to_download)} berhasil)", "warning")
            update_status(f"Beberapa model gagal diunduh ({download_process.success_count}/{len(models_to_download)} berhasil)", "warning")
        
        # Update progress stage
        update_stage(ui_components, ProcessStage.COMPLETE)
        
        # Reset progress bar
        if progress_bar and hasattr(progress_bar, 'reset'):
            progress_bar.reset(100, "Proses selesai")
            
    except Exception as e:
        # Log error
        log_message(ui_components, f"❌ Error saat proses download: {str(e)}", "error")
        update_status(f"Gagal mengunduh model: {str(e)}", "error")
        
        # Update progress stage
        update_stage(ui_components, ProcessStage.ERROR)
        
        # Reset progress bar
        if progress_bar and hasattr(progress_bar, 'reset'):
            progress_bar.reset(100, "Proses gagal")