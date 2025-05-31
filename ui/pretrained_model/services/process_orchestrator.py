"""
File: smartcash/ui/pretrained_model/services/process_orchestrator.py
Deskripsi: Orchestrator untuk proses download dan sinkronisasi model pretrained
"""

import time
from typing import Dict, Any, Callable, List, Tuple
from pathlib import Path
from IPython.display import display, HTML
from enum import Enum

from smartcash.ui.pretrained_model.utils.download_utils import prepare_model_info, check_models_in_drive, get_models_to_download, check_model_exists
from smartcash.ui.pretrained_model.utils.logger_utils import get_module_logger, log_message
from smartcash.ui.pretrained_model.utils.model_utils import ModelManager
from smartcash.ui.pretrained_model.pretrained_initializer import is_drive_mounted, mount_drive
from smartcash.ui.pretrained_model.services.sync_service import sync_drive_to_local, sync_local_to_drive
from smartcash.ui.pretrained_model.services.subprocess_download import download_models
from smartcash.ui.utils.constants import ICONS, COLORS

# Definisi tahapan proses sebagai Enum untuk tracking progress
class ProcessStage(Enum):
    INIT = (0, "Inisialisasi dan persiapan")
    CHECK_DRIVE = (10, "Memeriksa Google Drive")
    CHECK_MODELS = (20, "Memeriksa model yang tersedia")
    SYNC_FROM_DRIVE = (30, "Sinkronisasi dari Drive ke lokal")
    DOWNLOAD_YOLO = (50, "Download model YOLOv5")
    DOWNLOAD_EFFICIENTNET = (70, "Download model EfficientNet")
    SYNC_TO_DRIVE = (90, "Sinkronisasi ke Google Drive")
    COMPLETE = (100, "Proses selesai")
    
    def __init__(self, progress: int, description: str):
        self.progress = progress
        self.description = description

# Gunakan logger dari utils
logger = get_module_logger()

def process_download_sync(ui_components: Dict[str, Any]) -> None:
    """
    Memproses download dan sinkronisasi model pretrained.
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    # Dapatkan komponen yang diperlukan dengan safe checking - one-liner style
    status_panel = ui_components.get('status')
    log_output = ui_components.get('log')
    models_dir = ui_components.get('models_dir', '/content/models')
    drive_models_dir = ui_components.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')
    
    # Inisialisasi ModelManager untuk pengelolaan metadata model
    model_manager = ModelManager(models_dir)
    
    # Inisialisasi progress tracking dengan tahapan proses
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](ProcessStage.INIT.progress, ProcessStage.INIT.description, show_progress=True)
        
    # Fungsi helper untuk update progress berdasarkan tahapan
    def update_stage(stage: ProcessStage):
        if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
            ui_components['progress_bar'].value = stage.progress
        if 'progress_label' in ui_components and hasattr(ui_components['progress_label'], 'value'):
            ui_components['progress_label'].value = stage.description
        log_ui(f"Tahap: {stage.description}", "info")
        return stage.progress
    
    # Bersihkan log output jika tersedia
    if log_output: log_output.clear_output(wait=True)
    
    # Alias untuk fungsi log_message dari utils untuk kemudahan penggunaan
    def log_ui(message: str, message_type='info'):
        log_message(ui_components, message, message_type)
    
    try:
        # Cek apakah di Colab dengan mencoba mengimpor google.colab
        in_colab = False
        try:
            import google.colab
            in_colab = True
        except ImportError:
            in_colab = False
        
        # Update progress: Memeriksa Google Drive
        update_stage(ProcessStage.CHECK_DRIVE)
        
        # Cek apakah Drive terpasang
        is_drive_mounted_val = is_drive_mounted()
        
        # Jika di Colab tapi Drive belum ter-mount, coba mount
        if in_colab and not is_drive_mounted_val:
            log_ui(f"{ICONS.get('sync', '🔄')} Mounting Google Drive...")
            success, message = mount_drive()
            log_ui(message)
            is_drive_mounted_val = is_drive_mounted()
            
        # Update progress: Memeriksa model yang tersedia
        update_stage(ProcessStage.CHECK_MODELS)
        
        # Cek apakah model sudah ada di Drive
        drive_models_exist = False
        if is_drive_mounted_val:
            drive_path = Path(drive_models_dir)
            if drive_path.exists():
                # Cek apakah file model ada di Drive
                yolo_path = drive_path / "yolov5s.pt"
                efficientnet_path = drive_path / "efficientnet_b4_ra2_288-7934f29e.pth"
                
                if yolo_path.exists() and efficientnet_path.exists():
                    log_ui(f"{ICONS.get('success', '✅')} Model ditemukan di Google Drive")
                    drive_models_exist = True
                    
                    # Update progress: Sinkronisasi dari Drive ke lokal
                    update_stage(ProcessStage.SYNC_FROM_DRIVE)
                    
                    # Sinkronkan dari Drive ke lokal
                    log_ui(f"{ICONS.get('download', '📥')} Menyinkronkan model dari Drive ke lokal...")
                    sync_drive_to_local(drive_models_dir, models_dir, log_ui, ui_components)
                    
                    # Update status
                    if status_panel:
                        status_panel.clear_output(wait=True)
                        with status_panel:
                            display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                    color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                                    border-left:4px solid {COLORS['alert_success_text']}">
                                <p style="margin:5px 0">{ICONS.get('success', '✅')} Model berhasil disinkronkan dari Drive</p>
                            </div>"""))
        
        # Jika model tidak ada di Drive, download dan sinkronkan
        if not drive_models_exist:
            log_ui(f"{ICONS.get('download', '📥')} Model tidak ditemukan di Drive, memulai download...")
            
            # Download model dengan progress bar - one-liner style
            # Definisi model yang akan diunduh
            model_info = [
                {
                    "name": "yolov5s.pt",
                    "url": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt",
                    "min_size": 10 * 1024 * 1024,  # 10MB
                    "idx": 0,
                    "id": "yolov5s_v6.2",
                    "version": "v6.2",
                    "source": "ultralytics/yolov5"
                },
                {
                    "name": "efficientnet-b4",
                    "url": "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_288-7934f29e.pth",
                    "path": Path(models_dir) / "efficientnet_b4_ra2_288-7934f29e.pth",
                    "min_size": 50 * 1024 * 1024,  # 50MB
                    "size": 75*1024*1024,
                    "idx": 1,
                    "id": "efficientnet_b4_timm-1.0",
                    "version": "timm-1.0",
                    "source": "timm"
                }
            ]
            
            # Buat direktori model jika belum ada
            Path(models_dir).mkdir(parents=True, exist_ok=True)
            
            # Inisialisasi dictionary untuk menyimpan informasi model
            model_data = {}
            download_list = []
            
            # Periksa model mana yang perlu diunduh
            for model in model_info:
                model_path = Path(models_dir) / model["name"]
                
                # Periksa apakah model sudah ada dan ukurannya sesuai
                if not model_path.exists() or model_path.stat().st_size < model["min_size"]:
                    log_ui(f"Model {model['name']} perlu diunduh", "info")
                    download_list.append({
                        "name": model["name"],
                        "url": model["url"],
                        "path": model_path
                    })
                    
                    # Tambahkan placeholder ke model_data
                    model_data[model["name"].split(".")[0]] = {
                        "path": str(model_path),
                        "size_mb": 0,  # Akan diupdate setelah download selesai
                        "status": "downloading"
                    }
                else:
                    # Model sudah ada, tambahkan ke model_data
                    model_info = {
                        "name": model["name"],
                        "path": str(model_path),
                        "size_mb": model_path.stat().st_size / (1024 * 1024)
                    }
                    model_data[model["name"].split(".")[0]] = {
                        "path": model_info["path"],
                        "size_mb": model_info["size_mb"],
                        "status": "ready"
                    }
            
            # Jalankan download jika ada model yang perlu diunduh
            if download_list:
                log_ui(f"Mengunduh {len(download_list)} model...", "download")
                
                # Tambahkan fungsi log dan update_stage ke ui_components
                ui_components['log_message'] = log_ui
                
                # Update progress berdasarkan model yang akan diunduh
                yolo_download = any(model['name'] == 'yolov5s' for model in download_list)
                efficientnet_download = any(model['name'] == 'efficientnet-b4' for model in download_list)
                
                if yolo_download:
                    update_stage(ProcessStage.DOWNLOAD_YOLO)
                elif efficientnet_download:
                    update_stage(ProcessStage.DOWNLOAD_EFFICIENTNET)
                
                # Tambahkan callback untuk update progress saat download
                def on_model_download_start(model_name):
                    if model_name == 'yolov5s':
                        update_stage(ProcessStage.DOWNLOAD_YOLO)
                    elif model_name == 'efficientnet-b4':
                        update_stage(ProcessStage.DOWNLOAD_EFFICIENTNET)
                
                ui_components['on_model_download_start'] = on_model_download_start
                
                # Gunakan subprocess download
                download_models(download_list, ui_components)
            else:
                log_ui("Semua model sudah tersedia", "success")
        
        # Update progress: Sinkronisasi ke Google Drive
        update_stage(ProcessStage.SYNC_TO_DRIVE)
        
        # Sinkronkan ke Drive jika tersedia
        if is_drive_mounted_val:
            log_ui(f"{ICONS.get('sync', '🔄')} Menyinkronkan model ke Google Drive...")
            model_info_dict = {"models": model_data}
            sync_local_to_drive(models_dir, drive_models_dir, model_info_dict, log_ui, ui_components)
        
        # Update status
        if status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                if len(model_data) == len(model_info):
                    display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                            color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                            border-left:4px solid {COLORS['alert_success_text']}">
                        <p style="margin:5px 0">{ICONS.get('success', '✅')} Semua model berhasil diunduh dan disinkronkan!</p>
                    </div>"""))
                else:
                    display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                            color:{COLORS['alert_warning_text']}; border-radius:4px; margin:5px 0;
                            border-left:4px solid {COLORS['alert_warning_text']}">
                        <p style="margin:5px 0">{ICONS.get('warning', '⚠️')} Beberapa model gagal diunduh. Silakan coba lagi.</p>
                    </div>"""))
            
            # Update status
            if status_panel:
                status_panel.clear_output(wait=True)
                with status_panel:
                    if len(model_data) == len(model_info):
                        display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                                color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                                border-left:4px solid {COLORS['alert_success_text']}">
                            <p style="margin:5px 0">{ICONS.get('success', '✅')} Semua model berhasil diunduh dan disinkronkan!</p>
                        </div>"""))
                    else:
                        display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                                color:{COLORS['alert_warning_text']}; border-radius:4px; margin:5px 0;
                                border-left:4px solid {COLORS['alert_warning_text']}">
                            <p style="margin:5px 0">{ICONS.get('warning', '⚠️')} Beberapa model gagal diunduh. Silakan coba lagi.</p>
                        </div>"""))
        
        # Update progress: Proses selesai
        update_stage(ProcessStage.COMPLETE)
        log_ui("Proses download dan sinkronisasi selesai", "success")
                    
    except Exception as e:
        logger.error(f"Error dalam proses download dan sinkronisasi: {str(e)}")
        
        # Update status error
        if status_panel:
            status_panel.clear_output(wait=True)
            with status_panel:
                display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                        color:{COLORS['alert_danger_text']}; border-radius:4px; margin:5px 0;
                        border-left:4px solid {COLORS['alert_danger_text']}">
                    <p style="margin:5px 0">{ICONS.get('error', '❌')} Error: {str(e)}</p>
                </div>"""))
