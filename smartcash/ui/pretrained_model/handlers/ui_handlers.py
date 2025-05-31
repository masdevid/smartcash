"""
File: smartcash/ui/pretrained_model/handlers/ui_handlers.py
Deskripsi: Handler UI untuk komponen model pretrained
"""

from typing import Dict, Any
from IPython.display import display, HTML, clear_output

from smartcash.ui.utils.constants import ICONS, COLORS
from smartcash.common.logger import get_logger
from smartcash.ui.pretrained_model.services.process_orchestrator import process_download_sync
from smartcash.ui.pretrained_model.utils.logger_utils import clear_log_output

logger = get_logger(__name__)

def handle_reset_ui_button(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol reset UI yang membersihkan log dan progress.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
    """
    # Pastikan status_panel tersedia
    status_panel = ui_components.get('status')
    if status_panel:
        status_panel.clear_output(wait=True)
        with status_panel:
            display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_success_bg']}; 
                     color:{COLORS['alert_success_text']}; border-radius:4px; margin:5px 0;
                     border-left:4px solid {COLORS['alert_success_text']}">
                <p style="margin:5px 0">{ICONS.get('success', '✅')} UI berhasil direset</p>
            </div>"""))
    
    # Reset progress tracking
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, "UI telah direset", show_progress=False)
    
    # Bersihkan log output
    log_output = ui_components.get('log')
    if log_output:
        clear_log_output(log_output)
        # Tambahkan pesan log baru
        if 'log_message' in ui_components and callable(ui_components['log_message']):
            ui_components['log_message'](f"{ICONS.get('cleanup', '🧹')} Log telah dibersihkan", "info")
    
    # Notify via observer jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            observer_manager.notify('UI_RESET', None, {
                'message': "UI telah direset",
                'timestamp': __import__('time').time()
            })
        except Exception:
            pass  # Silent fail untuk observer notification

def handle_download_sync_button(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol download dan sinkronisasi model pretrained dengan UI yang responsif.
    
    Args:
        b: Button widget yang dipicu
        ui_components: Dictionary berisi komponen UI
    """
    # Pastikan status_panel tersedia
    status_panel = ui_components.get('status')
    if not status_panel:
        logger.warning(f"{ICONS.get('warning', '⚠️')} Status panel tidak tersedia")
        return
    
    # Dapatkan informasi model
    model_info = ui_components.get('model_info', {})
    models_dir = ui_components.get('models_dir', '/content/models')
    drive_models_dir = ui_components.get('drive_models_dir', '/content/drive/MyDrive/SmartCash/models')
    
    # Tampilkan status processing dengan emoji kontekstual dan informasi lokasi model
    status_panel.clear_output(wait=True)
    with status_panel:
        display(HTML(f"""<div style="padding:10px; background-color:{COLORS['alert_info_bg']}; 
                 color:{COLORS['alert_info_text']}; border-radius:4px; margin:5px 0;
                 border-left:4px solid {COLORS['alert_info_text']}">
            <p style="margin:5px 0">{ICONS.get('processing', '⏳')} Memeriksa dan memproses model pretrained...</p>
            <p style="margin:5px 0"><b>Lokasi model:</b></p>
            <ul style="margin:5px 0">
                <li>Lokal: <code>{models_dir}</code></li>
                <li>Google Drive: <code>{drive_models_dir}</code></li>
            </ul>
        </div>"""))
    
    # Reset progress tracking dengan parameter show_progress=True untuk menampilkan progress bar yang bergerak
    if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
        ui_components['reset_progress_bar'](0, "Inisialisasi proses download model...", show_progress=True)
    
    # Buka log accordion secara otomatis jika tersedia
    if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
        ui_components['log_accordion'].selected_index = 0  # Expand log accordion
    
    # Nonaktifkan tombol selama proses berjalan
    b.disabled = True
    b.description = "Sedang Memproses..."
    b.icon = ICONS.get('processing', '⏳')
    
    # Tambahkan observer manager jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager and hasattr(observer_manager, 'notify'):
        try:
            observer_manager.notify('MODEL_PROCESS_START', None, {
                'message': "Memulai proses download dan sinkronisasi model",
                'timestamp': __import__('time').time()
            })
        except Exception:
            pass  # Silent fail untuk observer notification
    
    # Jalankan proses download dan sync dalam thread terpisah agar UI tetap responsif
    def run_process():
        try:
            # Jalankan proses download dan sync
            process_download_sync(ui_components)
            
            # Notify complete via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('MODEL_PROCESS_COMPLETE', None, {
                        'message': "Proses download dan sinkronisasi model selesai",
                        'timestamp': __import__('time').time()
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
        except Exception as e:
            logger.error(f"Error dalam proses download dan sinkronisasi: {str(e)}")
            
            # Notify error via observer
            if observer_manager and hasattr(observer_manager, 'notify'):
                try:
                    observer_manager.notify('MODEL_PROCESS_ERROR', None, {
                        'message': f"Error dalam proses download dan sinkronisasi: {str(e)}",
                        'timestamp': __import__('time').time(),
                        'error': str(e)
                    })
                except Exception:
                    pass  # Silent fail untuk observer notification
        finally:
            # Aktifkan kembali tombol setelah proses selesai
            b.disabled = False
            b.description = "Download & Sync Model"
            b.icon = ICONS.get('download', '📥')
    
    # Jalankan proses download langsung (tanpa threading)
    run_process()
